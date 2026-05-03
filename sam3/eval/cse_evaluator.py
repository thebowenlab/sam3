import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pickle

from sam3.train.data.mesh import MeshCatalog, create_mesh
from sam3.model.embedders import normalize_embeddings

logger = logging.getLogger(__name__)


# ============================================================================
# Utility: find closest mesh vertex for each pixel embedding
# ============================================================================

def squared_euclidean_distance_matrix(
    pts1: torch.Tensor, pts2: torch.Tensor
) -> torch.Tensor:
    """[M, D] x [K, D] -> [M, K] squared L2 distances."""
    edm = torch.mm(-2 * pts1, pts2.t())
    edm += (pts1 * pts1).sum(1, keepdim=True) + (pts2 * pts2).sum(
        1, keepdim=True
    ).t()
    return edm.contiguous()


def find_closest_vertices(
    pixel_embeddings: torch.Tensor,
    mesh_vertex_embeddings: torch.Tensor,
) -> torch.Tensor:
    """
    For each pixel embedding [P, D], find the nearest mesh vertex [K, D].
    Returns vertex indices [P] (values in 0..K-1).
    """
    edm = squared_euclidean_distance_matrix(pixel_embeddings, mesh_vertex_embeddings)
    return edm.argmin(dim=1)


# ============================================================================
# Metric 1: Per-Point GPS (OGPS) — per-instance, per-image
# ============================================================================

class PerPointGPSEvaluator:
    """
    Computes per-point Geodesic Point Similarity (GPS) for CSE predictions,
    mirroring DensePose's computeOgps_single_pair_cse.

    For each detected instance matched to a GT instance:
      1. Resample the predicted embedding [D, S, S] to the bbox size [D, H, W]
      2. At each GT-annotated point (px, py), extract the pixel embedding
      3. Find the closest mesh vertex by embedding distance
      4. Compute geodesic distance to the GT vertex
      5. Convert to GPS: exp(-d² / 2σ²)
      6. Average over all annotated points → per-instance OGPS

    The per-instance OGPS is then used as a matching score for COCO-style
    AP/AR computation.
    """

    def __init__(
        self,
        default_sigma: float = 0.255,
        out_dir: str = None,
    ):
        """
        Args:
            embedder: CSEEmbedder module (maps mesh_name -> [K, D] vertex embeddings)
            default_sigma: normalization constant σ for GPS.
                DensePose uses per-body-part sigmas for SMPL (0.107–0.351)
                and 0.255 as default for non-SMPL meshes.
        """
        self.default_sigma = default_sigma
        self.out_dir=out_dir

    @torch.no_grad()
    def compute_ogps_for_instance(
        self,
        pred_embedding: torch.Tensor,
        pred_mask: torch.Tensor,
        pred_bbox_xywh: torch.Tensor,
        gt_vertex_ids: torch.Tensor,
        mesh_name: str,
        gt_points_x: torch.Tensor,
        gt_points_y: torch.Tensor,
        gt_bbox_xywh: torch.Tensor,
        mesh_embedding: torch.Tensor,
        sigma: Optional[np.ndarray] = None,
    ) -> float:
        """
        Compute OGPS for a single predicted instance vs. a single GT instance.

        Args:
            pred_embedding: [D, S, S] predicted embedding feature map for this instance
            pred_coarse_segm: [C, S, S] coarse segmentation (C=2 for fg/bg)
            bbox_xywh: (x, y, w, h) absolute bbox of the detection
            gt_vertex_ids: [P] GT mesh vertex IDs for each annotated point (0-based)
            mesh_name: which mesh this instance uses (e.g. "smpl_27554")
            gt_points_xy: [P, 2] absolute (x, y) coords of GT annotated points
            sigma: [P] per-point normalization constants. If None, uses self.default_sigma.

        Returns:
            OGPS score in [0, 1]. Higher is better.
        """
        x, y, w, h = pred_bbox_xywh
        dw, dh = max(int(w), 1), max(int(h), 1)
        img_h, img_w = pred_mask.shape

        gt_bbox_xywh[0] *= img_w
        gt_bbox_xywh[2] *= img_w
        gt_bbox_xywh[1] *= img_h
        gt_bbox_xywh[3] *= img_h

        # Resize embedding to bbox pixel size: [D, S, S] -> [D, dh, dw]
        embedding_bbox = F.interpolate(
            pred_embedding.unsqueeze(0),
            (dh, dw),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)  # [D, dh, dw]

        # GT point coords: convert to bbox-relative pixel coords
        dy = int(pred_bbox_xywh[3])
        dx = int(pred_bbox_xywh[2])
        dp_x = gt_points_x * gt_bbox_xywh[2] / 255.0
        dp_y = gt_points_y * gt_bbox_xywh[3] / 255.0
        abs_px = (dp_x + gt_bbox_xywh[0]).to(torch.int)
        abs_py = (dp_y + gt_bbox_xywh[1]).to(torch.int)
        py = (abs_py - pred_bbox_xywh[1]).to(torch.int)
        px = (abs_px - pred_bbox_xywh[0]).to(torch.int)

        # Validity: points inside the bbox
        valid = (px >= 0) & (px < dw) & (py >= 0) & (py < dh)
        if not valid.any():
            return 0.0

        px_valid = px[valid]
        py_valid = py[valid]
        abs_px_valid = abs_px[valid]
        abs_py_valid = abs_py[valid]
        gt_vids_valid = gt_vertex_ids[valid]

        # Check foreground using the full-image boolean mask at absolute coords
        fg_mask = pred_mask[abs_py_valid, abs_px_valid]  # [J] boolean

        # Extract pixel embeddings at bbox-relative coords: [J, D]
        pixel_embs = embedding_bbox[:, py_valid, px_valid].t()  # [J, D]

        # Get mesh vertex embeddings: [K, D]
        pixel_embs = pixel_embs.to(mesh_embedding.device)

        # Find closest predicted vertices: [J]
        pred_vids = find_closest_vertices(pixel_embs, mesh_embedding)

        # Mark points outside foreground as invalid
        pred_vids[~fg_mask] = -1

        # Look up geodesic distances between GT and predicted vertices
        mesh = create_mesh(mesh_name, device="cpu")
        geodists = mesh.geodists  # [K, K]

        both_valid = fg_mask & (gt_vids_valid >= 0)
        if not both_valid.any():
            return 0.0

        gt_v = gt_vids_valid[both_valid].cpu()
        pred_v = pred_vids[both_valid].cpu()
        dists = geodists[gt_v, pred_v]  # [J_valid]

        # Compute GPS: exp(-d² / 2σ²)
        gps_values = torch.exp(-(dists ** 2) / (2 * self.default_sigma ** 2))

        return gps_values.mean().item() if len(gps_values) > 0 else 0.0

    @torch.no_grad()
    def evaluate_dataset(
        self,
        predictions: List[Dict],
        ground_truths: List[Dict],
        mesh_embeddings: Dict, 
        iou_thresholds: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        if iou_thresholds is None:
            iou_thresholds = np.linspace(0.5, 0.95, 10)

        # ---- Group by category (mesh_name) ----
        all_categories = sorted(
            set(g["mesh_name"] for g in ground_truths)
            | set(p["mesh_name"] for p in predictions)
        )
        all_categories = [c for c in all_categories if c]  # drop empty

        preds_by_cat = defaultdict(list)
        for p in predictions:
            preds_by_cat[p["mesh_name"]].append(p)

        gts_by_cat = defaultdict(list)
        for g in ground_truths:
            gts_by_cat[g["mesh_name"]].append(g)

        # ---- Compute AP per category, then average ----
        per_cat_aps = {}   # cat -> list of APs at each threshold
        per_cat_ars = {}   # cat -> list of ARs at each threshold

        matched_preds = []

        for cat in all_categories:
            cat_preds = preds_by_cat[cat]
            cat_gts = gts_by_cat[cat]

            # Group by image within this category
            preds_by_img = defaultdict(list)
            for p in cat_preds:
                preds_by_img[p["image_id"]].append(p)

            gts_by_img = defaultdict(list)
            for g in cat_gts:
                gts_by_img[g["image_id"]].append(g)

            all_img_ids = sorted(
                set(list(preds_by_img.keys()) + list(gts_by_img.keys()))
            )

            all_scores = []
            all_ogps = []
            n_gt_cat = 0

            for img_id in all_img_ids:
                dts = sorted(preds_by_img[img_id], key=lambda x: -x["score"])
                gts = gts_by_img[img_id]
                n_gt_cat += len(gts)

                if not dts or not gts:
                    # if there are no detections, it still hurts recall.
                    # if there are no gts, this image didn't have densepose annotations so we don't count against it.
                    continue

                # Compute OGPS matrix: [len(dts), len(gts)]
                ogps_matrix = np.zeros((len(dts), len(gts)))
                for i, dt in enumerate(dts):
                    for j, gt in enumerate(gts):
                        ogps_matrix[i, j] = self.compute_ogps_for_instance(
                            pred_embedding=dt["embedding"],
                            pred_mask=dt["mask"],
                            pred_bbox_xywh=dt["bbox"],
                            gt_vertex_ids=gt["dp_vertex"],
                            mesh_name=gt["mesh_name"],
                            gt_points_x=gt["dp_x"],
                            gt_points_y=gt["dp_y"],
                            gt_bbox_xywh=gt["bbox"],
                            mesh_embedding=mesh_embeddings[gt["mesh_name"]],
                        )

                # Greedy matching per detection (sorted by score desc)
                gt_matched = [False] * len(gts)
                for i, dt in enumerate(dts):
                    best_ogps = 0.0
                    best_j = -1
                    for j in range(len(gts)):
                        if gt_matched[j]:
                            continue
                        if ogps_matrix[i, j] > best_ogps:
                            best_ogps = ogps_matrix[i, j]
                            best_j = j
                    if best_j >= 0 and best_ogps > 0:
                        if self.out_dir is not None:
                            matched_preds.append(dt)
                        gt_matched[best_j] = True
                    all_scores.append(dt["score"])
                    all_ogps.append(best_ogps)

            # Skip categories with no GT
            if n_gt_cat == 0:
                continue

            # Compute AP at each threshold for this category
            all_scores = np.array(all_scores)
            all_ogps = np.array(all_ogps)
            sort_idx = np.argsort(-all_scores)
            all_ogps = all_ogps[sort_idx]

            cat_aps = []
            cat_ars = []
            for thresh in iou_thresholds:
                tp = np.cumsum(all_ogps >= thresh).astype(float)
                fp = np.cumsum(all_ogps < thresh).astype(float)
                recall = tp / n_gt_cat
                precision = tp / (tp + fp + 1e-10)

                # Monotonic decreasing precision
                for i in range(len(precision) - 2, -1, -1):
                    precision[i] = max(precision[i], precision[i + 1])

                # 101-point interpolated AP
                rec_thresholds = np.linspace(0, 1, 101)
                ap = 0.0
                for rt in rec_thresholds:
                    mask = recall >= rt
                    if mask.any():
                        ap += precision[mask][0]
                ap /= len(rec_thresholds)
                cat_aps.append(ap)

                # AR = max recall at this threshold
                ar = recall[-1] if len(recall) > 0 else 0.0
                cat_ars.append(ar)

            per_cat_aps[cat] = cat_aps
            per_cat_ars[cat] = cat_ars

        # ---- Average across categories (mAP) ----
        if not per_cat_aps:
            return {"AP": 0.0, "AP50": 0.0, "AP75": 0.0, "AR": 0.0}

        # Stack: [n_cats, n_thresholds]
        all_cat_aps = np.array(list(per_cat_aps.values()))
        all_cat_ars = np.array(list(per_cat_ars.values()))

        # mAP = mean over categories, then mean over thresholds
        results = {}
        results["AP"] = float(all_cat_aps.mean() * 100)
        for t in [0.5, 0.75]:
            idx = np.argmin(np.abs(iou_thresholds - t))
            results[f"AP{int(t*100)}"] = float(all_cat_aps[:, idx].mean() * 100)
        results["AR"] = float(all_cat_ars.mean() * 100)

        # Per-category breakdown
        for cat, aps in per_cat_aps.items():
            results[f"AP-{cat}"] = float(np.mean(aps) * 100)

        if self.out_dir is not None:
            data = {}
            data["predictions"] = matched_preds
            data["mesh_embeddings"] = mesh_embeddings
            with open(os.path.join(self.out_dir, "matched_predictions.pkl"), 'wb') as f:
                pickle.dump(data, f)

        return results

# ============================================================================
# Metric 1b: GPSm — GPS × mask IoU (geometric mean)
# ============================================================================

class GPSmEvaluator(PerPointGPSEvaluator):
    """
    GPSm = sqrt(GPS × mask_IoU).

    Extends PerPointGPSEvaluator to also compute mask IoU between predicted
    and GT segmentation masks, then uses the geometric mean as the matching
    score for COCO AP.

    To use this, predictions must also contain a "segmentation_mask" key
    (binary mask on the full image), and ground_truths must contain "segmentation_mask".
    """

    def compute_gpsm_for_instance(
        self, ogps: float, pred_mask: np.ndarray, gt_mask: np.ndarray
    ) -> float:
        """
        Args:
            ogps: OGPS score from PerPointGPSEvaluator
            pred_mask: [H, W] binary prediction mask
            gt_mask: [H, W] binary GT mask
        Returns:
            GPSm = sqrt(OGPS × IoU)
        """
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        union = np.logical_or(pred_mask, gt_mask).sum()
        iou = intersection / max(union, 1)
        return float(np.sqrt(ogps * iou))


# ============================================================================
# Metric 2: Cross-Mesh Alignment GPS
# ============================================================================

class MeshAlignmentEvaluator:
    """
    Evaluates whether the shared embedding space correctly aligns
    semantically corresponding vertices across different meshes.

    Directly mirrors DensePose's MeshAlignmentEvaluator.

    For each pair of meshes (A, B):
      1. Take keyvertices on mesh A (anatomical landmarks like "nose", "left_ear")
      2. Get their embeddings from the embedder
      3. Find the most similar vertex on mesh B by dot-product
      4. Measure geodesic distance on mesh B between the matched vertex
         and the true corresponding keyvertex on mesh B
      5. Convert to GPS: exp(-d² / 2σ²) with σ=0.255

    Requires a keyvertex mapping JSON (e.g. mesh_keyvertices_v0.json from DensePose).
    """

    DEFAULT_SIGMA = 0.255

    def __init__(
        self,
        embedder: nn.Module,
        mesh_keyvertices: Dict[str, Dict[str, int]],
        mesh_names: Optional[List[str]] = None,
    ):
        """
        Args:
            embedder: CSEEmbedder that maps mesh_name -> [K, D]
            mesh_keyvertices: dict mapping mesh_name -> {landmark_name: vertex_index}.
                Download from: https://dl.fbaipublicfiles.com/densepose/data/cse/mesh_keyvertices_v0.json
            mesh_names: which meshes to evaluate. If None, uses all from embedder.
        """
        self.embedder = embedder
        self.mesh_keyvertices = mesh_keyvertices
        self.mesh_names = mesh_names if mesh_names else list(embedder.mesh_names)

    @torch.no_grad()
    def evaluate(self) -> Dict[str, Dict]:
        """
        Returns:
            {
                "GE_mean": float,           # mean geodesic error (lower is better)
                "GPS_mean": float,           # mean GPS (higher is better)
                "per_mesh": {
                    mesh_name: {"GE": float, "GPS": float},
                    ...
                }
            }
        """
        ge_per_mesh = {}
        gps_per_mesh = {}

        for mesh_name_1 in self.mesh_names:
            avg_errors = []
            avg_gps = []

            embeddings_1 = self.embedder(mesh_name_1)  # [K1, D]
            keyvertices_1 = self.mesh_keyvertices.get(mesh_name_1, {})
            if not keyvertices_1:
                continue
            keyvertex_names_1 = list(keyvertices_1.keys())
            keyvertex_indices_1 = [keyvertices_1[name] for name in keyvertex_names_1]

            for mesh_name_2 in self.mesh_names:
                if mesh_name_1 == mesh_name_2:
                    continue

                embeddings_2 = self.embedder(mesh_name_2)  # [K2, D]
                keyvertices_2 = self.mesh_keyvertices.get(mesh_name_2, {})
                if not keyvertices_2:
                    continue

                # Similarity: embeddings of keyvertices on mesh1 vs ALL vertices on mesh2
                # [n_keyvertices, D] @ [D, K2] -> [n_keyvertices, K2]
                sim_matrix_12 = embeddings_1[keyvertex_indices_1].mm(embeddings_2.T)

                # For each keyvertex on mesh1, find the best-matching vertex on mesh2
                matched_vertices_on_2 = sim_matrix_12.argmax(dim=1)  # [n_keyvertices]

                # True corresponding keyvertices on mesh2
                true_vertices_on_2 = torch.tensor(
                    [keyvertices_2[name] for name in keyvertex_names_1],
                    dtype=torch.long,
                )

                # Geodesic distance on mesh2 between matched and true vertices
                mesh_2 = create_mesh(mesh_name_2, device=embeddings_2.device)
                geodists = mesh_2.geodists[
                    matched_vertices_on_2.cpu(), true_vertices_on_2
                ]  # [n_keyvertices]

                # GPS = exp(-d² / 2σ²)
                sigma = self.DEFAULT_SIGMA
                gps = torch.exp(-(geodists ** 2) / (2 * sigma ** 2))

                avg_errors.append(geodists.mean().item())
                avg_gps.append(gps.mean().item())

            if avg_errors:
                ge_per_mesh[mesh_name_1] = float(torch.tensor(avg_errors).mean())
                gps_per_mesh[mesh_name_1] = float(torch.tensor(avg_gps).mean())

        ge_mean = float(torch.tensor(list(ge_per_mesh.values())).mean()) if ge_per_mesh else 0.0
        gps_mean = float(torch.tensor(list(gps_per_mesh.values())).mean()) if gps_per_mesh else 0.0

        return {
            "GE_mean": ge_mean,
            "GPS_mean": gps_mean,
            "per_mesh": {
                name: {"GE": ge_per_mesh.get(name, 0), "GPS": gps_per_mesh.get(name, 0)}
                for name in self.mesh_names
            },
        }