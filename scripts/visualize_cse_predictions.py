"""
Visualize CSE predictions from a pickle file using PCA-based coloring.

Usage:
    python -m visualize_cse_predictions \
        --predictions predictions.pkl \
        --images_dir /path/to/images \
        --output_dir /path/to/output \
        --lbo_dir /path/to/lbo/folder 
"""

import argparse
import os
import pickle
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# PCA vertex coloring
# ============================================================================

@torch.no_grad()
def get_vertex_colors_from_embeddings(
    embeddings: torch.Tensor,
    mesh_name: str,
    device: torch.device,
) -> torch.Tensor:
    """
    Map vertex embeddings [K, D] -> RGB colors [K, 3] via PCA.
    Returns: [K, 3] tensor of floats in [0, 1].
    """

    # If using the smpl human mesh, load the mapping directly
    if mesh_name == "smpl_27554":
        embed_map, _ = np.load("/home/camposadmin/Documents/lvis/mds_d=256.npy", allow_pickle=True)
        embed_map = torch.tensor(embed_map).float()[:, 0]
        embed_map -= embed_map.min()
        embed_map /= embed_map.max()
        embed_map *= 255
        color_map = cv2.applyColorMap((embed_map.to(dtype=torch.uint8)).numpy(), cv2.COLORMAP_JET)
        return torch.tensor(color_map, dtype=torch.float32, device=device).squeeze(1)/255

    mean = embeddings.mean(dim=0, keepdim=True)
    centered = embeddings - mean

    U, S, V = torch.pca_lowrank(centered, q=3)
    projected = U * S.unsqueeze(0)  # [K, 3]

    for c in range(3):
        col = projected[:, c]
        projected[:, c] = (col - col.min()) / (col.max() - col.min() + 1e-8)

    return projected.to(dtype=torch.float32, device=device)  # [K, 3] RGB in [0, 1]


# ============================================================================
# Squared euclidean distance
# ============================================================================

def squared_euclidean_distance_matrix(pts1: torch.Tensor, pts2: torch.Tensor) -> torch.Tensor:
    edm = torch.mm(-2 * pts1, pts2.t())
    edm += (pts1 * pts1).sum(1, keepdim=True) + (pts2 * pts2).sum(1, keepdim=True).t()
    return edm.contiguous()


# ============================================================================
# Per-instance visualization
# ============================================================================

@torch.no_grad()
def visualize_instance(
    image_bgr: np.ndarray,
    pred_embedding: torch.Tensor,
    pred_mask: torch.Tensor,
    bbox_xywh: Tuple[float, float, float, float],
    mesh_vertex_embs: torch.Tensor,
    vertex_colors: torch.Tensor,
    device: torch.device,
    alpha: float = 0.7,
) -> np.ndarray:
    """
    Overlay PCA-colored CSE embedding visualization for one instance.

    Args:
        image_bgr: [H_img, W_img, 3] uint8
        pred_embedding: [D, S, S]
        pred_mask: [H_img, W_img] boolean
        bbox_xywh: (x, y, w, h)
        mesh_vertex_embs: [K, D] precomputed
        vertex_colors: [K, 3] precomputed PCA colors in [0, 1]
        device: torch device
        alpha: overlay transparency
    """
    x, y, w, h = [int(v) for v in bbox_xywh]
    # pred_mask = (pred_mask.int()*0+1).bool()
    if w <= 0 or h <= 0:
        return image_bgr


    img_h, img_w = image_bgr.shape[:2]
    # print(x,y,w,h)

    # Resize embedding to bbox size: [D, S, S] -> [D, h, w]
    embedding_bbox = F.interpolate(
        pred_embedding.unsqueeze(0).to(device), (h, w),
        mode="bilinear", align_corners=False,
    ).squeeze(0)  # [D, h, w]

    # Crop mask to bbox region
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, img_w), min(y + h, img_h)
    ex0, ey0 = x0 - x, y0 - y
    ex1, ey1 = x1 - x, y1 - y
    mask_crop = pred_mask[y0:y1, x0:x1]

    new_h, new_w = y1-y0, x1-x0
    mask_bbox = torch.zeros(new_h, new_w, dtype=torch.bool, device=device)
    
    mask_bbox = mask_crop.to(device)

    if not mask_bbox.any():
        return image_bgr

    embedding_bbox = embedding_bbox[:, ey0:ey1, ex0:ex1]
    # Find closest mesh vertex for each foreground pixel
    fg_embs = embedding_bbox[:, mask_bbox].t()  # [J, D]

    chunk_size = 10_000
    closest_list = []
    for i in range(0, len(fg_embs), chunk_size):
        chunk = fg_embs[i:i + chunk_size]
        edm = squared_euclidean_distance_matrix(chunk, mesh_vertex_embs)
        closest_list.append(edm.argmin(dim=1))
    closest_verts = torch.cat(closest_list)  # [J]
    print(closest_verts)
    print("Unique Verts: ", closest_verts.unique().numel())


    # Map to PCA colors
    color_map = torch.zeros(new_h, new_w, 3, device=device)
    color_map[mask_bbox] = vertex_colors[closest_verts]

    rgb_uint8 = (color_map.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    colored_bgr = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)

    # Alpha blend (foreground only)
    mask_np = mask_bbox.cpu().numpy()
    mask_3ch = np.stack([mask_np] * 3, axis=-1)

    roi = image_bgr[y0:y1, x0:x1].astype(np.float32)
    blended = roi * (1.0 - alpha) + colored_bgr.astype(np.float32) * alpha
    roi[mask_3ch] = blended[mask_3ch]
    image_bgr[y0:y1, x0:x1] = roi.astype(np.uint8)

    return image_bgr


# ============================================================================
# Main script
# ============================================================================

def load_predictions(path: str) -> List[Dict]:
    with open(path, "rb") as f:
        return pickle.load(f)


def build_image_id_to_path(images_dir: str) -> Dict:
    """
    Build mapping from image_id to file path.
    Supports two layouts:
      1. Flat: images_dir/{image_id}.jpg
      2. COCO-style: images_dir/COCO_*_{image_id:012d}.jpg
    """
    mapping = {}
    for fname in os.listdir(images_dir):
        fpath = os.path.join(images_dir, fname)
        if not os.path.isfile(fpath):
            continue
        stem = os.path.splitext(fname)[0]
        # Try to parse the image id from the filename
        # COCO style: COCO_val2014_000000123456 -> 123456
        parts = stem.split("_")
        try:
            img_id = int(parts[-1])
        except ValueError:
            continue
        mapping[img_id] = fpath
    return mapping


def main():
    parser = argparse.ArgumentParser(description="Visualize CSE predictions")
    parser.add_argument("--predictions", required=True, help="Path to predictions pickle file")
    parser.add_argument("--images_dir", required=True, help="Directory containing input images")
    parser.add_argument("--lbo_dir", required=True, help="Directory containing init LBO feats")
    parser.add_argument("--output_dir", required=True, help="Directory to save visualizations")
    parser.add_argument("--alpha", type=float, default=0.7, help="Overlay transparency")
    parser.add_argument("--device", default="cuda", help="Device")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    # device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    device = "cpu"

    # Load predictions
    data = load_predictions(args.predictions)
    predictions = data["predictions"]
    mesh_embeddings = data["mesh_embeddings"]
    print(f"Loaded {len(predictions)} predictions")

    # Precompute vertex embeddings and PCA colors (once)
    vertex_colors = {}
    for mesh_name in mesh_embeddings.keys():
        with open(os.path.join(args.lbo_dir, "phi_" + mesh_name + "_256.pkl"), "rb") as hFile:
            data = pickle.load(hFile)
            vertex_colors[mesh_name] = get_vertex_colors_from_embeddings(torch.tensor(data["features"], device = args.device), mesh_name, device)  # [K, 3]


    # Group predictions by image_id
    preds_by_img = {}
    for pred in predictions:
        img_id = pred["image_id"]
        if img_id not in preds_by_img:
            preds_by_img[img_id] = []
        preds_by_img[img_id].append(pred)

    # Build image path mapping
    img_id_to_path = build_image_id_to_path(args.images_dir)

    # Visualize each image
    for img_id, preds in preds_by_img.items():
        if img_id not in img_id_to_path:
            print(f"Warning: image {img_id} not found in {args.images_dir}, skipping")
            continue

        image_bgr = cv2.imread(img_id_to_path[img_id])
        if image_bgr is None:
            print(f"Warning: could not read {img_id_to_path[img_id]}, skipping")
            continue

        # Sort by score descending so high-confidence instances render on top
        preds = sorted(preds, key=lambda p: p.get("score", 0.0))
        print("img id: ", img_id, ", number of predictions: ", len(preds))
        for pred in preds:

            pred_embedding = pred["embedding"]  # [D, S, S]
            pred_mask = pred["mask"]        # [H_img, W_img] bool
            bbox_xywh = pred["bbox"]             # (x, y, w, h)

            if isinstance(pred_embedding, np.ndarray):
                pred_embedding = torch.from_numpy(pred_embedding)
            if isinstance(pred_mask, np.ndarray):
                pred_mask = torch.from_numpy(pred_mask)

            image_bgr = visualize_instance(
                image_bgr=image_bgr,
                pred_embedding=pred_embedding,
                pred_mask=pred_mask,
                bbox_xywh=bbox_xywh,
                mesh_vertex_embs=mesh_embeddings[pred["mesh_name"]],
                vertex_colors=vertex_colors[pred["mesh_name"]],
                device=device,
                alpha=args.alpha,
            )

        out_path = os.path.join(args.output_dir, f"{img_id:012d}.jpg")
        cv2.imwrite(out_path, image_bgr)
        print(f"Saved {out_path}")

    print(f"Done. Visualizations saved to {args.output_dir}")


if __name__ == "__main__":
    main()