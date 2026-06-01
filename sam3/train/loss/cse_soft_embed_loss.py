# Adapted from detectron2 cse loss

import torch
import torch.nn.functional as F
from torch import nn
from typing import Any, Tuple
from .loss_fns import LossWithWeights
from sam3.model.box_ops import box_cxcywh_to_xywh
from sam3.train.data.mesh import MeshCatalog, create_mesh



def _linear_interpolation_utilities(v_norm, v0_src, size_src, v0_dst, size_dst, size_z):
    """
    Computes utility values for linear interpolation at points v.
    The points are given as normalized offsets in the source interval
    (v0_src, v0_src + size_src), more precisely:
        v = v0_src + v_norm * size_src / 256.0
    The computed utilities include lower points v_lo, upper points v_hi,
    interpolation weights v_w and flags j_valid indicating whether the
    points falls into the destination interval (v0_dst, v0_dst + size_dst).

    Args:
        v_norm (:obj: `torch.Tensor`): tensor of size N containing
            normalized point offsets
        v0_src (:obj: `torch.Tensor`): tensor of size N containing
            left bounds of source intervals for normalized points
        size_src (:obj: `torch.Tensor`): tensor of size N containing
            source interval sizes for normalized points
        v0_dst (:obj: `torch.Tensor`): tensor of size N containing
            left bounds of destination intervals
        size_dst (:obj: `torch.Tensor`): tensor of size N containing
            destination interval sizes
        size_z (int): interval size for data to be interpolated

    Returns:
        v_lo (:obj: `torch.Tensor`): int tensor of size N containing
            indices of lower values used for interpolation, all values are
            integers from [0, size_z - 1]
        v_hi (:obj: `torch.Tensor`): int tensor of size N containing
            indices of upper values used for interpolation, all values are
            integers from [0, size_z - 1]
        v_w (:obj: `torch.Tensor`): float tensor of size N containing
            interpolation weights
        j_valid (:obj: `torch.Tensor`): uint8 tensor of size N containing
            0 for points outside the estimation interval
            (v0_est, v0_est + size_est) and 1 otherwise
    """
    v = v0_src + v_norm * size_src / 256.0
    j_valid = (v - v0_dst >= 0) * (v - v0_dst < size_dst)
    v_grid = (v - v0_dst) * size_z / size_dst
    v_lo = v_grid.floor().long().clamp(min=0, max=size_z - 1)
    v_hi = (v_lo + 1).clamp(max=size_z - 1)
    v_grid = torch.min(v_hi.float(), v_grid)
    v_w = v_grid - v_lo.float()
    # print("final_tensors: ", v_lo, v_hi, v_w, j_valid)
    return v_lo, v_hi, v_w, j_valid


class BilinearInterpolationHelper:
    """
    Args:
        packed_annotations: object that contains packed annotations
        j_valid (:obj: `torch.Tensor`): uint8 tensor of size M containing
            0 for points to be discarded and 1 for points to be selected
        y_lo (:obj: `torch.Tensor`): int tensor of indices of upper values
            in z_est for each point
        y_hi (:obj: `torch.Tensor`): int tensor of indices of lower values
            in z_est for each point
        x_lo (:obj: `torch.Tensor`): int tensor of indices of left values
            in z_est for each point
        x_hi (:obj: `torch.Tensor`): int tensor of indices of right values
            in z_est for each point
        w_ylo_xlo (:obj: `torch.Tensor`): float tensor of size M;
            contains upper-left value weight for each point
        w_ylo_xhi (:obj: `torch.Tensor`): float tensor of size M;
            contains upper-right value weight for each point
        w_yhi_xlo (:obj: `torch.Tensor`): float tensor of size M;
            contains lower-left value weight for each point
        w_yhi_xhi (:obj: `torch.Tensor`): float tensor of size M;
            contains lower-right value weight for each point
    """

    def __init__(
        self,
        j_valid: torch.Tensor,
        y_lo: torch.Tensor,
        y_hi: torch.Tensor,
        x_lo: torch.Tensor,
        x_hi: torch.Tensor,
        w_ylo_xlo: torch.Tensor,
        w_ylo_xhi: torch.Tensor,
        w_yhi_xlo: torch.Tensor,
        w_yhi_xhi: torch.Tensor,
    ):
        for k, v in locals().items():
            if k != "self":
                setattr(self, k, v)

    @staticmethod
    def from_matches(
        pred_box, gt_box, x_gt, y_gt, densepose_outputs_size_hw: Tuple[int, int]
    ) -> "BilinearInterpolationHelper":
        """
        Args:
            packed_annotations: annotations packed into tensors, the following
                attributes are required:
                 - bbox_xywh_gt
                 - bbox_xywh_est
                 - x_gt
                 - y_gt
                 - point_bbox_with_dp_indices
                 - point_bbox_indices
            densepose_outputs_size_hw (tuple [int, int]): resolution of
                DensePose predictor outputs (H, W)
        Return:
            An instance of `BilinearInterpolationHelper` used to perform
            interpolation for the given annotation points and output resolution
        """        
        zh, zw = densepose_outputs_size_hw
        # print("pred_box shape:", pred_box.shape)
        # print("gt_box shape:", gt_box.shape)

        x0_gt, y0_gt, w_gt, h_gt = box_cxcywh_to_xywh(gt_box).unbind(dim=1)
        x0_est, y0_est, w_est, h_est = box_cxcywh_to_xywh(pred_box).unbind(dim=1)
        x_lo, x_hi, x_w, jx_valid = _linear_interpolation_utilities(
            x_gt, x0_gt, w_gt, x0_est, w_est, zw
        )
        y_lo, y_hi, y_w, jy_valid = _linear_interpolation_utilities(
            y_gt, y0_gt, h_gt, y0_est, h_est, zh
        )
        j_valid = jx_valid * jy_valid

        w_ylo_xlo = (1.0 - x_w) * (1.0 - y_w)
        w_ylo_xhi = x_w * (1.0 - y_w)
        w_yhi_xlo = (1.0 - x_w) * y_w
        w_yhi_xhi = x_w * y_w

        return BilinearInterpolationHelper(
            j_valid,
            y_lo,
            y_hi,
            x_lo,
            x_hi,
            w_ylo_xlo,
            w_ylo_xhi,
            w_yhi_xlo,
            w_yhi_xhi,
        )

    def extract_at_points(
        self,
        z_est,
        idx_tensor,
        slice_fine_segm=None,
        w_ylo_xlo=None,
        w_ylo_xhi=None,
        w_yhi_xlo=None,
        w_yhi_xhi=None,
    ):
        """
        Extract ground truth values z_gt for valid point indices and estimated
        values z_est using bilinear interpolation over top-left (y_lo, x_lo),
        top-right (y_lo, x_hi), bottom-left (y_hi, x_lo) and bottom-right
        (y_hi, x_hi) values in z_est with corresponding weights:
        w_ylo_xlo, w_ylo_xhi, w_yhi_xlo and w_yhi_xhi.
        Use slice_fine_segm to slice dim=1 in z_est
        """
        # slice_fine_segm = (
        #     self.packed_annotations.fine_segm_labels_gt
        #     if slice_fine_segm is None
        #     else slice_fine_segm
        # )
        # idx_tensor = torch.zeros(w_ylo_xlo.shape[0], dtype=torch.int, device=z_est.device)
        # print(z_est.shape)

        w_ylo_xlo = self.w_ylo_xlo if w_ylo_xlo is None else w_ylo_xlo
        w_ylo_xhi = self.w_ylo_xhi if w_ylo_xhi is None else w_ylo_xhi
        w_yhi_xlo = self.w_yhi_xlo if w_yhi_xlo is None else w_yhi_xlo
        w_yhi_xhi = self.w_yhi_xhi if w_yhi_xhi is None else w_yhi_xhi

        # index_bbox = self.packed_annotations.point_bbox_indices
        z_est_sampled = (
            z_est[idx_tensor, slice_fine_segm, self.y_lo, self.x_lo] * w_ylo_xlo
            + z_est[idx_tensor, slice_fine_segm, self.y_lo, self.x_hi] * w_ylo_xhi
            + z_est[idx_tensor, slice_fine_segm, self.y_hi, self.x_lo] * w_yhi_xlo
            + z_est[idx_tensor, slice_fine_segm, self.y_hi, self.x_hi] * w_yhi_xhi
        )
        return z_est_sampled


def resample_data(
    z, bbox_xywh_src, bbox_xywh_dst, wout, hout, mode: str = "nearest", padding_mode: str = "zeros"
):
    """
    Args:
        z (:obj: `torch.Tensor`): tensor of size (N,C,H,W) with data to be
            resampled
        bbox_xywh_src (:obj: `torch.Tensor`): tensor of size (N,4) containing
            source bounding boxes in format XYWH
        bbox_xywh_dst (:obj: `torch.Tensor`): tensor of size (N,4) containing
            destination bounding boxes in format XYWH
    Return:
        zresampled (:obj: `torch.Tensor`): tensor of size (N, C, Hout, Wout)
            with resampled values of z, where D is the discretization size
    """
    n = bbox_xywh_src.size(0)
    assert n == bbox_xywh_dst.size(0), (
        "The number of "
        "source ROIs for resampling ({}) should be equal to the number "
        "of destination ROIs ({})".format(bbox_xywh_src.size(0), bbox_xywh_dst.size(0))
    )
    x0src, y0src, wsrc, hsrc = bbox_xywh_src.unbind(dim=1)
    x0dst, y0dst, wdst, hdst = bbox_xywh_dst.unbind(dim=1)
    x0dst_norm = 2 * (x0dst - x0src) / wsrc - 1
    y0dst_norm = 2 * (y0dst - y0src) / hsrc - 1
    x1dst_norm = 2 * (x0dst + wdst - x0src) / wsrc - 1
    y1dst_norm = 2 * (y0dst + hdst - y0src) / hsrc - 1
    grid_w = torch.arange(wout, device=z.device, dtype=torch.float) / wout
    grid_h = torch.arange(hout, device=z.device, dtype=torch.float) / hout
    grid_w_expanded = grid_w[None, None, :].expand(n, hout, wout)
    grid_h_expanded = grid_h[None, :, None].expand(n, hout, wout)
    # pyre-fixme[16]: `float` has no attribute `__getitem__`.
    dx_expanded = (x1dst_norm - x0dst_norm)[:, None, None].expand(n, hout, wout)
    dy_expanded = (y1dst_norm - y0dst_norm)[:, None, None].expand(n, hout, wout)
    x0_expanded = x0dst_norm[:, None, None].expand(n, hout, wout)
    y0_expanded = y0dst_norm[:, None, None].expand(n, hout, wout)
    grid_x = grid_w_expanded * dx_expanded + x0_expanded
    grid_y = grid_h_expanded * dy_expanded + y0_expanded
    grid = torch.stack((grid_x, grid_y), dim=3)
    # resample Z from (N, C, H, W) into (N, C, Hout, Wout)
    zresampled = F.grid_sample(z, grid, mode=mode, padding_mode=padding_mode, align_corners=True)
    return zresampled

def squared_euclidean_distance_matrix(
    pts1: torch.Tensor, pts2: torch.Tensor
) -> torch.Tensor:
    """
    Pairwise squared Euclidean distances.
    Args:
        pts1: [M, D]
        pts2: [N, D]
    Returns:
        [M, N] distance matrix
    """
    edm = torch.mm(-2 * pts1, pts2.t())
    edm += (pts1 * pts1).sum(1, keepdim=True) + (pts2 * pts2).sum(
        1, keepdim=True
    ).t()
    edm = edm.clamp(min=0.0)
    return edm.contiguous()


def normalize_embeddings(
    embeddings: torch.Tensor, epsilon: float = 1e-6
) -> torch.Tensor:
    """Normalize embeddings to unit L2 norm."""
    return embeddings / torch.clamp(
        embeddings.norm(p=None, dim=1, keepdim=True), min=epsilon
    )

def dummy_loss(pred_embeddings, mesh_vertex_embeddings):
    return pred_embeddings.sum()*0 + mesh_vertex_embeddings.sum()*0 

class CSESoftEmbeddingLoss(LossWithWeights):
    """
    CSE Soft Embedding Loss adapted from DensePose/detectron2 for SAM3.

    Assumes:
      - The model outputs contain a key "pred_embeddings" of shape [N, D, S, S]
        (the CSE embedding head output, one per detected instance).
      - Targets contain:
        - "vertex_ids_gt": [P] ground-truth mesh vertex IDs per annotated point
        - "vertex_mesh_ids_gt": [P] mesh ID per annotated point
        - "point_coords_gt": [P, 2] normalized (x, y) coords of annotated points
          within each instance's bounding box (values in [0, 1])
        - "point_instance_indices": [P] which instance (0..N-1) each point belongs to
      - An `embedder` nn.Module is accessible (e.g. stored on the model) that maps
        mesh_name -> [K, D] vertex embeddings.
      - A `mesh_catalog` mapping mesh_id -> mesh_name, and mesh_name -> geodesic
        distance matrix [K, K].

    Config params:
      - embdist_gauss_sigma: temperature for embedding distance softmax (default 0.01)
      - geodist_gauss_sigma: temperature for geodesic distance soft targets (default 0.01)
    """

    def __init__(
        self,
        weight_dict=None,
        compute_aux=False,
        embdist_gauss_sigma: float = 0.01,
        geodist_gauss_sigma: float = 0.01,
    ):
        super().__init__(weight_dict, compute_aux)
        self.embdist_gauss_sigma = embdist_gauss_sigma
        self.geodist_gauss_sigma = geodist_gauss_sigma
        self.target_keys.extend(
            [
                "vertex_ids_gt",
                "vertex_mesh_ids_gt",
                "point_coords_gt",
                "point_instance_indices",
            ]
        )


    def get_loss(self, outputs, targets, indices, num_boxes):
        """
        Compute the CSE soft embedding loss.

        This mirrors detectron2's SoftEmbeddingLoss but fits into SAM3's
        LossWithWeights interface (receives outputs, targets, indices, num_boxes).
        """
        # embeddings are already filtered to only matched predictions.
        pred_embeddings = outputs["pred_embeddings"]  # [N, D, S, S]
        # print(pred_embeddings.norm(dim=1).mean())
        # print(indices)
        h, w = pred_embeddings.shape[2:]

        pred_boxes = outputs["pred_boxes"][indices[0], indices[1]]
        target_boxes = targets["boxes"] if indices[2] is None else targets["boxes"][indices[2]] # [M, 4] where M is number of matches

        dp_x = targets["dp_x"] if indices[2] is None else [targets["dp_x"][indices[2][i]] for i in range(len(indices[2]))]
        dp_y = targets["dp_y"] if indices[2] is None else [targets["dp_y"][indices[2][i]] for i in range(len(indices[2]))]

        ref_model = targets["ref_model"] if indices[2] is None else [targets["ref_model"][indices[2][i]] for i in range(len(indices[2]))]
        dp_vertex = targets["dp_vertex"] if indices[2] is None else [targets["dp_vertex"][indices[2][i]] for i in range(len(indices[2]))]

        total_loss = pred_embeddings.sum()*0
        matches = indices[1].shape[0]

        for m in range(matches):
            if dp_x[m] is None:
                dp_x[m] = torch.tensor([], dtype=torch.float, device=pred_embeddings.device) 
                dp_y[m] = torch.tensor([], dtype=torch.float, device=pred_embeddings.device)
                dp_vertex[m] = torch.tensor([], dtype=torch.int, device=pred_embeddings.device)
                ref_model[m] = ""

        # Collate all point data
        all_pred_boxes = torch.cat([pred_boxes[m].repeat(len(dp_x[m]), 1) for m in range(matches)], dim=0)  # [P, 4]
        all_gt_boxes = torch.cat([target_boxes[m].repeat(len(dp_x[m]), 1) for m in range(matches)], dim=0)  # [P, 4]
        all_dp_x = torch.cat([dp_x[m] for m in range(matches)], dim=0)  # [P]
        all_dp_y = torch.cat([dp_y[m] for m in range(matches)], dim=0)  # [P]
        all_dp_vertex = torch.cat([dp_vertex[m] for m in range(matches)], dim=0)  # [P]
        all_mesh_names = sum([[ref_model[m]] * len(dp_x[m]) for m in range(matches)], [])

        # Embeddings should already be filtered to matched predictions
        embed_index = torch.tensor(range(matches), dtype=torch.long, device=pred_embeddings.device)
        all_embed_idx = torch.cat([embed_index[m].repeat(len(dp_x[m]), 1) for m in range(matches)], dim=0).squeeze()  # [P]
        

        interpolator = BilinearInterpolationHelper.from_matches(
            all_pred_boxes,
            all_gt_boxes,
            all_dp_x,
            all_dp_y,
            (h, w),
        )


        losses = {}
        for mesh_id in outputs["mesh_embeddings"].keys():
            # valid points are those that fall into estimated bbox
            # and correspond to the current mesh
            losses[mesh_id] = dummy_loss(pred_embeddings, outputs["mesh_embeddings"][mesh_id])
            j_valid = interpolator.j_valid * torch.tensor([mesh_name == mesh_id for mesh_name in all_mesh_names], dtype=torch.bool, device=pred_embeddings.device)
            if not torch.any(j_valid):
                continue
            # extract estimated embeddings for valid points
            # -> tensor [J, D]
            vertex_embeddings_i = normalize_embeddings(
                interpolator.extract_at_points(
                    pred_embeddings,
                    all_embed_idx,
                    slice_fine_segm=slice(None),
                    w_ylo_xlo=interpolator.w_ylo_xlo[:, None],  # pyre-ignore[16]
                    w_ylo_xhi=interpolator.w_ylo_xhi[:, None],  # pyre-ignore[16]
                    w_yhi_xlo=interpolator.w_yhi_xlo[:, None],  # pyre-ignore[16]
                    w_yhi_xhi=interpolator.w_yhi_xhi[:, None],  # pyre-ignore[16]
                )[j_valid, :]
            )
            # extract vertex ids for valid points
            # -> tensor [J]
            vertex_indices_i = all_dp_vertex[j_valid]
            # embeddings for all mesh vertices
            # -> tensor [K, D]
            mesh_vertex_embeddings = outputs["mesh_embeddings"][mesh_id]
            # softmax values of geodesic distances for GT mesh vertices
            # -> tensor [J, K]
            mesh = create_mesh(mesh_id, mesh_vertex_embeddings.device)
            # print(vertex_indices_i.dtype)
            geodist_softmax_values = F.softmax(
                mesh.geodists[vertex_indices_i] / (-self.geodist_gauss_sigma), dim=1
            )
            # logsoftmax values for valid points
            # -> tensor [J, K]
            embdist_logsoftmax_values = F.log_softmax(
                squared_euclidean_distance_matrix(vertex_embeddings_i, mesh_vertex_embeddings)
                / (-self.embdist_gauss_sigma),
                dim=1,
            )
            losses[mesh_id] = (-geodist_softmax_values * embdist_logsoftmax_values).sum(1).mean()

        # pyre-fixme[29]: `Union[(self: Tensor) -> Any, Module, Tensor]` is not a
        #  function.
        total_loss += sum(losses.values())
        return {"loss_cse_embed": total_loss}

    # def get_loss(self, outputs, targets, indices, num_boxes):
    #     """
    #     Compute the CSE soft embedding loss.

    #     This mirrors detectron2's SoftEmbeddingLoss but fits into SAM3's
    #     LossWithWeights interface (receives outputs, targets, indices, num_boxes).
    #     """

    #     pred_embeddings = outputs["pred_embeddings"]  # [N, D, S, S]
    #     # print(pred_embeddings.norm(dim=1).mean())
    #     # print(indices)
    #     h, w = pred_embeddings.shape[2:]
    #     numQ = outputs["pred_boxes"].shape[1]

    #     pred_boxes = outputs["pred_boxes"][indices[0], indices[1]]
    #     target_boxes = targets["boxes"] if indices[2] is None else targets["boxes"][indices[2]] # [M, 4] where M is number of matches

    #     dp_x = targets["dp_x"] if indices[2] is None else [targets["dp_x"][indices[2][i]] for i in range(len(indices[2]))]
    #     dp_y = targets["dp_y"] if indices[2] is None else [targets["dp_y"][indices[2][i]] for i in range(len(indices[2]))]

    #     ref_model = targets["ref_model"] if indices[2] is None else [targets["ref_model"][indices[2][i]] for i in range(len(indices[2]))]
    #     dp_vertex = targets["dp_vertex"] if indices[2] is None else [targets["dp_vertex"][indices[2][i]] for i in range(len(indices[2]))]

    #     total_loss = pred_embeddings.sum()*0
    #     num_matches = indices[1].shape[0]
    #     num_contributing_points = 0

    #     for match_num in range(num_matches):
    #         mesh_name = ref_model[match_num]
    #         if mesh_name is None or mesh_name not in outputs["mesh_embeddings"].keys():
    #             continue
            
    #         # -> tensor [K, D]
    #         mesh_vertex_embeddings = outputs["mesh_embeddings"][mesh_name]

    #         mesh = create_mesh(mesh_name, mesh_vertex_embeddings.device)
    #         dp_vertices = torch.tensor(dp_vertex[match_num], device = pred_embeddings.device)
    #         # print(match_num, pred_boxes[match_num], target_boxes[match_num], dp_x[match_num], dp_y[match_num])

    #         interpolator = BilinearInterpolationHelper.from_matches(
    #             pred_boxes[match_num].unsqueeze(0),
    #             target_boxes[match_num].unsqueeze(0),
    #             dp_x[match_num],
    #             dp_y[match_num],
    #             (h, w),
    #         )
    #         j_valid = interpolator.j_valid


    #         if torch.sum(j_valid) == 0: # no valid points shouldn't contribute to loss
    #             total_loss += dummy_loss(pred_embeddings, mesh_vertex_embeddings)
    #             continue
    #         embed_index = indices[0][match_num]*numQ+indices[1][match_num]
    #         vertex_embeddings_i = normalize_embeddings(
    #             interpolator.extract_at_points(
    #                 pred_embeddings[embed_index].unsqueeze(0),
    #                 slice_fine_segm=slice(None),
    #                 w_ylo_xlo=interpolator.w_ylo_xlo[:, None],  # pyre-ignore[16]
    #                 w_ylo_xhi=interpolator.w_ylo_xhi[:, None],  # pyre-ignore[16]
    #                 w_yhi_xlo=interpolator.w_yhi_xlo[:, None],  # pyre-ignore[16]
    #                 w_yhi_xhi=interpolator.w_yhi_xhi[:, None],  # pyre-ignore[16]
    #             )[j_valid, :]
    #         )
    #         # print(vertex_embeddings_i.shape)

    #         geodist_softmax_values = F.softmax(
    #             mesh.geodists[dp_vertices[j_valid]] / (-self.geodist_gauss_sigma), dim=1
    #         ) # 


    #         # logsoftmax values for valid points
    #         # -> tensor [J, K]
    #         embdist_logsoftmax_values = F.log_softmax(
    #             squared_euclidean_distance_matrix(vertex_embeddings_i, mesh_vertex_embeddings)
    #             / (-self.embdist_gauss_sigma),
    #             dim=1,
    #         )


    #         # dists = squared_euclidean_distance_matrix(vertex_embeddings_i, mesh_vertex_embeddings)
    #         # print("dist range:", dists.min().item(), dists.max().item(), dists.mean().item())
    #         # dists = squared_euclidean_distance_matrix(vertex_embeddings_i, vertex_embeddings_i)
    #         # print("distself range:", dists.min().item(), dists.max().item(), dists.mean().item())
    #         # dists = squared_euclidean_distance_matrix(mesh_vertex_embeddings, mesh_vertex_embeddings)
    #         # print("meshdist range:", dists.min().item(), dists.max().item(), dists.mean().item())

    #         num_contributing_points += j_valid.sum()
    #         # total_loss += (-geodist_softmax_values * embdist_logsoftmax_values).sum(1).mean()
    #         total_loss += (-geodist_softmax_values * embdist_logsoftmax_values).sum()
    #     # print(total_loss)
    #     if num_contributing_points > 0:
    #         total_loss /= num_contributing_points
    #     return {"loss_cse_embed": total_loss}