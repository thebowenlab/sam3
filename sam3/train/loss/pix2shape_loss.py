import torch
import torch.nn.functional as F
from torch import nn
from typing import Any, Tuple
from .loss_fns import LossWithWeights
from sam3.model.box_ops import box_cxcywh_to_xywh
from sam3.train.data.mesh import MeshCatalog, create_mesh
from sam3.train.loss.cse_soft_embed_loss import squared_euclidean_distance_matrix, normalize_embeddings


def _create_pixel_dist_matrix(grid_size: int) -> torch.Tensor:
    rows = torch.arange(grid_size)
    cols = torch.arange(grid_size)
    # at index `i` contains [row, col], where
    # row = i // grid_size
    # col = i % grid_size
    pix_coords = (
        torch.stack(torch.meshgrid(rows, cols), -1).reshape((grid_size * grid_size, 2)).float()
    )
    return squared_euclidean_distance_matrix(pix_coords, pix_coords)

def _sample_fg_pixels_randperm(fg_mask: torch.Tensor, sample_size: int) -> torch.Tensor:
    fg_mask_flattened = fg_mask.reshape((-1,))
    num_pixels = int(fg_mask_flattened.sum().item())
    fg_pixel_indices = fg_mask_flattened.nonzero(as_tuple=True)[0]
    if (sample_size <= 0) or (num_pixels <= sample_size):
        return fg_pixel_indices
    sample_indices = torch.randperm(num_pixels, device=fg_mask.device)[:sample_size]
    return fg_pixel_indices[sample_indices]

def _crop_and_rescale_mask(mask_gt, box_gt, emb_h, emb_w):
    # takes full image scale mask, crops to gt box, scales to emb dims
    box_xywh = box_cxcywh_to_xywh(box_gt)
    mask_h, mask_w = mask_gt.shape
    crop_y = box_xywh[1]*mask_h
    crop_y2 = (box_xywh[1]+box_xywh[3])*mask_h
    crop_x = box_xywh[0]*mask_w
    crop_x2 = (box_xywh[0]+box_xywh[2])*mask_w
    cropped_mask = mask_gt[crop_y.int():crop_y2.int(), crop_x.int():crop_x2.int()]
    return F.interpolate(
        cropped_mask.double().unsqueeze(0).unsqueeze(0), (emb_h, emb_w),
        mode="nearest",
    ).squeeze(0).squeeze(0)


class Pix2ShapeLoss(LossWithWeights):
    """
    Pix2Shape adapted from DensePose/detectron2 for SAM3.

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
    """

    def __init__(
        self,
        weight_dict=None,
        compute_aux=False,
        embed_dim=16,
        norm_p=2,
        num_pixels=100,
        pix_sigma=5.0,
        temp_pix_to_vertex=.05,
        temp_vertex_to_pix=.05,
    ):
        super().__init__(weight_dict, compute_aux)
        self.embed_dim = embed_dim
        self.norm_p = norm_p
        self.num_pixels = num_pixels
        self.pix_sigma = pix_sigma
        self.temp_pix_to_vertex = temp_pix_to_vertex
        self.temp_vertex_to_pix = temp_vertex_to_pix
        self.pixel_dists = None
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
        Compute the Pix2Shape loss.
        """
        pred_embeddings = outputs["pred_embeddings"]

        if self.pixel_dists is None:
            # Should be N x D x S x S, we take S
            self.pixel_dists = _create_pixel_dist_matrix(pred_embeddings.shape[2]).to(device = pred_embeddings.device)
        gt_masks = targets["masks"] if indices[2] is None else targets["masks"][indices[2]]
        gt_boxes = targets["boxes"] if indices[2] is None else targets["boxes"][indices[2]] # [M, 4] where M is number of matches


        h, w = pred_embeddings.shape[2:]
        numQ = outputs["pred_boxes"].shape[1]

        ref_model = targets["ref_model"] if indices[2] is None else [targets["ref_model"][indices[2][i]] for i in range(len(indices[2]))]

        num_matches = indices[1].shape[0]
        num_contributing_points = 0

        all_match_losses = []
        for match_num in range(num_matches):
            mesh_name = ref_model[match_num]
            if mesh_name is None or mesh_name not in outputs["mesh_embeddings"].keys():
                continue
            box_gt = gt_boxes[match_num]
            mask_gt = _crop_and_rescale_mask(gt_masks[match_num], box_gt, h, w)

            embed_index = indices[0][match_num]*numQ+indices[1][match_num]
            pixel_embeddings = pred_embeddings[embed_index]
            # -> tensor [K, D]
            mesh_vertex_embeddings = outputs["mesh_embeddings"][mesh_name]

            # pixel indices [M]
            pixel_indices_flattened = _sample_fg_pixels_randperm(
                mask_gt, self.num_pixels
            )
            # pixel distances [M, M]
            pixel_dists = self.pixel_dists.to(pixel_embeddings.device)[
                torch.meshgrid(pixel_indices_flattened, pixel_indices_flattened)
            ]
            # pixel embeddings [M, D]
            pixel_embeddings_sampled = normalize_embeddings(
                pixel_embeddings.reshape((self.embed_dim, -1))[:, pixel_indices_flattened].T
            )
            # pixel-vertex similarity [M, K]
            sim_matrix = pixel_embeddings_sampled.mm(mesh_vertex_embeddings.T)
            c_pix_vertex = F.softmax(sim_matrix / self.temp_pix_to_vertex, dim=1)
            c_vertex_pix = F.softmax(sim_matrix.T / self.temp_vertex_to_pix, dim=1)
            c_cycle = c_pix_vertex.mm(c_vertex_pix)
            loss_cycle = torch.norm(pixel_dists * c_cycle, p=self.norm_p)
            all_match_losses.append(loss_cycle)


        total_loss = {}
        if all_match_losses:
            total_loss["loss_pix2shape_cycle"] = torch.stack(all_match_losses).mean()
        else:
            total_loss["loss_pix2shape_cycle"] = pred_embeddings.sum() * 0
        return total_loss
