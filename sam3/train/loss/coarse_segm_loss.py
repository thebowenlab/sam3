"""
Coarse segmentation loss for the SAM3 DensePose head.

Analogous to the original DensePose coarse_segm loss (MaskLoss variant),
but adapted for SAM3's LossWithWeights interface. Uses GT instance masks
as supervision for a per-instance foreground/background predictor branch
on the densepose head.
"""

import torch
import torch.nn.functional as F
from torch import nn

from .loss_fns import LossWithWeights


def _crop_and_resize_masks(
    masks: torch.Tensor,
    boxes: torch.Tensor,
    out_h: int,
    out_w: int,
) -> torch.Tensor:
    """
    Crop each mask to its corresponding box and resize to (out_h, out_w).

    Args:
        masks: [K, H_img, W_img] binary masks (full image scale)
        boxes: [K, 4] boxes in pixel coords (x1, y1, x2, y2)
        out_h, out_w: output spatial resolution

    Returns:
        [K, out_h, out_w] masks cropped to each box and resized
    """
    K = masks.shape[0]
    if K == 0:
        return masks.new_zeros((0, out_h, out_w))

    img_h, img_w = masks.shape[1], masks.shape[2]
    results = []

    for i in range(K):
        raw_x1 = int(boxes[i, 0])
        raw_y1 = int(boxes[i, 1])
        raw_x2 = int(boxes[i, 2])
        raw_y2 = int(boxes[i, 3])

        box_h = raw_y2 - raw_y1
        box_w = raw_x2 - raw_x1

        if box_h <= 0 or box_w <= 0:
            results.append(masks.new_zeros((out_h, out_w)))
            continue

        # Clamp to image bounds for the actual crop
        x1 = max(raw_x1, 0)
        y1 = max(raw_y1, 0)
        x2 = min(raw_x2, img_w)
        y2 = min(raw_y2, img_h)

        crop = masks[i, y1:y2, x1:x2].float()  # [crop_h, crop_w]

        # Pad with zeros where the box exceeds image bounds
        pad_left = x1 - raw_x1
        pad_right = raw_x2 - x2
        pad_top = y1 - raw_y1
        pad_bottom = raw_y2 - y2
        crop = F.pad(crop, (pad_left, pad_right, pad_top, pad_bottom), value=0.0)
        # crop is now [box_h, box_w]

        resized = F.interpolate(
            crop.unsqueeze(0).unsqueeze(0),
            size=(out_h, out_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).squeeze(0)
        results.append(resized)

    return torch.stack(results, dim=0)  # [K, out_h, out_w]

class SegmentationLoss(LossWithWeights):
    """
    Cross-entropy loss between predicted coarse segmentation logits from the
    densepose head and ground truth instance masks.

    The densepose head is expected to output a tensor `pred_coarse_segm` of shape
    [N, C, H, W] where:
      - N = number of matched instances
      - C = number of coarse segmentation channels (2 for fg/bg)
      - H, W = spatial resolution of the densepose head output

    The GT masks are resized to (H, W) and converted to long labels (0=bg, 1=fg).
    """

    def __init__(
        self,
        weight_dict=None,
        compute_aux=False,
    ):
        super().__init__(weight_dict, compute_aux)
        self.n_segm_channels = 2
        self.target_keys.extend(["masks", "is_valid_mask"])

    def get_loss(self, outputs, targets, indices, num_boxes):
        pred_coarse_segm = outputs["pred_coarse_segm"]  # [N, C, H, W]

        if pred_coarse_segm is None or pred_coarse_segm.shape[0] == 0:
            return {"loss_coarse_segm": pred_coarse_segm.sum() * 0}

        if targets["masks"] is None:
            return {"loss_coarse_segm": pred_coarse_segm.sum() * 0}

        target_masks = (
            targets["masks"] if indices[2] is None else targets["masks"][indices[2]]
        )
        keep = (
            targets["is_valid_mask"]
            if indices[2] is None
            else targets["is_valid_mask"][indices[2]]
        )

        # Select matched predictions and their corresponding predicted boxes
        coarse_segm_est = pred_coarse_segm# [K, C, H, W]
        pred_boxes_xyxy = outputs["pred_boxes_xyxy"][(indices[0], indices[1])]  # [K, 4]

        # Filter to valid masks only
        # print(coarse_segm_est.shape)
        # print(target_masks.shape)
        # print(pred_boxes_xyxy.shape)
        # print(keep.shape)
        coarse_segm_est = coarse_segm_est[keep]
        target_masks = target_masks[keep]
        pred_boxes_xyxy = pred_boxes_xyxy[keep]

        if coarse_segm_est.shape[0] == 0:
            return {"loss_coarse_segm": pred_coarse_segm.sum() * 0}

        h, w = coarse_segm_est.shape[2], coarse_segm_est.shape[3]
        with torch.no_grad():
            # Crop GT masks to each predicted box, then resize to (h, w)
            # pred_boxes_xyxy are normalized [0,1], scale to pixel coords
            img_h, img_w = target_masks.shape[-2], target_masks.shape[-1]
            boxes_pixel = pred_boxes_xyxy.clone()
            boxes_pixel[:, 0] *= img_w
            boxes_pixel[:, 1] *= img_h
            boxes_pixel[:, 2] *= img_w
            boxes_pixel[:, 3] *= img_h

            # Crop and resize each GT mask to the predicted box region
            masks_gt = _crop_and_resize_masks(
                target_masks, boxes_pixel, h, w
            )  # [K, H, W]

            if self.n_segm_channels == 2:
                masks_gt = (masks_gt > 0.5).long()
            else:
                masks_gt = masks_gt.long()

        loss = F.cross_entropy(coarse_segm_est, masks_gt)

        return {"loss_coarse_segm": loss}
