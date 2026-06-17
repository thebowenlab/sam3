"""SAM3-compatible meters for CSE evaluation."""

import json
import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from sam3.eval.cse_evaluator import (
    PerPointGPSEvaluator,
)

from sam3.model.data_misc import BatchedInferenceMetadata, interpolate


from sam3.model.box_ops import box_cxcywh_to_xywh


logger = logging.getLogger(__name__)

class CSEPerPointGPSMeter:
    """
    Meter that accumulates per-instance OGPS scores during validation
    and computes AP_GPS at epoch end.
    """

    def __init__(self, detection_threshold = .3, maxdets = 100, use_presence = True, default_sigma: float = 0.255, cat_to_mesh = {}, out_dir = None):
        self.gps_evaluator = PerPointGPSEvaluator(
            default_sigma=default_sigma,
            out_dir=out_dir,
            use_gpsm=False,
        )
        self.gpsm_evaluator = PerPointGPSEvaluator(
            default_sigma=default_sigma,
            out_dir=out_dir,
            use_gpsm=True,
        )
        self.detection_threshold = detection_threshold
        self.maxdets = maxdets
        self.use_presence = use_presence
        self._predictions = []
        self._ground_truths = []
        self._mesh_embeddings = {}
        self.cat_to_mesh = cat_to_mesh
        self.out_dir=out_dir

    def reset(self):
        self._predictions.clear()
        self._ground_truths.clear()
        self._mesh_embeddings.clear()

    def _process_boxes_and_labels(self, target_sizes, forced_labels, out_bbox, out_probs):
        if out_bbox is None:
            return None, None, None, None
        assert len(out_probs) == len(target_sizes)
        
        out_probs = out_probs.cpu()
        scores, labels = out_probs.max(-1)
        if forced_labels is None:
            labels = torch.ones_like(labels)
        else:
            labels = forced_labels[:, None].expand_as(labels)

        # convert to [x0, y0, x1, y1] format
        boxes = box_cxcywh_to_xywh(out_bbox)

        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        
        boxes = boxes.cpu()

        keep = None
        if self.detection_threshold > 0:
            # Filter out the boxes with scores below the detection threshold
            keep = scores > self.detection_threshold
            assert len(keep) == len(boxes) == len(scores) == len(labels)

            boxes = [b[k.to(b.device)] for b, k in zip(boxes, keep)]
            scores = [s[k.to(s.device)].to(torch.float16) for s, k in zip(scores, keep)]
            labels = [l[k.to(l.device)] for l, k in zip(labels, keep)]

        return boxes, scores, labels, keep

    def _process_masks(self, target_sizes, pred_masks, keep=None):
        if pred_masks is None:
            return None
        
        out_masks = [[]] * len(pred_masks)

        assert keep is None or len(keep) == len(pred_masks)
        for i, mask in enumerate(pred_masks):
            h, w = target_sizes[i]
            if keep is not None:
                mask = mask[keep[i]]
            # Uses the gpu version fist, moves masks to cpu if it fails"""
            try:
                interpolated = (
                    interpolate(
                        mask.unsqueeze(1),
                        (h, w),
                        mode="bilinear",
                        align_corners=False,
                    ).sigmoid()
                    > 0.5
                )
            except Exception as e:
                logging.info("Issue found, reverting to CPU mode!")
                mask_device = mask.device
                mask = mask.cpu()
                interpolated = (
                    interpolate(
                        mask.unsqueeze(1),
                        (h, w),
                        mode="bilinear",
                        align_corners=False,
                    ).sigmoid()
                    > 0.5
                )
                interpolated = interpolated.to(mask_device)

            out_masks[i] = interpolated
            out_masks[i] = out_masks[i].cpu()
        return out_masks


    def process_output(self, outputs, metadata):
        out_bbox = outputs["pred_boxes"] if "pred_boxes" in outputs else None
        out_logits = outputs["pred_logits"]
        pred_masks = outputs["pred_masks"]
        out_probs = out_logits.sigmoid()
        if self.use_presence:
            presence_score = outputs["presence_logit_dec"].sigmoid().unsqueeze(1)
            out_probs = out_probs * presence_score


        target_sizes_boxes = target_sizes_masks = metadata.original_size
        assert target_sizes_boxes.shape[1] == 2
        assert target_sizes_masks.shape[1] == 2
        batch_size = target_sizes_boxes.shape[0]

        boxes, scores, labels, keep = self._process_boxes_and_labels(
            target_sizes_boxes, metadata.original_category_id, out_bbox, out_probs
        )
        
        assert boxes is None or len(boxes) == batch_size

        out_masks = self._process_masks(
            target_sizes_masks, pred_masks, keep=keep
        )
        del pred_masks

        for mesh_name in outputs["mesh_embeddings"].keys():
            if mesh_name not in self._mesh_embeddings.keys():
                self._mesh_embeddings[mesh_name] = outputs["mesh_embeddings"][mesh_name].cpu()

        # go batch by batch, add only samples that have detections
        for i in range(len(keep)):
            img_ids = metadata.original_image_id[i]
            predictions_per_batch = keep.shape[1]
            embedding_index_start = i*predictions_per_batch
            batch_embeddings = outputs["pred_embeddings"][embedding_index_start:embedding_index_start+predictions_per_batch]
            batch_embeddings = batch_embeddings[keep[i]]
            for j in range(len(boxes[i])):
                self._predictions.append({
                        "image_id": metadata.original_image_id[i].item(),
                        "bbox": boxes[i][j],
                        "score": scores[i][j],
                        "embedding": batch_embeddings[j].cpu(),
                        "mask": out_masks[i][j].squeeze(0),
                        "mesh_name": self.cat_to_mesh[metadata.original_category_id[i].item()],
                    })
    
    def process_targets(self, targets, metadata):
        for i in range(len(targets["boxes"])):
            if targets["ref_model"][i] is not None:
                self._ground_truths.append({
                    "image_id": targets["img_id"][i],
                    "bbox": box_cxcywh_to_xywh(targets["boxes"][i].cpu()),
                    "dp_vertex": targets["dp_vertex"][i].cpu(),
                    "dp_x": targets["dp_x"][i].cpu(),
                    "dp_y": targets["dp_y"][i].cpu(),
                    "mesh_name": targets["ref_model"][i],
                    "mask": targets["masks"][i].cpu(),
                })



    def update(self, find_stages, find_metadatas, find_targets, **kwargs):
        """
        Called per-batch. Accumulate predictions and GT for later AP computation.
        You'll need to adapt this to match your SAM3 output format.
        """
        for output, metadata, targets in zip(find_stages, find_metadatas, find_targets):
            self.process_output(output, metadata)
            self.process_targets(targets, metadata)

    def compute_synced(self) -> Dict[str, float]:
        if not self._predictions:
            return {"AP_GPS": 0.0, "AP50_GPS": 0.0, "AP75_GPS": 0.0, "AR_GPS": 0.0}
        results = self.gps_evaluator.evaluate_dataset(
            self._predictions, self._ground_truths, self._mesh_embeddings
        )
        results_gpsm = self.gpsm_evaluator.evaluate_dataset(
            self._predictions, self._ground_truths, self._mesh_embeddings
        )
        return {
            "AP_GPS": results["AP"],
            "AP50_GPS": results["AP50"],
            "AP75_GPS": results["AP75"],
            "AR_GPS": results["AR"],
            "AP_GPSM": results_gpsm["AP"],
            "AP50_GPSM": results_gpsm["AP50"],
            "AP75_GPSM": results_gpsm["AP75"],
            "AR_GPSM": results_gpsm["AR"],
        }
    def compute(self):
        """
        Compute without synchronization.

        Returns:
            Empty metric dictionary.
        """
        return {"": 0.0}

    @staticmethod
    def is_better(new_val, old_val):
        return new_val > old_val