"""
Adapted from Ultralytics
https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/plotting.py

Standalone batch visualization for SAM3 training data (sam3_densepose_head branch).

Plots bounding boxes, segmentation masks, and DensePose ground truth points
from SAM3's BatchedDatapoint format.

Usage:
    from plot_sam3_batches import plot_sam3_batches, visualize_n_batches

    visualize_n_batches(dataloader, n_batches=5, save_dir="vis_out")
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
from PIL import Image

# ─── Color palette ────────────────────────────────────────────────────────────

_HEXS = (
    "042AFF", "0BDBEB", "F3F3F3", "00DFB7", "111F68",
    "FF6FDD", "FF444F", "CCED00", "00F344", "BD00FF",
    "00B4FF", "DD00BA", "00FFFF", "26C000", "01FFB3",
    "7D24FF", "7B0068", "FF1B6C", "FC6D2F", "A2FF0B",
)
PALETTE = [tuple(int(h[i:i+2], 16) for i in (0, 2, 4)) for h in _HEXS]


def _color(idx: int) -> Tuple[int, int, int]:
    return PALETTE[idx % len(PALETTE)]


# ─── SAM3 normalization constants ─────────────────────────────────────────────

MEAN_IMG = np.array([0.5, 0.5, 0.5])
STD_IMG = np.array([0.5, 0.5, 0.5])


def _denormalize_image(img_tensor: torch.Tensor) -> np.ndarray:
    """Convert SAM3-normalized [C,H,W] tensor to uint8 RGB [H,W,3]."""
    img = img_tensor.detach().cpu().float().numpy().transpose(1, 2, 0)
    img = (img * STD_IMG) + MEAN_IMG
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img


# ─── Drawing helpers ──────────────────────────────────────────────────────────

def _draw_bbox_on_image(canvas, box_cxcywh, color, thickness=2, label=""):
    """Draw a normalized CxCyWH bounding box. Returns (x1,y1,x2,y2) in pixels."""
    h, w = canvas.shape[:2]
    cx, cy, bw, bh = box_cxcywh
    x1 = int((cx - bw / 2) * w)
    y1 = int((cy - bh / 2) * h)
    x2 = int((cx + bw / 2) * w)
    y2 = int((cy + bh / 2) * h)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
    if label:
        fs, ft = 0.4, 1
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, ft)
        cv2.rectangle(canvas, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(canvas, label, (x1, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), ft, cv2.LINE_AA)
    return x1, y1, x2, y2


def _overlay_mask(canvas, mask, color, alpha=0.45):
    """Overlay a binary mask with transparency."""
    if mask.shape[:2] != canvas.shape[:2]:
        mask = cv2.resize(mask.astype(np.uint8), (canvas.shape[1], canvas.shape[0]),
                          interpolation=cv2.INTER_NEAREST)
    mask_bool = mask.astype(bool)
    color_arr = np.array(color, dtype=np.float32)
    canvas[mask_bool] = (
        canvas[mask_bool].astype(np.float32) * (1 - alpha) + color_arr * alpha
    ).astype(np.uint8)


def _is_valid_seg(is_valid_segment, idx):
    if is_valid_segment is None:
        return True
    if isinstance(is_valid_segment, torch.Tensor):
        return bool(is_valid_segment[idx]) if idx < len(is_valid_segment) else True
    if isinstance(is_valid_segment, (list, np.ndarray)):
        return bool(is_valid_segment[idx]) if idx < len(is_valid_segment) else True
    return True


def _draw_segment_overlay(canvas, segments, is_valid_segment, box_idx, color):
    """Draw a single segment mask overlay."""
    if isinstance(segments, torch.Tensor):
        seg_arr = segments.detach().cpu().numpy()
        if seg_arr.ndim >= 2 and box_idx < len(seg_arr):
            if _is_valid_seg(is_valid_segment, box_idx) and seg_arr[box_idx].any():
                _overlay_mask(canvas, seg_arr[box_idx], color)
    elif isinstance(segments, list) and box_idx < len(segments):
        seg_item = segments[box_idx]
        if isinstance(seg_item, torch.Tensor):
            seg_np = seg_item.detach().cpu().numpy()
        else:
            seg_np = np.asarray(seg_item)
        if _is_valid_seg(is_valid_segment, box_idx) and seg_np.any():
            _overlay_mask(canvas, seg_np, color)


def _draw_densepose_points(canvas, dp_xs, dp_ys, box_cxcywh, color, radius=2):
    """
    Draw DensePose points on the canvas.

    dp_x/dp_y are in a 256x256 local coordinate frame within the bounding box.
    We map them to pixel coordinates using the box location.
    """
    h_img, w_img = canvas.shape[:2]
    cx, cy, bw, bh = box_cxcywh

    # Box pixel coords
    box_x1 = (cx - bw / 2) * w_img
    box_y1 = (cy - bh / 2) * h_img
    box_w = bw * w_img
    box_h = bh * h_img

    for pi in range(len(dp_xs)):
        # dp_x, dp_y are in [0, 255] local box coordinates
        local_x = float(dp_xs[pi])
        local_y = float(dp_ys[pi])

        # Map from 256x256 local frame to image pixel coordinates
        px = int(box_x1 + (local_x / 255.0) * box_w)
        py = int(box_y1 + (local_y / 255.0) * box_h)

        # Clamp to image bounds
        px = max(0, min(px, w_img - 1))
        py = max(0, min(py, h_img - 1))

        cv2.circle(canvas, (px, py), radius + 1, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(canvas, (px, py), radius, color, -1, cv2.LINE_AA)


# ─── Render a single BatchedDatapoint as a mosaic ─────────────────────────────

def _render_single_batched_dp(
    batched_dp,
    save_path: Path,
    filename: str,
    max_images: int = 16,
    plot_masks: bool = True,
    plot_densepose: bool = True,
):
    """Render one BatchedDatapoint into a mosaic grid and save it."""
    img_batch = batched_dp.img_batch
    if isinstance(img_batch, torch.Tensor):
        img_batch = img_batch.detach().cpu()

    find_targets = batched_dp.find_targets
    find_inputs = batched_dp.find_inputs
    text_batch = batched_dp.find_text_batch

    B = min(img_batch.shape[0], max_images)
    ns = int(math.ceil(B ** 0.5))
    h, w = img_batch.shape[2], img_batch.shape[3]

    mosaic = np.full((ns * h, ns * w, 3), 240, dtype=np.uint8)

    for i in range(B):
        row, col = divmod(i, ns)
        y_off, x_off = row * h, col * w
        canvas = _denormalize_image(img_batch[i]).copy()

        for ft, fi in zip(find_targets, find_inputs):
            # ── Determine which queries map to this image ─────────
            img_ids = fi.img_ids
            if isinstance(img_ids, torch.Tensor):
                img_ids = img_ids.detach().cpu().numpy()
            else:
                img_ids = np.array(img_ids)

            query_mask = img_ids == i
            if not query_mask.any():
                continue

            query_indices = np.where(query_mask)[0]

            # ── Unpack target fields ──────────────────────────────
            num_boxes = ft.num_boxes
            if isinstance(num_boxes, torch.Tensor):
                num_boxes = num_boxes.detach().cpu().numpy()
            else:
                num_boxes = np.array(num_boxes)

            boxes_all = ft.boxes
            if isinstance(boxes_all, torch.Tensor):
                boxes_all = boxes_all.detach().cpu().numpy()

            segments = ft.segments
            is_valid_segment = ft.is_valid_segment

            # DensePose fields (packed per-object, same order as boxes)
            dp_xs_all = getattr(ft, 'dp_xs', None)
            dp_ys_all = getattr(ft, 'dp_ys', None)
            dp_vertices_all = getattr(ft, 'dp_vertices', None)

            box_offset = np.concatenate([[0], np.cumsum(num_boxes)])

            # ── Draw each object ──────────────────────────────────
            obj_counter = 0
            for q_idx in query_indices:
                n_boxes_q = int(num_boxes[q_idx])
                start = int(box_offset[q_idx])

                # Get query text
                text_ids = fi.text_ids
                if isinstance(text_ids, torch.Tensor):
                    text_ids = text_ids.detach().cpu().numpy()
                text_id = int(text_ids[q_idx]) if q_idx < len(text_ids) else 0
                query_text = text_batch[text_id] if text_id < len(text_batch) else ""
                short_text = query_text[:20]

                for j in range(n_boxes_q):
                    box_idx = start + j
                    if box_idx >= len(boxes_all):
                        break

                    box = boxes_all[box_idx][0]  # [cx, cy, w, h] normalized
                    color = _color(obj_counter)
                    label = short_text if short_text else f"obj{obj_counter}"

                    # Draw bounding box
                    _draw_bbox_on_image(canvas, box, color, thickness=2, label=label)

                    # Draw mask overlay
                    if plot_masks and segments is not None:
                        _draw_segment_overlay(
                            canvas, segments, is_valid_segment, box_idx, color
                        )

                    # Draw DensePose points
                    if plot_densepose and dp_xs_all is not None and dp_ys_all is not None:
                        # Get this object's densepose data
                        if box_idx < len(dp_xs_all):
                            dp_x = dp_xs_all[box_idx]
                            dp_y = dp_ys_all[box_idx]

                            # Skip if None (no densepose annotation for this object)
                            if dp_x is not None and dp_y is not None:
                                if isinstance(dp_x, torch.Tensor):
                                    dp_x = dp_x.detach().cpu().numpy()
                                if isinstance(dp_y, torch.Tensor):
                                    dp_y = dp_y.detach().cpu().numpy()

                                dp_x = np.asarray(dp_x, dtype=np.float32)
                                dp_y = np.asarray(dp_y, dtype=np.float32)

                                if len(dp_x) > 0:
                                    _draw_densepose_points(
                                        canvas, dp_x, dp_y, box, color
                                    )

                    obj_counter += 1

        mosaic[y_off:y_off + h, x_off:x_off + w] = canvas

    # Grid lines
    for i in range(1, ns):
        cv2.line(mosaic, (i * w, 0), (i * w, ns * h), (200, 200, 200), 1)
        cv2.line(mosaic, (0, i * h), (ns * w, i * h), (200, 200, 200), 1)

    out_file = save_path / filename
    Image.fromarray(mosaic).save(str(out_file), quality=92)
    print(f"Saved: {out_file}")


# ─── Main entry point ────────────────────────────────────────────────────────

def plot_sam3_batches(
    batch: list,
    batch_idx: int = 0,
    save_dir: str = "vis_out",
    max_images_per_batch: int = 16,
    plot_masks: bool = True,
    plot_densepose: bool = True,
):
    """
    Visualize a batch from SAM3's dataloader.

    Always expects a list of {"key": BatchedDatapoint} dicts.
    Each chunk gets its own output mosaic image.

    Args:
        batch: List of {"key": BatchedDatapoint} dicts.
        batch_idx: Current batch index (for output filenames).
        save_dir: Output directory.
        max_images_per_batch: Max images per mosaic grid.
        plot_masks: Whether to draw segmentation mask overlays.
        plot_densepose: Whether to draw DensePose point annotations.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    for chunk_idx, chunk_dict in enumerate(batch):
        _, batched_dp = next(iter(chunk_dict.items()))

        if len(batch) == 1:
            filename = f"batch_{batch_idx:04d}.jpg"
        else:
            filename = f"batch_{batch_idx:04d}_chunk{chunk_idx}.jpg"

        _render_single_batched_dp(
            batched_dp,
            save_path=save_path,
            filename=filename,
            max_images=max_images_per_batch,
            plot_masks=plot_masks,
            plot_densepose=plot_densepose,
        )


# ─── Convenience runner ───────────────────────────────────────────────────────

def visualize_n_batches(dataloader, n_batches=5, save_dir="vis_out", **kwargs):
    """Iterate a SAM3 DataLoader and save the first n_batches as mosaics."""
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= n_batches:
            break
        # Wrap single dict in list for uniform handling
        if isinstance(batch, dict):
            batch = [batch]
        plot_sam3_batches(batch, batch_idx=batch_idx, save_dir=save_dir, **kwargs)
    print(f"Done. Visualized {min(n_batches, batch_idx + 1)} batches → {save_dir}/")