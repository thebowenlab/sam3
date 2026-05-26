import os
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np

import torch
from sam3.model.model_misc import SAM3Output
from sam3.model.sam1_task_predictor import SAM3InteractiveImagePredictor
from sam3.model.vl_combiner import SAM3VLBackbone
from sam3.perflib.nms import nms_masks
from sam3.train.data.collator import BatchedDatapoint
from sam3.model.sam3_image import Sam3Image

from .act_ckpt_utils import activation_ckpt_wrapper
from .box_ops import box_cxcywh_to_xyxy
from .geometry_encoders import Prompt
from .model_misc import inverse_sigmoid



class Sam3DensePoseImage(Sam3Image):
    def __init__(
        self,
        backbone: SAM3VLBackbone,
        transformer,
        input_geometry_encoder,
        densepose_head,
        segmentation_head=None,
        cse_embedder=None,
        num_feature_levels=1,
        o2m_mask_predict=True,
        dot_prod_scoring=None,
        use_instance_query: bool = True,
        multimask_output: bool = True,
        use_act_checkpoint_seg_head: bool = True,
        interactivity_in_encoder: bool = True,
        matcher=None,
        use_dot_prod_scoring=True,
        supervise_joint_box_scores: bool = False,  # only relevant if using presence token/score
        detach_presence_in_joint_score: bool = False,  # only relevant if using presence token/score
        separate_scorer_for_instance: bool = False,
        num_interactive_steps_val: int = 0,
        inst_interactive_predictor: SAM3InteractiveImagePredictor = None,
        **kwargs,
    ):
        super().__init__(backbone, 
            transformer,
            input_geometry_encoder,
            segmentation_head,
            num_feature_levels,
            o2m_mask_predict,
            dot_prod_scoring,
            use_instance_query,
            multimask_output,
            use_act_checkpoint_seg_head,
            interactivity_in_encoder,
            matcher,
            use_dot_prod_scoring,
            supervise_joint_box_scores,
            detach_presence_in_joint_score,
            separate_scorer_for_instance,
            num_interactive_steps_val,
            inst_interactive_predictor,
            **kwargs)
        self.densepose_head = densepose_head
        self.cse_embedder = cse_embedder

    def _run_densepose_head(self, out, backbone_out, img_ids, encoder_hidden_states, prompt, prompt_mask):
 
        densepose_head_outputs, densepose_head_outputs_o2m = self.densepose_head(
                out=out,
                backbone_out=backbone_out,
                image_ids=img_ids,
                encoder_hidden_states=encoder_hidden_states,
                prompt=prompt,
                prompt_mask=prompt_mask,
            )
        out["pred_embeddings"] = densepose_head_outputs
        out["pred_embeddings_o2m"] = densepose_head_outputs_o2m


    def forward_grounding(
        self,
        backbone_out,
        find_input,
        find_target,
        geometric_prompt: Prompt,
    ):
        with torch.profiler.record_function("SAM3DensePoseImage._encode_prompt"):
            prompt, prompt_mask, backbone_out = self._encode_prompt(
                backbone_out, find_input, geometric_prompt
            )
        # Run the encoder
        with torch.profiler.record_function("SAM3DensePoseImage._run_encoder"):
            backbone_out, encoder_out, _ = self._run_encoder(
                backbone_out, find_input, prompt, prompt_mask
            )
        out = {
            "encoder_hidden_states": encoder_out["encoder_hidden_states"],
            "prev_encoder_out": {
                "encoder_out": encoder_out,
                "backbone_out": backbone_out,
            },
        }

        # Run the decoder
        with torch.profiler.record_function("SAM3DensePoseImage._run_decoder"):
            out, hs = self._run_decoder(
                memory=out["encoder_hidden_states"],
                pos_embed=encoder_out["pos_embed"],
                src_mask=encoder_out["padding_mask"],
                out=out,
                prompt=prompt,
                prompt_mask=prompt_mask,
                encoder_out=encoder_out,
            )

        # Run segmentation heads
        with torch.profiler.record_function("SAM3DensePoseImage._run_segmentation_heads"):
            self._run_segmentation_heads(
                out=out,
                backbone_out=backbone_out,
                img_ids=find_input.img_ids,
                vis_feat_sizes=encoder_out["vis_feat_sizes"],
                encoder_hidden_states=out["encoder_hidden_states"],
                prompt=prompt,
                prompt_mask=prompt_mask,
                hs=hs,
            )

        if self.training or self.num_interactive_steps_val > 0:
            self._compute_matching(out, self.back_convert(find_target))

        # Run densepose head
        with torch.profiler.record_function("SAM3DensePoseImage._run_densepose_head"):
            self._run_densepose_head(
                out=out,
                backbone_out=backbone_out,
                img_ids=find_input.img_ids,
                encoder_hidden_states=out["encoder_hidden_states"],
                prompt=prompt,
                prompt_mask=prompt_mask,
            )


        return out

    def back_convert(self, targets):
        batched_targets = {
            "boxes": targets.boxes.view(-1, 4),
            "boxes_xyxy": box_cxcywh_to_xyxy(targets.boxes.view(-1, 4)),
            "boxes_padded": targets.boxes_padded,
            "positive_map": targets.boxes.new_ones(len(targets.boxes), 1),
            "num_boxes": targets.num_boxes,
            "masks": targets.segments,
            "semantic_masks": targets.semantic_segments,
            "is_valid_mask": targets.is_valid_segment,
            "is_exhaustive": targets.is_exhaustive,
            "object_ids_packed": targets.object_ids,
            "object_ids_padded": targets.object_ids_padded,
            "dp_vertex": targets.dp_vertices,
            "dp_x": targets.dp_xs,
            "dp_y": targets.dp_ys,
            "ref_model": targets.ref_model,
            "img_id": targets.img_ids,
        }
        return batched_targets


    def forward(self, input: BatchedDatapoint):
        
        device = self.device
        backbone_out = {"img_batch_all_stages": input.img_batch}
        backbone_out.update(self.backbone.forward_image(input.img_batch))

        num_frames = len(input.find_inputs)
        assert num_frames == 1

        text_outputs = self.backbone.forward_text(input.find_text_batch, device=device)
        backbone_out.update(text_outputs)

        previous_stages_out = SAM3Output(
            iter_mode=SAM3Output.IterMode.LAST_STEP_PER_STAGE
        )

        find_input = input.find_inputs[0]
        find_target = input.find_targets[0]

        if find_input.input_points is not None and find_input.input_points.numel() > 0:
            print("Warning: Point prompts are ignored in PCS.")

        num_interactive_steps = 0 if self.training else self.num_interactive_steps_val
        geometric_prompt = Prompt(
            box_embeddings=find_input.input_boxes,
            box_mask=find_input.input_boxes_mask,
            box_labels=find_input.input_boxes_label,
        )

        # Init vars that are shared across the loop.
        stage_outs = []
        for cur_step in range(num_interactive_steps + 1):
            if cur_step > 0:
                # We sample interactive geometric prompts (boxes, points)
                geometric_prompt, _ = self.interactive_prompt_sampler.sample(
                    geo_prompt=geometric_prompt,
                    find_target=find_target,
                    previous_out=stage_outs[-1],
                )
            out = self.forward_grounding(
                backbone_out=backbone_out,
                find_input=find_input,
                find_target=find_target,
                geometric_prompt=geometric_prompt.clone(),
            )
            out_ref_model = {} # dict of mesh name -> mesh embeddings
            for ref_model in find_target.ref_model:  # get embeddings for each mesh in this batch
                if ref_model is not None:
                    out_ref_model[ref_model] = self.cse_embedder(ref_model)
            out["mesh_embeddings"] = out_ref_model
            stage_outs.append(out)


        previous_stages_out.append(stage_outs)
        return previous_stages_out
