# Largely adapted or taken from Detectron2 framework: https://github.com/facebookresearch/detectron2/tree/main
import math
from functools import partial
from typing import Callable, List, Optional, Tuple, Union, Dict
from dataclasses import dataclass
from sam3.model.maskformer_segmentation import PixelDecoder
from sam3.model.focal_block import FocalModulationBlock


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import fvcore.nn.weight_init as weight_init
from sam3.model.poolers import ROIPooler, Boxes

from sam3.model.model_misc import (
    MultiheadAttentionWrapper as MultiheadAttention,
)



def initialize_module_params(module: nn.Module) -> None:
    for name, param in module.named_parameters():
        if "bias" in name:
            nn.init.constant_(param, 0)
        elif "weight" in name:
            nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")

def convert_prediction_boxes(pred_boxes, indices=None):
    if indices is not None:
        box_list = []
        for i in range(pred_boxes.shape[0]):
            mask = indices[0]==i
            box_list.append(Boxes(pred_boxes[indices[0][mask], indices[1][mask]]))
        return box_list

    return [Boxes(pred_boxes[i]) for i in range(pred_boxes.shape[0])]


class Conv2d(torch.nn.Conv2d):
    """
    A wrapper around :class:`torch.nn.Conv2d` to support empty inputs and more features.
    """

    def __init__(self, *args, **kwargs):
        """
        Extra keyword arguments supported in addition to those in `torch.nn.Conv2d`:

        Args:
            norm (nn.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation function

        It assumes that norm layer is used before activation.
        """
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)

        self.norm = norm
        self.activation = activation

    def forward(self, x):
        # torchscript does not support SyncBatchNorm yet
        # https://github.com/pytorch/pytorch/issues/40507
        # and we skip these codes in torchscript since:
        # 1. currently we only support torchscript in evaluation mode
        # 2. features needed by exporting module to torchscript are added in PyTorch 1.6 or
        # later version, `Conv2d` in these PyTorch versions has already supported empty inputs.
        # if not torch.jit.is_scripting():
        #     # Dynamo doesn't support context managers yet
        #     is_dynamo_compiling = check_if_dynamo_compiling()
        #     if not is_dynamo_compiling:
        #         with warnings.catch_warnings(record=True):
        #             if x.numel() == 0 and self.training:
        #                 # https://github.com/pytorch/pytorch/issues/12013
        #                 assert not isinstance(
        #                     self.norm, torch.nn.SyncBatchNorm
        #                 ), "SyncBatchNorm does not support empty inputs!"

        x = F.conv2d(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

class DensePoseDeepLabHead(nn.Module):
    """
    DensePose head using DeepLabV3 model from
    "Rethinking Atrous Convolution for Semantic Image Segmentation"
    <https://arxiv.org/abs/1706.05587>.
    """

    def __init__(self, input_channels: int):
        super(DensePoseDeepLabHead, self).__init__()
        # fmt: off
        hidden_dim           = 512
        kernel_size          = 3
        norm                 = "GN"
        self.n_stacked_convs = 8

        # fmt: on
        pad_size = kernel_size // 2
        n_channels = input_channels

        self.ASPP = ASPP(input_channels, [6, 12, 56], n_channels)  # 6, 12, 56
        self.add_module("ASPP", self.ASPP)
        self.focal_blocks = []

        for i in range(self.n_stacked_convs):
            norm_module = nn.GroupNorm(32, hidden_dim) if norm == "GN" else None
            layer = Conv2d(
                n_channels,
                hidden_dim,
                kernel_size,
                stride=1,
                padding=pad_size,
                bias=not norm,
                norm=norm_module,
            )
            weight_init.c2_msra_fill(layer)
            n_channels = hidden_dim
            layer_name = self._get_layer_name(i)
            self.add_module(layer_name, layer)
            if i % 2 == 1:
                focal_block = FocalModulationBlock(
                            channels= hidden_dim,
                            focal_levels=3,
                            focal_windows=[3,5,7],
                )
                self.focal_blocks.append(focal_block)
        self.focal_blocks = nn.ModuleList(self.focal_blocks)
        self.n_out_channels = hidden_dim
        # initialize_module_params(self)

    def forward(self, features):
        x0 = features
        x = self.ASPP(x0)
        output = x
        for i in range(self.n_stacked_convs):
            layer_name = self._get_layer_name(i)
            x = getattr(self, layer_name)(x)
            x = F.relu(x)
            if i % 2 == 1:
                x = self.focal_blocks[i//2](x)
            output = x
        return output

    def _get_layer_name(self, i: int):
        layer_name = "body_conv_fcn{}".format(i + 1)
        return layer_name


# Copied from
# https://github.com/pytorch/vision/blob/master/torchvision/models/segmentation/deeplabv3.py
# See https://arxiv.org/pdf/1706.05587.pdf for details
class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(
                in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False
            ),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(),
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels):
        super(ASPP, self).__init__()
        modules = []
        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.GroupNorm(32, out_channels),
                nn.ReLU(),
            )
        )

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            # nn.Dropout(0.5)
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


class DensePoseV1ConvXHead(nn.Module):
    """
    Fully convolutional DensePose head.
    """

    def __init__(self, input_channels: int):
        """
        Initialize DensePose fully convolutional head

        Args:
            cfg (CfgNode): configuration options
            input_channels (int): number of input channels
        """
        super(DensePoseV1ConvXHead, self).__init__()
        # fmt: off
        hidden_dim           = 512
        kernel_size          = 3
        self.n_stacked_convs = 8
        # fmt: on
        pad_size = kernel_size // 2
        n_channels = input_channels
        for i in range(self.n_stacked_convs):
            layer = Conv2d(n_channels, hidden_dim, kernel_size, stride=1, padding=pad_size)
            layer_name = self._get_layer_name(i)
            self.add_module(layer_name, layer)
            n_channels = hidden_dim
        self.n_out_channels = n_channels
        initialize_module_params(self)

    def forward(self, features: torch.Tensor):
        """
        Apply DensePose fully convolutional head to the input features

        Args:
            features (tensor): input features
        Result:
            A tensor of DensePose head outputs
        """
        x = features
        output = x
        for i in range(self.n_stacked_convs):
            layer_name = self._get_layer_name(i)
            x = getattr(self, layer_name)(x)
            x = F.relu(x)
            output = x
        return output

    def _get_layer_name(self, i: int):
        layer_name = "body_conv_fcn{}".format(i + 1)
        return layer_name


class DensePoseHead(nn.Module):
    def __init__(self, bb_channels = 256, 
        bb_scales = 288, #1x1 box pred to 288x288 feature map
        dp_pooler_resolution = 28,
        dp_pooler_sampling_ratio = 2,
        dp_pooler_type = "ROIAlignV2",
        dp_head_conv_dim = 512,
        dp_embed_dim = 16,
        dp_deconv_kernel = 4,
        use_deeplab_head = False,
        ):
        super().__init__()
        self.bb_channels = bb_channels
        self.bb_scales = bb_scales
        self.dp_pooler_resolution = dp_pooler_resolution
        self.dp_pooler_sampling_ratio = dp_pooler_sampling_ratio
        self.dp_pooler_type = dp_pooler_type
        self.dp_head_conv_dim = dp_head_conv_dim
        self.dp_embed_dim = dp_embed_dim
        self.dp_deconv_kernel = dp_deconv_kernel
        self.use_deeplab_head = use_deeplab_head
        self._init_densepose_head()

    def _init_densepose_head(self):       

        self.decoder = PixelDecoder(
            num_upsampling_stages=3,
            interpolation_mode="nearest",
            hidden_dim=self.bb_channels,
        )


        self.densepose_pooler = ROIPooler(
            output_size=self.dp_pooler_resolution,
            scales=[self.bb_scales],
            sampling_ratio=self.dp_pooler_sampling_ratio,
            pooler_type=self.dp_pooler_type,
        )

        self.cross_attend_prompt = MultiheadAttention(
            num_heads=8,
            dropout=0,
            embed_dim=256,
        )
        self.cross_attn_norm = nn.LayerNorm(256)

        if self.use_deeplab_head:
            self.densepose_head = DensePoseDeepLabHead(self.bb_channels)
        else:
            self.densepose_head = DensePoseV1ConvXHead(self.bb_channels)

        self.embed_lowres = torch.nn.ConvTranspose2d(
            self.dp_head_conv_dim, self.dp_embed_dim, self.dp_deconv_kernel, stride=2, padding=int(self.dp_deconv_kernel / 2 - 1)
        )
        nn.init.kaiming_normal_(self.embed_lowres.weight, mode="fan_out", nonlinearity="relu")
        nn.init.constant_(self.embed_lowres.bias, 0)

    def compute_densepose_outputs(self, features_list, output_boxes_xyxy, indices=None):
        pred_boxes = convert_prediction_boxes(output_boxes_xyxy, indices)
        features_dp = self.densepose_pooler(features_list, pred_boxes)
        if len(features_dp) > 0:
            densepose_head_outputs = self.densepose_head(features_dp)

            #transposed conv
            densepose_head_outputs = self.embed_lowres(densepose_head_outputs)
            densepose_head_outputs = F.interpolate(densepose_head_outputs, scale_factor=2, mode="bilinear", align_corners=False)

        else:
            densepose_head_outputs = None
        return densepose_head_outputs


    def forward(self, out, backbone_out, image_ids, encoder_hidden_states, prompt, prompt_mask):
        backbone_feats = backbone_out["backbone_fpn"]

        # Leverage encoder hidden states in a similar way that the segmentation head does.
        if self.cross_attend_prompt is not None:
            tgt2 = self.cross_attn_norm(encoder_hidden_states)
            tgt2 = self.cross_attend_prompt(
                query=tgt2,
                key=prompt,
                value=prompt,
                key_padding_mask=prompt_mask,
            )[0]
            encoder_hidden_states = tgt2 + encoder_hidden_states

        if backbone_feats[0].shape[0] > 1:
                # For bs > 1, we construct the per query backbone features
                backbone_visual_feats = []
                for feat in backbone_feats:
                    # Copy the img features per query (pixel decoder won't share img feats)
                    backbone_visual_feats.append(feat[image_ids, ...].to(backbone_feats[0].device))
        else:
            # Bs=1, we rely on broadcasting for query-based processing
            backbone_visual_feats = [bb_feat.clone() for bb_feat in backbone_feats]
        # Extract visual embeddings
        encoder_hidden_states = encoder_hidden_states.permute(1, 2, 0)
        spatial_dim = math.prod(backbone_feats[-1].shape[-2:])
        encoder_visual_embed = encoder_hidden_states[..., :spatial_dim].reshape(
            -1, *backbone_feats[-1].shape[1:]
        )

        backbone_visual_feats[-1] = encoder_visual_embed
        pixel_embed = self.decoder(backbone_visual_feats)

        features_list = [pixel_embed]

        indices = None
        indices_o2m = None

        # If we have indices for matched boxes, only use those matches in densepose head
        if "indices" in out.keys():
            indices = out["indices"]
        densepose_predictor_outputs = self.compute_densepose_outputs(features_list, out["pred_boxes_xyxy"], indices)

        densepose_predictor_outputs_o2m = None
        if "pred_boxes_xyxy_o2m" in out.keys():
            # If we have indices for matched o2m boxes, only use those in densepose head
            if "indices_o2m" in out.keys():
                indices_o2m = out["indices_o2m"]
            densepose_predictor_outputs_o2m = self.compute_densepose_outputs(features_list, out["pred_boxes_xyxy_o2m"], indices_o2m)

        return densepose_predictor_outputs, densepose_predictor_outputs_o2m
