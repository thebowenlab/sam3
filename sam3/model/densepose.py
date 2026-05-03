# All code here adapted or taken from Detectron2 framework: https://github.com/facebookresearch/detectron2/tree/b599f139756bd3646a26a909caf86a1a159e53a7 
import math
from functools import partial
from typing import Callable, List, Optional, Tuple, Union, Dict
from dataclasses import dataclass


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from sam3.model.poolers import ROIPooler, Boxes



def initialize_module_params(module: nn.Module) -> None:
    for name, param in module.named_parameters():
        if "bias" in name:
            nn.init.constant_(param, 0)
        elif "weight" in name:
            nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")

def convert_prediction_boxes(pred_boxes):
    return [Boxes(pred_boxes[i]) for i in range(pred_boxes.shape[0])]

@dataclass
class ShapeSpec:
    """
    A simple structure that contains basic shape specification about a tensor.
    It is often used as the auxiliary inputs/outputs of models,
    to complement the lack of shape inference ability among pytorch modules.
    """

    channels: Optional[int] = None
    height: Optional[int] = None
    width: Optional[int] = None
    stride: Optional[int] = None

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
        dp_deconv_kernel = 4):
        super().__init__()
        self.bb_channels = bb_channels
        self.bb_scales = bb_scales
        self.dp_pooler_resolution = dp_pooler_resolution
        self.dp_pooler_sampling_ratio = dp_pooler_sampling_ratio
        self.dp_pooler_type = dp_pooler_type
        self.dp_head_conv_dim = dp_head_conv_dim
        self.dp_embed_dim = dp_embed_dim
        self.dp_deconv_kernel = dp_deconv_kernel
        self._init_densepose_head()

    def _init_densepose_head(self):
        # self.densepose_data_filter = build_densepose_data_filter(cfg)
       


        self.densepose_pooler = ROIPooler(
            output_size=self.dp_pooler_resolution,
            scales=[self.bb_scales],
            sampling_ratio=self.dp_pooler_sampling_ratio,
            pooler_type=self.dp_pooler_type,
        )
        self.densepose_head = DensePoseV1ConvXHead(self.bb_channels)

        self.embed_lowres = torch.nn.ConvTranspose2d(
            self.dp_head_conv_dim, self.dp_embed_dim, self.dp_deconv_kernel, stride=2, padding=int(self.dp_deconv_kernel / 2 - 1)
        )

        # self.densepose_losses = build_densepose_losses(cfg)
        # self.embedder = build_densepose_embedder(cfg)


    def compute_densepose_outputs(self, features_list, output_boxes_xyxy):
        pred_boxes = convert_prediction_boxes(output_boxes_xyxy)
        features_dp = self.densepose_pooler(features_list, pred_boxes)
        if len(features_dp) > 0:
            densepose_head_outputs = self.densepose_head(features_dp)
            densepose_predictor_outputs = self.embed_lowres(densepose_head_outputs)
            densepose_predictor_outputs = F.interpolate(densepose_predictor_outputs, scale_factor=2, mode="bilinear", align_corners=False)
        else:
            densepose_predictor_outputs = None
        return densepose_predictor_outputs


    def forward(self, out, backbone_out):

        features_list = [out["pixel_embed"]]

        densepose_predictor_outputs = self.compute_densepose_outputs(features_list, out["pred_boxes_xyxy"])
        densepose_predictor_outputs_o2m = None
        if "pred_boxes_xyxy_o2m" in out.keys():
            densepose_predictor_outputs_o2m = self.compute_densepose_outputs(features_list, out["pred_boxes_xyxy_o2m"])


        # densepose_inference(densepose_predictor_outputs, instances)
        return densepose_predictor_outputs, densepose_predictor_outputs_o2m
