import math
from functools import partial
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

class DensePoseHead(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, backbone_out, proposals):
        print("denseposehead forward")
        return backbone_out
