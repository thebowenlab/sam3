import torch.nn as nn
import torch

# Implementation of focal modulation blocks from https://arxiv.org/pdf/2203.11926

class FocalModulationBlock(nn.Module):
    """
    Focal modulation block for feature maps.
    """
    def __init__(self, channels, focal_levels=3, focal_windows=[3, 5, 7], focal_factor=2, proj_drop=0.0):
        super().__init__()
        self.channels = channels
        self.focal_levels = focal_levels

        # Linear projections for q (query/input), ctx (context), gates
        self.f = nn.Linear(channels, 2 * channels + (focal_levels + 1), bias=True)

        # Hierarchical contextualization layers (depth-wise convs at increasing kernel sizes)
        self.focal_layers = nn.ModuleList()
        for k in range(focal_levels):
            kernel_size = focal_windows[k]
            self.focal_layers.append(nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=kernel_size, 
                         padding=kernel_size // 2, groups=channels, bias=False),
                nn.GELU(),
            ))

        self.h = nn.Conv2d(channels, channels, kernel_size=1, stride=1, bias=True)
        self.proj = nn.Linear(channels, channels)
        self.proj_drop = nn.Dropout(proj_drop)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) feature map from preceding conv+relu
        """
        B, C, H, W = x.shape
        residual = x

        # Reshape to (B, H, W, C) for linear projections
        x_flat = x.permute(0, 2, 3, 1)
        x_flat = self.norm(x_flat)

        # Project to query, context seed, and gates
        qcg = self.f(x_flat)  # (B, H, W, 2*C + focal_levels + 1)
        q, ctx, gates = torch.split(qcg, [C, C, self.focal_levels + 1], dim=-1)

        # Permute context back to spatial for depth-wise convs, (BHWC) -> (BCHW)
        q = q.permute(0, 3, 1, 2)
        ctx = ctx.permute(0, 3, 1, 2)  # (B, C, H, W)
        gates = gates.permute(0, 3, 1, 2)  # (B, levels+1, H, W)

        # Hierarchical context aggregation
        ctx_all = torch.zeros_like(ctx)
        for lvl, focal_layer in enumerate(self.focal_layers):
            ctx = focal_layer(ctx)  # progressively larger receptive field
            ctx_all = ctx_all + ctx * gates[:, lvl:lvl+1, :, :]

        ctx_global = nn.functional.gelu(ctx.mean(dim=(-2, -1), keepdim=True))
        ctx_all = ctx_all + ctx_global * gates[:, self.focal_levels:]

        # Project context into modulator, then multiply with query
        modulator = self.h(ctx_all)         # 1×1 conv on (B, C, H, W)
        x_out = q * modulator

        # Output projection
        x_out = x_out.permute(0, 2, 3, 1)  # (B, H, W, C)
        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out)

        # Reshape back to (B, C, H, W) and add residual
        x_out = x_out.permute(0, 3, 1, 2)
        return x_out + residual