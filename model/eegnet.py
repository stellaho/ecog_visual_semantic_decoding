"""
EEGNet implementation based on Lawhern et al. (2018).
"""

import torch
import torch.nn as nn


class Conv2dWithMaxNorm(nn.Conv2d):
    # Conv2d layer with optional max-norm constraint

    def __init__(self, *args, max_norm=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_norm = max_norm

    def forward(self, x):
        if self.max_norm is not None:
            with torch.no_grad():
                self.weight.renorm_(p=2, dim=0, maxnorm=self.max_norm)
            
            # Use a copy so later renorms do not affect this forward pass
            return self._conv_forward(x, self.weight.clone(), self.bias)
        return super().forward(x)


def same_temporal_padding(kernel_length):
    # Return left/right padding that preserves the temporal length
    total_padding = kernel_length - 1
    left = total_padding // 2
    right = total_padding - left
    return nn.ZeroPad2d((left, right, 0, 0))


class EEGNetEncoder(nn.Module):
    def __init__(
        self,
        n_inputs,
        f1=8,
        depth_multiplier=2,
        f2=None,
        temporal_kernel_length=64,
        separable_kernel_length=16,
        pool1_kernel=4,
        pool2_kernel=8,
        dropout=0.25,
        spatial_max_norm=1.0,
        pool_output=True,
    ):
        super().__init__()
        
        if f2 is None:
            f2 = f1 * depth_multiplier
        
        self.n_inputs = n_inputs
        self.f1 = f1
        self.depth_multiplier = depth_multiplier
        self.f2 = f2
        self.pool_output = pool_output
        self.n_outputs = f2
        
        self.temporal = nn.Sequential(
            same_temporal_padding(temporal_kernel_length),
            nn.Conv2d(
                in_channels=1,
                out_channels=f1,
                kernel_size=(1, temporal_kernel_length),
                bias=False,
            ),
            nn.BatchNorm2d(f1),
        )
        
        self.spatial = nn.Sequential(
            Conv2dWithMaxNorm(
                in_channels=f1,
                out_channels=f1 * depth_multiplier,
                kernel_size=(n_inputs, 1),
                groups=f1,
                bias=False,
                max_norm=spatial_max_norm,
            ),
            nn.BatchNorm2d(f1 * depth_multiplier),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, pool1_kernel)),
            nn.Dropout(dropout),
        )
        
        self.separable = nn.Sequential(
            same_temporal_padding(separable_kernel_length),
            nn.Conv2d(
                in_channels=f1 * depth_multiplier,
                out_channels=f1 * depth_multiplier,
                kernel_size=(1, separable_kernel_length),
                groups=f1 * depth_multiplier,
                bias=False,
            ),
            nn.Conv2d(
                in_channels=f1 * depth_multiplier,
                out_channels=f2,
                kernel_size=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, pool2_kernel)),
            nn.Dropout(dropout),
        )
        
        if pool_output:
            self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.temporal(x)
        x = self.spatial(x)
        x = self.separable(x)
        x = x.squeeze(2)
        
        if self.pool_output:
            x = self.pool(x).squeeze(-1)
        return x
