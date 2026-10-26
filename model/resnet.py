"""
1D ResNet Encoder
"""

import torch.nn as nn


class ResBlock(nn.Module):
    # Residual Block with pre-activation design

    def __init__(
        self, 
        in_channels, 
        out_channels, 
        kernel_size, 
        stride=1, 
        dropout=0.2,
    ):
        super().__init__()
        
        if kernel_size % 2 == 0:
            raise ValueError("kernel size must be odd.")
        
        padding = (kernel_size - 1) // 2
        
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv1d(
            in_channels, 
            out_channels, 
            kernel_size, 
            stride=stride, 
            padding=padding,
        )
        self.dropout1 = nn.Dropout(dropout)
        
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv1d(
            out_channels, 
            out_channels, 
            kernel_size, 
            stride=1, 
            padding=padding,
        )
        self.dropout2 = nn.Dropout(dropout)
        
        if in_channels != out_channels or stride != 1:
            self.downsample = nn.Conv1d(
                in_channels, 
                out_channels, 
                kernel_size=1, 
                stride=stride,
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)
        out = self.dropout1(out)
        
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)
        out = self.dropout2(out)
        
        res = self.downsample(x)
        return out + res


class ResNet1DEncoder(nn.Module):
    def __init__(
        self,
        n_inputs,
        base_channels=64,
        kernel_size=7,
        dropout=0.2,
        n_layers=3,
        stride=2,
        pool_output=True,
    ):
        super().__init__()
        
        self.pool_output = pool_output
        
        self.input_proj = nn.Conv1d(
            n_inputs, 
            base_channels, 
            kernel_size=7, 
            padding=3
        )
        self.input_bn = nn.BatchNorm1d(base_channels)
        self.input_relu = nn.ReLU()
        
        layers = []
        curr_in_channels = base_channels
        
        for i in range(n_layers):
            curr_out_channels = base_channels * 2**i
            
            layers.append(
                ResBlock(
                    in_channels=curr_in_channels,
                    out_channels=curr_out_channels,
                    kernel_size=kernel_size,
                    stride=stride if i > 0 else 1,
                    dropout=dropout,
                )
            )
            
            curr_in_channels = curr_out_channels
        
        self.network = nn.Sequential(*layers)
        self.n_outputs = curr_in_channels
        
        if pool_output:
            self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.input_bn(x)
        x = self.input_relu(x)
        
        x = self.network(x)
        
        if self.pool_output:
            x = self.pool(x)
            x = x.squeeze(-1)
        
        return x
