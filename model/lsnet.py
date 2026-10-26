import torch
import torch.nn as nn


class LSBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        large_kernel=51,
        small_kernel=5,
        stride=1,
        dropout=0.2,
    ):
        super().__init__()
        
        if large_kernel % 2 == 0 or small_kernel % 2 == 0:
            raise ValueError("large_kernel and small_kernel must be odd.")
        
        large_channels = out_channels // 2
        small_channels = out_channels - large_channels
        
        large_padding = (large_kernel - 1) // 2
        self.large_branch = nn.Sequential(
            nn.Conv1d(
                in_channels,
                in_channels,
                kernel_size=large_kernel,
                stride=stride,
                padding=large_padding,
                groups=in_channels,
            ),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(),
            nn.Conv1d(
                in_channels, 
                large_channels, 
                kernel_size=1,
            ),
            nn.BatchNorm1d(large_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        small_padding = (small_kernel - 1) // 2
        self.small_branch = nn.Sequential(
            nn.Conv1d(
                in_channels,
                small_channels,
                kernel_size=small_kernel,
                stride=stride,
                padding=small_padding,
            ),
            nn.BatchNorm1d(small_channels),
            nn.ReLU(),
            nn.Conv1d(
                small_channels,
                small_channels,
                kernel_size=small_kernel,
                stride=1,
                padding=small_padding,
            ),
            nn.BatchNorm1d(small_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        if in_channels != out_channels or stride != 1:
            self.residual = nn.Conv1d(
                in_channels, 
                out_channels, 
                kernel_size=1, 
                stride=stride,
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        large_out = self.large_branch(x)
        small_out = self.small_branch(x)
        
        out = torch.cat([large_out, small_out], dim=1)
        res = self.residual(x)
        
        return out + res


class LSNetEncoder(nn.Module):
    def __init__(
        self,
        n_inputs,
        base_channels=64,
        large_kernel=51,
        small_kernel=5,
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
            padding=3,
        )
        
        self.input_bn = nn.BatchNorm1d(base_channels)
        self.input_relu = nn.ReLU()
        
        layers = []
        curr_in_channels = base_channels
        
        for i in range(n_layers):
            curr_out_channels = base_channels * 2**i
            
            layer_large_kernel = max(
                large_kernel // 2**i, 
                small_kernel + 2,
            )
            
            if layer_large_kernel % 2 == 0:
                layer_large_kernel += 1
            
            layers.append(
                LSBlock(
                    in_channels=curr_in_channels,
                    out_channels=curr_out_channels,
                    large_kernel=layer_large_kernel,
                    small_kernel=small_kernel,
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
