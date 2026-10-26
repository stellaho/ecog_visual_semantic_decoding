import torch.nn as nn


class Chomp1d(nn.Module):
    # Trim causal padding

    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size > 0:
            return x[:, :, : -self.chomp_size].contiguous()
        return x


class TCNBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        dropout=0.2,
        causal=True,
    ):
        super().__init__()
        
        self.causal = causal
        
        if causal:
            padding = (kernel_size - 1) * dilation
            self.chomp = Chomp1d(padding)
        else:
            padding = (kernel_size - 1) * dilation // 2
            self.chomp = nn.Identity()
        
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=padding if causal else (kernel_size - 1) * dilation // 2,
            dilation=dilation,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.chomp2 = Chomp1d(padding) if causal else nn.Identity()
        
        if in_channels != out_channels or stride != 1:
            self.downsample = nn.Conv1d(
                in_channels, 
                out_channels, 
                kernel_size=1, 
                stride=stride
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        out = self.conv1(x)
        out = self.chomp(out)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        
        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        
        res = self.downsample(x)
        
        if out.size(2) != res.size(2):
            min_len = min(out.size(2), res.size(2))
            out = out[:, :, :min_len]
            res = res[:, :, :min_len]
        
        return out + res


class TCNEncoder(nn.Module):
    def __init__(
        self,
        n_inputs,
        base_channels=64,
        kernel_size=7,
        dropout=0.2,
        n_layers=3,
        dilation_base=2,
        causal=False,
        stride=1,
        pool_output=True,
    ):
        super().__init__()
        
        self.causal = causal
        self.pool_output = pool_output
        
        self.input_proj = nn.Conv1d(
            n_inputs, 
            base_channels, 
            kernel_size=1,
        )
        self.input_bn = nn.BatchNorm1d(base_channels)
        
        layers = []
        curr_in_channels = base_channels
        
        for i in range(n_layers):
            curr_out_channels = base_channels * 2**i
            layer_dilation = dilation_base**i
            
            layers.append(
                TCNBlock(
                    in_channels=curr_in_channels,
                    out_channels=curr_out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=layer_dilation,
                    dropout=dropout,
                    causal=causal,
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
        
        x = self.network(x)
        
        if self.pool_output:
            x = self.pool(x)
            x = x.squeeze(-1)
        
        return x



