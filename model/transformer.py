import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    # Sinusoidal positional encoding

    def __init__(self, hidden_dim, max_length=1000, device=None):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        padded_dim = hidden_dim + hidden_dim % 2
        encoding = torch.zeros(1, padded_dim, max_length)
        
        position = torch.arange(
            0, 
            max_length, 
            dtype=torch.float,
        ).unsqueeze(0)
        
        div_term = torch.exp(
            torch.arange(0, padded_dim, 2).float() 
            * (-math.log(10000.0) / padded_dim)
        )
        
        encoding[0, 0::2, :] = torch.sin(
            position * div_term.unsqueeze(1)
        )
        encoding[0, 1::2, :] = torch.cos(
            position * div_term.unsqueeze(1)
        )
        
        encoding = encoding[:, :hidden_dim, :]
        self.register_buffer("encoding", encoding)

    def forward(self, x):
        return x + self.encoding[:, :, : x.size(2)]


class FeedForward(nn.Module):
    # Position-wise feed-forward network

    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, n_chl, n_heads=8, dropout=0.1):
        super().__init__()
        
        self.padded_C = math.ceil(n_chl / n_heads) * n_heads
        self.pad = self.padded_C - n_chl if self.padded_C > n_chl else 0
        self.n_chl = n_chl
        
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=self.padded_C,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        self.norm1 = nn.LayerNorm(self.padded_C)
        self.norm2 = nn.LayerNorm(self.padded_C)
        self.ff = FeedForward(self.padded_C, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.transpose(1, 2)
        
        if self.pad > 0:
            x = F.pad(x, (0, self.pad))
        
        x_norm = self.norm1(x)
        attn_out, _ = self.multihead_attn(
            x_norm, 
            x_norm, 
            x_norm
        )
        x = x + self.dropout(attn_out)
        
        x_norm = self.norm2(x)
        x = x + self.dropout(self.ff(x_norm))
        
        x = x[:, :, : self.n_chl]
        x = x.transpose(1, 2)
        
        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        n_chl,
        n_time=250,
        base_channels=None,
        n_layers=4,
        n_heads=8,
        dropout=0.1,
        pos_encoding=True,
        max_length=None,
        pool_output=True,
        device=None,
    ):
        super().__init__()
        
        self.n_chl = n_chl
        self.pool_output = pool_output
        self.pos_enc_flag = pos_encoding
        
        if base_channels is not None and base_channels != n_chl:
            self.input_proj = nn.Conv1d(
                n_chl, 
                base_channels, 
                kernel_size=1,
            )
            self.input_bn = nn.BatchNorm1d(base_channels)
            self.hidden_dim = base_channels
        else:
            self.input_proj = None
            self.hidden_dim = n_chl
        
        if self.pos_enc_flag:
            if max_length is None:
                max_length = n_time
            
            self.pos_encoding = PositionalEncoding(
                self.hidden_dim, 
                max_length, 
                device,
            )
        
        self.trf_blocks = nn.Sequential(
            *[
                TransformerBlock(
                    self.hidden_dim, 
                    n_heads, 
                    dropout,
                )
                for _ in range(n_layers)
            ]
        )
        
        self.n_outputs = self.hidden_dim
        
        if pool_output:
            self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        if self.input_proj is not None:
            x = self.input_proj(x)
            x = self.input_bn(x)
        
        if self.pos_enc_flag:
            x = self.pos_encoding(x)
        
        x = self.trf_blocks(x)
        
        if self.pool_output:
            x = self.pool(x)
            x = x.squeeze(-1)
        
        return x
