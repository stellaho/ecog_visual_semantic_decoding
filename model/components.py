import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelDropout(nn.Module):
    def __init__(self, p=0.2):
        super().__init__()
        
        if not 0 <= p < 1:
            raise ValueError(f"Dropout probability must be in [0, 1), got {p}")
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x

        mask = x.new_empty(x.size(0), x.size(1), 1).bernoulli_(1 - self.p)
        return x * mask


class TemporalAttention(nn.Module):
    # Learnable weighted sum over time dimension.

    def __init__(self, n_channels, hidden_dim=None):
        super().__init__()
        
        hidden_dim = hidden_dim or max(n_channels // 4, 32)
        
        self.layer_norm = nn.LayerNorm(n_channels)
        self.query = nn.Linear(n_channels, hidden_dim)
        self.key = nn.Linear(n_channels, hidden_dim)
        self.scale = hidden_dim ** (-0.5)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.layer_norm(x)
        
        q = self.query(x.mean(dim=1, keepdim=True))
        k = self.key(x)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, x)
        out = out.squeeze(1)
        return out


class ContrastiveHead(nn.Module):
    def __init__(self, n_inputs, proj_dim=128, hidden_dim=None):
        super().__init__()
        
        hidden_dim = hidden_dim or n_inputs
        
        self.mlp = nn.Sequential(
            nn.Linear(n_inputs, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, proj_dim),
        )

    def forward(self, x):
        return self.mlp(x)


class ECoGAugmenter(nn.Module):
    def __init__(self, channel_dropout_p=0.2, time_shift_max=10, noise_std=0.01):
        super().__init__()
        
        self.channel_dropout = ChannelDropout(channel_dropout_p)
        self.time_shift_max = time_shift_max
        self.noise_std = noise_std

    def forward(self, x):
        if not self.training:
            return x

        x = self.channel_dropout(x)

        if self.time_shift_max > 0:
            x = self._time_shift(x)

        if self.noise_std > 0:
            x = self._add_noise(x)

        return x

    def _time_shift(self, x):
        # Randomly shift each signal in time
        B = x.size(0) # [B, C, T]
        shifts = torch.randint(-self.time_shift_max, self.time_shift_max + 1, (B,))
        x_shifted = torch.zeros_like(x)
        for i in range(B):
            x_shifted[i] = torch.roll(x[i], shifts=shifts[i].item(), dims=-1)
        return x_shifted

    def _add_noise(self, x):
        noise = torch.randn_like(x) * self.noise_std
        return x + noise


def NT_Xent(z1, z2, temperature=0.5):
    B = z1.size(0)
    device = z1.device
    
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    z = torch.cat([z1, z2], dim=0)
    
    sim = torch.mm(z, z.t()) / temperature
    
    mask = torch.eye(2 * B, device=device).bool()
    sim.masked_fill_(mask, float("-inf"))
    
    pos_indices = torch.cat(
        [
            torch.arange(B, 2 * B, device=device),
            torch.arange(0, B, device=device),
        ]
    )
    labels = pos_indices
    loss = F.cross_entropy(sim, labels)
    return loss
