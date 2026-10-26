"""
Per-Channel Signal Scaler
"""

import torch


class ChannelScaler:
    def __init__(self, mode="standard"):
        if mode not in ("standard", "robust"):
            raise ValueError(f"mode must be 'standard' or 'robust', got {mode}")

        self.mode = mode
        self.center = None
        self.scale = None
        self.fitted = False

    def fit(self, X):
        _, C, _ = X.shape  # X shape: (B,C,T)
        X_flat = X.permute(1, 0, 2).reshape(C, -1)

        if self.mode == "standard":
            self.center = X_flat.mean(dim=1)
            self.scale = X_flat.std(dim=1) + 1e-06
        else:
            self.center = X_flat.median(dim=1).values
            q75 = torch.quantile(X_flat, 0.75, dim=1)
            q25 = torch.quantile(X_flat, 0.25, dim=1)
            self.scale = q75 - q25 + 1e-06

        self.fitted = True
        return self

    def transform(self, X):
        if not self.fitted:
            raise RuntimeError("Scaler not fitted. Call fit() first.")

        center = self.center.to(X.device).view(1, -1, 1)
        scale = self.scale.to(X.device).view(1, -1, 1)

        return (X - center) / scale
