import torch
from torch import nn

from model.components import ChannelDropout, TemporalAttention


def mixup_data(x, y, lam, index):
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index]


class EncoderClassifier(nn.Module):
    def __init__(
        self,
        encoder,
        num_classes=3,
        dropout=0.3,
        channel_dropout=0.2,
        freeze_encoder=False,
        adapter_dim=None,
        use_attention=False,
        attn_hidden_dim=None,
    ):
        super().__init__()

        self.encoder = encoder
        self.num_classes = num_classes
        self.freeze_encoder = freeze_encoder
        self.use_attention = use_attention
        self.scaler = None

        self.channel_dropout = (
            ChannelDropout(p=channel_dropout) if channel_dropout > 0 else None
        )

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.attention = None
        if use_attention:
            self.attention = TemporalAttention(
                n_channels=encoder.n_outputs,
                hidden_dim=attn_hidden_dim or encoder.n_outputs // 4,
            )

        if adapter_dim is not None:
            self.adapter = nn.Sequential(
                nn.Linear(encoder.n_outputs, adapter_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(adapter_dim, adapter_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            classifier_input_dim = adapter_dim
        else:
            self.adapter = None
            classifier_input_dim = encoder.n_outputs

        self.classifier = nn.Linear(classifier_input_dim, num_classes)

    def train(self, mode=True):
        super().train(mode)

        if self.freeze_encoder:
            self.encoder.eval()

        return self

    def forward(self, x, y=None, mixup_layer=None, lam=None, index=None):
        y_a = y_b = None

        if self.scaler is not None:
            x = self.scaler.transform(x)

        if self.channel_dropout is not None:
            x = self.channel_dropout(x)

        if mixup_layer == 0 and lam is not None and index is not None:
            x, y_a, y_b = mixup_data(x, y, lam, index)

        if self.freeze_encoder:
            with torch.no_grad():
                features = self.encoder(x)
        else:
            features = self.encoder(x)

        if self.attention is not None:
            features = self.attention(features)

        if mixup_layer == 1 and lam is not None and index is not None:
            features, y_a, y_b = mixup_data(features, y, lam, index)

        if self.adapter is not None:
            features = self.adapter(features)

        logits = self.classifier(features)

        if mixup_layer is not None and y_a is not None:
            return logits, y_a, y_b

        return logits
