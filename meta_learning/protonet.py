import torch
import torch.nn as nn

from model.components import TemporalAttention
from model.encoder import build_encoder


class PrototypicalNetwork(nn.Module):
    def __init__(
        self,
        encoder,
        n_channels,
        use_attention=False,
        attn_hidden_dim=64,
        temperature=0.1,
    ):
        super().__init__()

        self.encoder = encoder
        self.use_attention = use_attention
        self.temperature = nn.Parameter(
            torch.tensor(temperature),
            requires_grad=False,
        )

        self.attention = None
        if use_attention:
            self.attention = TemporalAttention(
                n_channels=n_channels,
                hidden_dim=attn_hidden_dim,
            )

    def embed(self, x):
        features = self.encoder(x)

        if self.attention is not None:
            return self.attention(features)

        return features.mean(dim=2)

    def compute_prototypes(self, embeddings, labels):
        classes = torch.unique(labels)

        return torch.stack(
            [
                embeddings[labels == cls].mean(dim=0)
                for cls in classes
            ]
        )

    def squared_euclidean_distance(self, queries, prototypes):
        return torch.sum(
            (queries.unsqueeze(1) - prototypes.unsqueeze(0)) ** 2,
            dim=2,
        )

    def forward(self, support_x, support_y, query_x):
        support_embeddings = self.embed(support_x)
        query_embeddings = self.embed(query_x)

        prototypes = self.compute_prototypes(
            support_embeddings,
            support_y,
        )
        distances = self.squared_euclidean_distance(
            query_embeddings,
            prototypes,
        )

        return -distances / self.temperature

    def classify_with_prototypes(self, prototypes, query_x):
        query_embeddings = self.embed(query_x)
        distances = self.squared_euclidean_distance(
            query_embeddings,
            prototypes,
        )

        return -distances / self.temperature


def build_protonet(
    encoder_type,
    n_inputs,
    use_attention=False,
    attn_hidden_dim=64,
    temperature=0.1,
    n_time=None,
    **encoder_kwargs,
):
    encoder = build_encoder(
        encoder_type,
        encoder_kwargs,
        n_inputs,
        n_time,
        pool_output=False,
    )

    return PrototypicalNetwork(
        encoder,
        encoder.n_outputs,
        use_attention,
        attn_hidden_dim,
        temperature,
    )