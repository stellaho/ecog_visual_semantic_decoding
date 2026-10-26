from collections import defaultdict

import numpy as np
import torch


class BalancedTaskSampler:
    def __init__(
        self,
        labels,
        n_way=3,
        k_shot=5,
        n_query=15,
        n_episodes=2000,
        seed=None,
    ):
        if isinstance(labels, torch.Tensor):
            labels = labels.numpy()

        self.labels = np.asarray(labels)
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_query = n_query
        self.n_episodes = n_episodes
        self.rng = np.random.RandomState(seed)

        self.class_indices = self._build_class_indices()
        self._validate_and_adjust()

    def _build_class_indices(self):
        class_indices = defaultdict(list)

        for i, label in enumerate(self.labels):
            class_indices[int(label)].append(i)

        return {
            cls: np.asarray(indices)
            for cls, indices in class_indices.items()
        }

    def _validate_and_adjust(self):
        class_sizes = {
            cls: len(indices)
            for cls, indices in self.class_indices.items()
        }

        if len(class_sizes) < self.n_way:
            raise ValueError(
                f"Dataset has fewer than {self.n_way} classes"
            )

        min_class_size = min(class_sizes.values())

        if min_class_size < 2:
            raise ValueError(
                "Each class needs separate support and query samples"
            )

        required = self.k_shot + self.n_query

        # Reduce the episode size only when a class is too small.
        if min_class_size < required:
            minimum = 3 if min_class_size >= 6 else 1
            ratio = self.k_shot / required

            self.k_shot = max(
                minimum,
                int(min_class_size * ratio),
            )
            self.n_query = min_class_size - self.k_shot

            if self.n_query < minimum:
                self.n_query = minimum
                self.k_shot = min_class_size - self.n_query

    def _sample_episode(self):
        classes = self.rng.choice(
            list(self.class_indices),
            size=self.n_way,
            replace=False,
        )

        support_indices = []
        query_indices = []

        for cls in classes:
            indices = self.class_indices[cls]
            samples = self.rng.choice(
                indices,
                size=self.k_shot + self.n_query,
                replace=False,
            )

            support_indices.extend(samples[: self.k_shot])
            query_indices.extend(samples[self.k_shot :])

        return (
            np.asarray(support_indices),
            np.asarray(query_indices),
            classes,
        )

    def __iter__(self):
        for _ in range(self.n_episodes):
            support_idx, query_idx, _ = self._sample_episode()
            yield support_idx, query_idx

    def __len__(self):
        return self.n_episodes


class EpisodeDataLoader:
    def __init__(self, dataset, sampler, device="cpu"):
        self.dataset = dataset
        self.sampler = sampler
        self.device = device

    def __iter__(self):
        for support_idx, query_idx in self.sampler:
            support_x = torch.stack(
                [self.dataset[i][0] for i in support_idx]
            )
            support_y = torch.tensor(
                [self.dataset[i][1] for i in support_idx]
            )

            query_x = torch.stack(
                [self.dataset[i][0] for i in query_idx]
            )
            query_y = torch.tensor(
                [self.dataset[i][1] for i in query_idx]
            )

            # Map the selected classes to 0, ..., N-1 within each episode.
            classes = torch.unique(support_y)
            class_map = {
                cls.item(): i
                for i, cls in enumerate(classes)
            }

            support_y = torch.tensor(
                [class_map[y.item()] for y in support_y]
            )
            query_y = torch.tensor(
                [class_map[y.item()] for y in query_y]
            )

            yield (
                support_x.to(self.device),
                support_y.to(self.device),
                query_x.to(self.device),
                query_y.to(self.device),
            )

    def __len__(self):
        return len(self.sampler)