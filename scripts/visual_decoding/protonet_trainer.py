import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.utils.data import ConcatDataset

from meta_learning.protonet import build_protonet
from meta_learning.sampler import BalancedTaskSampler, EpisodeDataLoader
from model.scaler import ChannelScaler


class ProtoNetTrainer:
    def __init__(
        self,
        model,
        device,
        scaler_mode="standard",
        enable_wandb=True,
    ):
        self.model = model.to(device)
        self.device = device
        self.scaler_mode = scaler_mode
        self.enable_wandb = enable_wandb
        self.scaler = None

    def fit_scaler(self, train_dataset):
        X = torch.stack(
            [train_dataset[i][0] for i in range(len(train_dataset))]
        )

        self.scaler = ChannelScaler(mode=self.scaler_mode)
        self.scaler.fit(X)

        return self.scaler

    def _apply_scaler(self, x):
        if self.scaler is not None:
            return self.scaler.transform(x)

        return x

    def train_episode(
        self,
        support_x,
        support_y,
        query_x,
        query_y,
        optimizer,
        criterion,
    ):
        self.model.train()

        support_x = self._apply_scaler(support_x)
        query_x = self._apply_scaler(query_x)

        optimizer.zero_grad()

        logits = self.model(
            support_x,
            support_y,
            query_x,
        )
        loss = criterion(logits, query_y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            10.0,
        )
        optimizer.step()

        accuracy = (
            (logits.argmax(dim=1) == query_y)
            .float()
            .mean()
            .item()
        )

        return loss.item(), accuracy

    def eval_episode(
        self,
        support_x,
        support_y,
        query_x,
        query_y,
        criterion,
    ):
        self.model.eval()

        with torch.no_grad():
            support_x = self._apply_scaler(support_x)
            query_x = self._apply_scaler(query_x)

            logits = self.model(
                support_x,
                support_y,
                query_x,
            )
            loss = criterion(logits, query_y)

            accuracy = (
                (logits.argmax(dim=1) == query_y)
                .float()
                .mean()
                .item()
            )

        return loss.item(), accuracy

    def train_fold(
        self,
        train_dataset,
        val_dataset,
        n_episodes=2000,
        k_shot=5,
        n_query=15,
        lr=0.0001,
        weight_decay=0.0,
        label_smoothing=0.0,
        patience=200,
        log_interval=50,
    ):
        self.fit_scaler(train_dataset)

        train_labels = torch.tensor(
            [train_dataset[i][1] for i in range(len(train_dataset))]
        )
        val_labels = torch.tensor(
            [val_dataset[i][1] for i in range(len(val_dataset))]
        )

        train_sampler = BalancedTaskSampler(
            train_labels,
            n_way=3,
            k_shot=k_shot,
            n_query=n_query,
            n_episodes=n_episodes,
            seed=42,
        )
        val_sampler = BalancedTaskSampler(
            val_labels,
            n_way=3,
            k_shot=k_shot,
            n_query=n_query,
            n_episodes=max(1, min(100, n_episodes // 10)),
            seed=123,
        )

        self.episode_sizes = {
            "train": {
                "k_shot": train_sampler.k_shot,
                "n_query": train_sampler.n_query,
            },
            "validation": {
                "k_shot": val_sampler.k_shot,
                "n_query": val_sampler.n_query,
            },
        }

        train_loader = EpisodeDataLoader(
            train_dataset,
            train_sampler,
            self.device,
        )
        val_loader = EpisodeDataLoader(
            val_dataset,
            val_sampler,
            self.device,
        )

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        criterion = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing
        )

        best_loss = float("inf")
        accuracy_at_best_loss = 0.0
        best_state = None
        patience_count = 0

        train_losses = []
        train_accs = []

        for episode, data in enumerate(train_loader, start=1):
            support_x, support_y, query_x, query_y = data

            train_loss, train_acc = self.train_episode(
                support_x,
                support_y,
                query_x,
                query_y,
                optimizer,
                criterion,
            )

            train_losses.append(train_loss)
            train_accs.append(train_acc)

            if episode % log_interval != 0 and episode != n_episodes:
                continue

            val_losses = []
            val_accs = []

            for val_data in val_loader:
                support_x, support_y, query_x, query_y = val_data

                val_loss, val_acc = self.eval_episode(
                    support_x,
                    support_y,
                    query_x,
                    query_y,
                    criterion,
                )

                val_losses.append(val_loss)
                val_accs.append(val_acc)

            mean_train_loss = np.mean(train_losses)
            mean_train_acc = np.mean(train_accs)
            mean_val_loss = np.mean(val_losses)
            mean_val_acc = np.mean(val_accs)

            train_losses = []
            train_accs = []

            if self.enable_wandb:
                wandb.log(
                    {
                        "episode": episode,
                        "train_loss": mean_train_loss,
                        "train_acc": mean_train_acc,
                        "val_loss": mean_val_loss,
                        "val_acc": mean_val_acc,
                    }
                )

            if mean_val_loss < best_loss:
                best_loss = mean_val_loss
                accuracy_at_best_loss = mean_val_acc
                best_state = {
                    key: value.cpu().clone()
                    for key, value in self.model.state_dict().items()
                }
                patience_count = 0
            else:
                patience_count += log_interval

                if patience_count >= patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        return best_loss, accuracy_at_best_loss

    def compute_final_prototypes(self, train_dataset):
        self.model.eval()
        class_embeddings = {}

        with torch.no_grad():
            for i in range(len(train_dataset)):
                x, y = train_dataset[i]

                x = x.unsqueeze(0).to(self.device)
                x = self._apply_scaler(x)
                embedding = self.model.embed(x)

                label = y.item() if isinstance(y, torch.Tensor) else y
                class_embeddings.setdefault(label, []).append(embedding)

        class_labels = sorted(class_embeddings)

        prototypes = torch.stack(
            [
                torch.cat(class_embeddings[label], dim=0).mean(dim=0)
                for label in class_labels
            ]
        )

        return prototypes, class_labels

    def evaluate_test(
        self,
        test_dataset,
        prototypes,
        class_labels=None,
        batch_size=32,
    ):
        self.model.eval()

        predictions = []
        labels = []

        if class_labels is None:
            class_labels = list(range(len(prototypes)))

        class_labels = torch.tensor(class_labels)

        with torch.no_grad():
            for start in range(0, len(test_dataset), batch_size):
                indices = range(
                    start,
                    min(start + batch_size, len(test_dataset)),
                )

                x = torch.stack(
                    [test_dataset[i][0] for i in indices]
                ).to(self.device)

                y = torch.tensor(
                    [test_dataset[i][1] for i in indices]
                )

                x = self._apply_scaler(x)

                logits = self.model.classify_with_prototypes(
                    prototypes,
                    x,
                )

                pred_idx = logits.argmax(dim=1).cpu()
                predictions.extend(
                    class_labels[pred_idx].tolist()
                )
                labels.extend(y.tolist())

        predictions = np.asarray(predictions)
        labels = np.asarray(labels)
        accuracy = (predictions == labels).mean()

        return accuracy, predictions, labels


def evaluate_loro_all_recordings(
    patient_dataset,
    config,
    encoder_type,
    device,
    seeds=(42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144),
    mask_indices=None,
    visual_response_ms=500,
    save_to_csv=True,
    use_attention=False,
):
    from scripts.utils.results import compute_metrics, save_results_to_csv

    recordings = {
        name: dataset
        for name, dataset in patient_dataset.recordings.items()
        if Path(name).stem != "Val01"
    }

    test_recordings = list(recordings)
    tuning = config.get("protonet_tuning")

    if tuning:
        expected = {
            "patient_id": patient_dataset.patient_id,
            "encoder_type": encoder_type,
            "use_attention": use_attention,
            "visual_response_ms": visual_response_ms,
            "mask_indices": mask_indices,
            "bandpass_filter": patient_dataset.bandpass_filter,
        }

        for key, value in expected.items():
            if tuning[key] != json.loads(json.dumps(value)):
                raise ValueError(
                    f"ProtoNet tuning and evaluation disagree on {key}"
                )

        test_recordings = [
            name
            for name in recordings
            if Path(name).stem == tuning["outer_test_recording"]
        ]

        training_ids = {
            Path(name).stem
            for name in recordings
            if name not in test_recordings
        }

        if training_ids != set(tuning["training_recordings"]):
            raise ValueError(
                "ProtoNet tuning and evaluation use different training recordings"
            )

    all_results = []

    for test_recording in test_recordings:
        test_set = recordings[test_recording]

        train_set = ConcatDataset(
            [
                dataset
                for name, dataset in recordings.items()
                if name != test_recording
            ]
        )

        sample_x, _ = test_set[0]
        seed_results = []

        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)

            model = build_protonet(
                encoder_type=encoder_type,
                n_inputs=sample_x.shape[0],
                n_time=sample_x.shape[-1],
                use_attention=use_attention,
                attn_hidden_dim=config["attn_hidden_dim"],
                temperature=config["cl_temperature"],
                **{
                    key: value
                    for key, value in config.items()
                    if key != "attn_hidden_dim"
                },
            ).to(device)

            trainer = ProtoNetTrainer(
                model=model,
                device=device,
                scaler_mode=config["scaler_mode"],
                enable_wandb=False,
            )
            trainer.fit_scaler(train_set)

            train_labels = torch.tensor(
                [train_set[i][1] for i in range(len(train_set))]
            )

            k_shot = config.get("k_shot", 5)
            n_query = config.get("n_query", 5)
            n_episodes = config.get("n_episodes", 2000)
            patience = config.get("patience", 300)

            sampler = BalancedTaskSampler(
                train_labels,
                n_way=3,
                k_shot=k_shot,
                n_query=n_query,
                n_episodes=n_episodes,
                seed=seed,
            )
            loader = EpisodeDataLoader(
                train_set,
                sampler,
                device,
            )

            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config["learning_rate"],
                weight_decay=config.get(
                    "weight_decay",
                    0.0001,
                ),
            )
            criterion = nn.CrossEntropyLoss(
                label_smoothing=config.get(
                    "label_smoothing",
                    0.0,
                )
            )

            best_loss = float("inf")
            patience_count = 0

            for support_x, support_y, query_x, query_y in loader:
                loss, _ = trainer.train_episode(
                    support_x,
                    support_y,
                    query_x,
                    query_y,
                    optimizer,
                    criterion,
                )

                if loss < best_loss:
                    best_loss = loss
                    patience_count = 0
                else:
                    patience_count += 1

                    if patience_count >= patience:
                        break

            prototypes, class_labels = trainer.compute_final_prototypes(
                train_set
            )

            _, predictions, labels = trainer.evaluate_test(
                test_set,
                prototypes,
                class_labels,
            )

            metrics = compute_metrics(
                predictions,
                labels,
                None,
            )
            metrics["seed"] = seed
            metrics["episode_sizes"] = {
                "k_shot": sampler.k_shot,
                "n_query": sampler.n_query,
            }

            seed_results.append(metrics)

            if save_to_csv:
                save_results_to_csv(
                    subject_id=patient_dataset.patient_id,
                    encoder=encoder_type,
                    random_seed=seed,
                    loro_test_id=Path(test_recording).stem,
                    visual_response_ms=visual_response_ms,
                    results=metrics,
                    mode="protonet",
                    use_attention=use_attention,
                    bandpass_filter=patient_dataset.bandpass_filter,
                )

        all_results.append(
            {
                "test_recording": test_recording,
                "mean_balanced_accuracy": np.mean(
                    [
                        result["balanced_accuracy"]
                        for result in seed_results
                    ]
                ),
                "std_balanced_accuracy": np.std(
                    [
                        result["balanced_accuracy"]
                        for result in seed_results
                    ]
                ),
                "seed_results": seed_results,
            }
        )

    return all_results