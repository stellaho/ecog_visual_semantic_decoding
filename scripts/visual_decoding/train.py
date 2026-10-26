import copy

import numpy as np
import torch
import wandb
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader


def mixup_criterion(criterion, logits, y_a, y_b, lam):
    return lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)


def select_epochs(fold_epochs):
    return max(1, int(np.rint(np.median(fold_epochs))))


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        device,
        use_mixup=False,
        mixup_alpha=2.0,
        mixup_layers=(0, 1),
        label_smoothing=0.0,
        representation_fit=None,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.use_mixup = use_mixup
        self.mixup_alpha = mixup_alpha
        self.mixup_layers = tuple(mixup_layers)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.representation_fit = representation_fit

        # Save the initial state so every inner fold starts from the same point.
        self.initial_model = copy.deepcopy(self.model.state_dict())
        self.initial_optimizer = copy.deepcopy(self.optimizer.state_dict())
        self.initial_scaler = copy.deepcopy(self.model.scaler)

    def _reset(self, train_set):
        self.model.load_state_dict(self.initial_model)
        self.optimizer.load_state_dict(copy.deepcopy(self.initial_optimizer))
        self.model.scaler = copy.deepcopy(self.initial_scaler)

        if self.representation_fit is not None:
            self.representation_fit(self.model, train_set)
        else:
            # Fit channel normalization using training data only.
            X = torch.stack([train_set[i][0] for i in range(len(train_set))])
            self.model.scaler.fit(X)

    @staticmethod
    def _loader(dataset, batch_size, shuffle):
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
        )

    def train_epoch(self, loader):
        self.model.train()

        # SimCLR encoder remains frozen during classifier training.
        if self.representation_fit is not None:
            self.model.encoder.eval()

        total_loss = 0.0

        for batch_id, (x, y) in enumerate(loader):
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            if self.use_mixup:
                lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
                index = torch.randperm(x.size(0), device=self.device)
                layer = int(np.random.choice(self.mixup_layers))

                logits, y_a, y_b = self.model(
                    x,
                    y=y,
                    mixup_layer=layer,
                    lam=lam,
                    index=index,
                )
                loss = mixup_criterion(
                    self.criterion,
                    logits,
                    y_a,
                    y_b,
                    lam,
                )
            else:
                loss = self.criterion(self.model(x), y)

            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
            self.optimizer.step()

            total_loss += loss.item()

            if wandb.run is not None:
                wandb.log(
                    {
                        "batch": batch_id,
                        "batch_train_loss": loss.item(),
                    }
                )

        return total_loss / len(loader)

    def validate_epoch(self, loader):
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                loss = self.criterion(self.model(x), y)
                total_loss += loss.item()

        return total_loss / len(loader)

    def cross_validate(self, folds, batch_size, n_epochs, patience):
        fold_losses = []
        fold_epochs = []

        for fold_id, (train_set, val_set) in enumerate(folds, start=1):
            self._reset(train_set)

            train_loader = self._loader(train_set, batch_size, shuffle=True)
            val_loader = self._loader(val_set, batch_size, shuffle=False)

            scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.5,
                patience=patience // 2,
            )

            best_loss = float("inf")
            best_epoch = None
            patience_count = 0

            for epoch in range(1, n_epochs + 1):
                train_loss = self.train_epoch(train_loader)
                val_loss = self.validate_epoch(val_loader)
                scheduler.step(val_loss)

                if wandb.run is not None:
                    wandb.log(
                        {
                            "fold": fold_id,
                            "epoch": epoch,
                            "train_loss": train_loss,
                            "val_loss": val_loss,
                            "learning_rate": self.optimizer.param_groups[0]["lr"],
                        }
                    )

                if val_loss < best_loss:
                    best_loss = val_loss
                    best_epoch = epoch
                    patience_count = 0
                else:
                    patience_count += 1

                    # Stop when validation loss has not improved for `patience` epochs.
                    if patience_count >= patience:
                        break

            fold_losses.append(float(best_loss))
            fold_epochs.append(best_epoch)

        results = {
            "cv_val_loss": float(np.mean(fold_losses)),
            "fold_best_val_losses": fold_losses,
            "fold_best_epochs": fold_epochs,
            "final_fit_epochs": select_epochs(fold_epochs),
        }

        if wandb.run is not None:
            wandb.log(
                {
                    "cv_val_loss": results["cv_val_loss"],
                    "final_fit_epochs": results["final_fit_epochs"],
                }
            )
            wandb.run.summary["fold_best_val_losses"] = fold_losses
            wandb.run.summary["fold_best_epochs"] = fold_epochs

        return results

    def fit(self, train_set, batch_size, n_epochs):
        # Start fresh before training on all outer-training recordings.
        self._reset(train_set)

        loader = self._loader(train_set, batch_size, shuffle=True)

        for epoch in range(1, n_epochs + 1):
            loss = self.train_epoch(loader)

            if wandb.run is not None:
                wandb.log(
                    {
                        "final_fit_epoch": epoch,
                        "final_fit_train_loss": loss,
                    }
                )

        return self.model, self.criterion