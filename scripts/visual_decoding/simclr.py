import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

from model.components import NT_Xent
from model.scaler import ChannelScaler


class SimCLRTrainer:
    def __init__(
        self,
        encoder,
        head,
        augmenter,
        attention=None,
        device="cpu",
    ):
        self.encoder = encoder.to(device)
        self.head = head.to(device)
        self.augmenter = augmenter.to(device)
        self.attention = (
            attention.to(device)
            if attention is not None
            else None
        )

        self.device = device
        self.scaler = None

        self.parameters = (
            list(encoder.parameters())
            + list(head.parameters())
        )

        if attention is not None:
            self.parameters += list(
                attention.parameters()
            )

    def _pool(self, z):
        if self.attention is not None:
            return self.attention(z)

        if z.dim() == 3:
            return z.mean(dim=-1)

        return z

    def train_epoch(
        self,
        dataloader,
        optimizer,
        temperature=0.5,
    ):
        self.encoder.train()
        self.head.train()
        self.augmenter.train()

        if self.attention is not None:
            self.attention.train()

        for (x,) in dataloader:
            x = x.to(
                self.device,
                non_blocking=True,
            )

            x_norm = self.scaler.transform(x)

            optimizer.zero_grad()

            x1 = self.augmenter(x_norm)
            x2 = self.augmenter(x_norm)

            z1 = self._pool(
                self.encoder(x1)
            )

            z2 = self._pool(
                self.encoder(x2)
            )

            z1 = self.head(z1)
            z2 = self.head(z2)

            loss = NT_Xent(
                z1,
                z2,
                temperature=temperature,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.parameters,
                10.0,
            )

            optimizer.step()

    def validate_epoch(
        self,
        dataloader,
        temperature=0.5,
    ):
        self.encoder.eval()
        self.head.eval()
        self.augmenter.train()

        if self.attention is not None:
            self.attention.eval()

        total_loss = 0.0

        with torch.no_grad():
            for (x,) in dataloader:
                x = x.to(
                    self.device,
                    non_blocking=True,
                )

                x_norm = self.scaler.transform(x)

                z1 = self.head(
                    self._pool(
                        self.encoder(
                            self.augmenter(x_norm)
                        )
                    )
                )

                z2 = self.head(
                    self._pool(
                        self.encoder(
                            self.augmenter(x_norm)
                        )
                    )
                )

                loss = NT_Xent(
                    z1,
                    z2,
                    temperature=temperature,
                )

                total_loss += loss.item()

        return total_loss / max(
            len(dataloader),
            1,
        )

    def tune(
        self,
        X,
        train_indices,
        val_indices,
        batch_size=32,
        n_epochs=100,
        lr=0.001,
        patience=10,
        weight_decay=0.0001,
        temperature=0.5,
        scaler_mode="standard",
        scaler_fit_indices=None,
    ):
        if set(train_indices) & set(val_indices):
            raise ValueError(
                "Pretraining training and validation indices overlap"
            )

        if not train_indices or not val_indices:
            raise ValueError(
                "Pretraining training and validation sets must be non-empty"
            )

        if len(train_indices) != len(set(train_indices)):
            raise ValueError(
                "Pretraining training indices must be unique"
            )

        if scaler_fit_indices is None:
            scaler_fit_indices = train_indices

        self.scaler = ChannelScaler(
            mode=scaler_mode
        ).fit(
            X[scaler_fit_indices]
        )

        train_dataset = TensorDataset(
            X[train_indices]
        )

        val_dataset = TensorDataset(
            X[val_indices]
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )

        optimizer = torch.optim.AdamW(
            self.parameters,
            lr=lr,
            weight_decay=weight_decay,
        )

        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=patience // 2,
        )

        best_val_loss = float("inf")
        best_epoch = None
        patience_counter = 0

        for epoch in range(n_epochs):
            self.train_epoch(
                train_loader,
                optimizer,
                temperature=temperature,
            )

            val_loss = self.validate_epoch(
                val_loader,
                temperature=temperature,
            )

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                patience_counter = 0
            else:
                patience_counter += 1

                if patience_counter >= patience:
                    break

        return best_val_loss, best_epoch

    def fit(
        self,
        X,
        batch_size=32,
        n_epochs=100,
        lr=0.001,
        weight_decay=0.0001,
        temperature=0.5,
        scaler_mode="standard",
        scaler_fit_indices=None,
    ):
        if len(X) == 0:
            raise ValueError(
                "Pretraining training set must be non-empty"
            )

        scaler_data = (
            X
            if scaler_fit_indices is None
            else X[scaler_fit_indices]
        )

        self.scaler = ChannelScaler(
            mode=scaler_mode
        ).fit(
            scaler_data
        )

        train_dataset = TensorDataset(X)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )

        optimizer = torch.optim.AdamW(
            self.parameters,
            lr=lr,
            weight_decay=weight_decay,
        )

        for _ in range(n_epochs):
            self.train_epoch(
                train_loader,
                optimizer,
                temperature=temperature,
            )