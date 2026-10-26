"""
Build, train, and test a visual-decoding classifier
"""

import copy

import torch
import wandb
from torch.utils.data import ConcatDataset

from model.classifier import EncoderClassifier
from model.encoder import ENCODERS, build_encoder
from model.scaler import ChannelScaler
from scripts.utils.splits import get_loro_splits, split_recordings
from scripts.visual_decoding.evaluate import Evaluation
from scripts.visual_decoding.train import Trainer, select_epochs


def validate_config(config, encoder_type):
    if encoder_type not in ENCODERS:
        raise ValueError(f"Unknown encoder: {encoder_type}")

    required = [
        "lr",
        "weight_decay",
        "batch_size",
        "n_epochs",
        "patience",
        "dropout",
        "channel_dropout",
        "use_mixup",
        "label_smoothing",
        "adapter_dim",
    ]

    required_by_encoder = {
        "tcn": ["base_channels", "n_layers", "kernel_size"],
        "resnet": ["base_channels", "n_layers", "kernel_size"],
        "lsnet": ["base_channels", "n_layers"],
        "lstm": ["base_channels", "n_layers"],
        "transformer": [
            "base_channels",
            "n_layers",
            "n_heads",
            "pos_encoding",
        ],
        "eegnet": [
            "eegnet_f1",
            "eegnet_depth_multiplier",
            "eegnet_f2",
            "eegnet_temporal_kernel",
            "eegnet_separable_kernel",
            "eegnet_pool1",
            "eegnet_pool2",
            "eegnet_spatial_max_norm",
        ],
    }

    required += required_by_encoder[encoder_type]

    if config.get("use_mixup"):
        required += ["mixup_alpha", "mixup_layers"]

    missing = [key for key in required if key not in config]

    if encoder_type == "tcn":
        if "dilation" not in config and "dilation_base" not in config:
            missing.append("dilation (or dilation_base)")

    if missing:
        raise KeyError(f"Missing config keys for {encoder_type}: {missing}")

    if encoder_type == "eegnet" and config["eegnet_f2"] is None:
        raise ValueError("eegnet_f2 must be explicitly specified")

    return copy.deepcopy(config)


class ClassificationPipeline:
    def __init__(
        self,
        patient_dataset,
        loro_test_id,
        config,
        device,
        enable_wandb=False,
        encoder_type="tcn",
        freeze_encoder=False,
        use_attention=False,
        attn_hidden_dim=None,
        scaler_mode="standard",
        wandb_run_name=None,
    ):
        self.patient_dataset = patient_dataset
        self.loro_test_id = loro_test_id
        self.config = copy.deepcopy(config)
        self.device = device
        self.enable_wandb = enable_wandb
        self.encoder_type = encoder_type
        self.freeze_encoder = freeze_encoder
        self.use_attention = use_attention
        self.attn_hidden_dim = attn_hidden_dim
        self.scaler_mode = scaler_mode
        self.wandb_run_name = wandb_run_name

        self.representation_fit = None

        if "contrastive_pretraining" in self.config:
            from scripts.visual_decoding.simclr_protocol import (
                SimCLRFit,
            )

            simclr_config = self.config["contrastive_pretraining"]

            self.representation_fit = SimCLRFit(
                simclr_config,
                patient_dataset,
                loro_test_id,
                encoder_type,
                use_attention,
                device,
            )

            self.freeze_encoder = True
            self.scaler_mode = self.config["contrastive_pretraining"]["encoder_config"][
                "scaler_mode"
            ]

        self.train_recordings, self.test_set = split_recordings(
            patient_dataset.recordings, loro_test_id
        )
        self.tuning = {
            "patient_id": patient_dataset.patient_id,
            "encoder_type": encoder_type,
            "outer_test_recording": loro_test_id.removesuffix(".pt"),
            "training_recordings": [
                name.removesuffix(".pt") for name in self.train_recordings
            ],
            "visual_response_ms": patient_dataset.visual_response,
            "mask_indices": patient_dataset.mask_indices,
            "bandpass_filter": list(patient_dataset.bandpass_filter)
            if patient_dataset.bandpass_filter is not None
            else None,
            "use_attention": bool(use_attention),
        }
        self.trainer = None
        self.trained_config = None
        self.fit_results = None

    def _input_shape(self):
        recording = next(iter(self.patient_dataset.recordings.values()))
        sample = recording[0][0]

        # Each trial has shape [C, T]
        return sample.shape[0], sample.shape[1]

    def create_model(self, params=None):
        if params is None:
            params = validate_config(
                self.config,
                self.encoder_type,
            )

        C, T = self._input_shape()
        num_classes = self.patient_dataset.get_num_classes()

        if self.use_attention and self.attn_hidden_dim is None:
            raise ValueError(
                "attn_hidden_dim is required when attention pooling is enabled"
            )

        encoder = build_encoder(
            self.encoder_type,
            params,
            C,
            T,
            pool_output=not self.use_attention,
        ).to(self.device)

        model = EncoderClassifier(
            encoder=encoder,
            num_classes=num_classes,
            dropout=params["dropout"],
            channel_dropout=params["channel_dropout"],
            freeze_encoder=self.freeze_encoder,
            adapter_dim=params["adapter_dim"] or None,
            use_attention=self.use_attention,
            attn_hidden_dim=self.attn_hidden_dim,
        ).to(self.device)

        model.scaler = ChannelScaler(mode=self.scaler_mode)

        optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=params["lr"],
            weight_decay=params["weight_decay"],
        )

        return model, optimizer

    def _make_trainer(self):
        if self.trainer is None:
            self.trained_config = validate_config(self.config, self.encoder_type)
            model, optimizer = self.create_model(self.trained_config)
            params = self.trained_config
            self.trainer = Trainer(
                model=model,
                optimizer=optimizer,
                device=self.device,
                use_mixup=params["use_mixup"],
                mixup_alpha=params["mixup_alpha"] if params["use_mixup"] else 0.0,
                mixup_layers=params["mixup_layers"] if params["use_mixup"] else (),
                label_smoothing=params["label_smoothing"],
                representation_fit=self.representation_fit,
            )
        return self.trainer

    def tune(self):
        trainer = self._make_trainer()
        _, folds = get_loro_splits(self.patient_dataset.recordings, self.loro_test_id)
        results = trainer.cross_validate(
            folds,
            batch_size=self.config["batch_size"],
            n_epochs=self.config["n_epochs"],
            patience=self.config["patience"],
        )
        self.config.update(results, loro_tuning=self.tuning)
        self.trained_config = copy.deepcopy(self.config)
        if wandb.run is not None:
            wandb.config.update({"loro_tuning": self.tuning}, allow_val_change=True)
        return copy.deepcopy(self.config)

    def fit(self):
        if self.config.get("loro_tuning") != self.tuning:
            raise ValueError(
                "Use a sweep configuration saved for this participant, outer holdout, and input settings"
            )
        fold_epochs = self.config.get("fold_best_epochs")
        if fold_epochs is not None and len(fold_epochs) != len(self.train_recordings):
            raise ValueError(
                "Saved inner folds do not match the outer-training recordings"
            )
        epochs = self.config.get("final_fit_epochs")
        if epochs is None:
            epochs = select_epochs(fold_epochs)
        owns_wandb_run = wandb.run is None
        if owns_wandb_run:
            wandb.init(
                project="ecog_visual_decoding",
                name=self.wandb_run_name,
                config=self.config,
                mode="online" if self.enable_wandb else "disabled",
            )
        try:
            trainer = self._make_trainer()
            train_set = ConcatDataset(list(self.train_recordings.values()))
            model, criterion = trainer.fit(train_set, self.config["batch_size"], epochs)
            losses = self.config.get("fold_best_val_losses")
            self.fit_results = {
                "best_val_loss": min(losses)
                if losses
                else self.config.get("cv_val_loss"),
                "cv_val_loss": self.config.get("cv_val_loss"),
                "fold_best_val_losses": losses,
                "fold_best_epochs": fold_epochs,
                "validation_procedure": "hpo_cv_not_repeated",
                "train_cv": False,
                "final_fit_procedure": "full_outer_train_refit_saved_epochs",
                "final_fit_epochs": epochs,
                "final_fit_recordings": list(self.train_recordings),
            }
            return model, criterion
        finally:
            if owns_wandb_run:
                wandb.finish()

    def test(self, model, criterion):
        if self.fit_results is None:
            raise RuntimeError("Train the pipeline before testing")

        evaluator = Evaluation(
            model=model,
            criterion=criterion,
            device=self.device,
            test_dataset=self.test_set,
        )

        test_batch_size = 4 * self.trained_config["batch_size"]

        return evaluator.test(test_batch_size)
