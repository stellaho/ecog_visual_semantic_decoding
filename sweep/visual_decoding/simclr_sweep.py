import json
from pathlib import Path

import numpy as np
import wandb

from scripts.visual_decoding.pipeline import ClassificationPipeline
from scripts.visual_decoding.simclr_protocol import (
    ENCODER_KEYS,
    PROTOCOLS,
    build_simclr,
    load_recordings,
    outer_training_recordings,
    recording_id,
    representation_recordings,
    validate_simclr_config,
)
from scripts.visual_decoding.train import select_epochs
from sweep.utils import create_sweep, save_best_run


def run_encoder_sweep(
    simclr_config,
    device,
    sweep_config_path,
    sweep_count=50,
):
    training_ids = simclr_config["training_recordings"]
    representation_ids = representation_recordings(
        simclr_config,
        training_ids,
    )
    sweep_id = create_sweep(
        sweep_config_path, use_attention=simclr_config["use_attention"]
    )

    def train_fn():
        with wandb.init() as run:
            config = dict(wandb.config)
            trial_config = dict(
                simclr_config,
                encoder_config=config,
            )

            losses = []
            epochs = []

            X, indices = load_recordings(
                trial_config,
                representation_ids,
            )

            # Inner LORO over the outer-training recordings
            for val_id in training_ids:
                simclr_trainer = build_simclr(
                    trial_config,
                    device,
                    seed=42,
                )

                train_indices = [
                    index
                    for recording in representation_ids
                    if recording != val_id
                    for index in indices[recording]
                ]

                loss, best_epoch = simclr_trainer.tune(
                    X,
                    train_indices,
                    indices[val_id],
                    batch_size=config["batch_size"],
                    n_epochs=config["n_epochs"],
                    lr=config["lr"],
                    patience=config["patience"],
                    weight_decay=config["weight_decay"],
                    scaler_mode=config["scaler_mode"],
                    temperature=config["temperature"],
                )

                if not np.isfinite(loss) or best_epoch is None:
                    raise RuntimeError(
                        f"No finite pretraining validation loss for {val_id}"
                    )

                losses.append(float(loss))
                epochs.append(best_epoch)

            cv_val_loss = float(np.mean(losses))
            final_epochs = select_epochs(epochs)

            wandb.log(
                {
                    "cv_val_loss": cv_val_loss,
                    "final_fit_epochs": final_epochs,
                }
            )
            run.summary.update(
                {
                    "fold_best_val_losses": losses,
                    "fold_best_epochs": epochs,
                }
            )

    wandb.agent(
        sweep_id,
        function=train_fn,
        count=sweep_count,
    )

    return sweep_id


def run_classifier_sweep(
    patient_dataset,
    loro_test_id,
    device,
    sweep_config_path=None,
    sweep_count=50,
    encoder_type="tcn",
    adapter_dim=None,
    simclr_config=None,
):
    if simclr_config is None:
        raise ValueError(
            "Classifier sweeps require a nested LORO pretraining config."
        )

    validate_simclr_config(
        simclr_config,
        patient_dataset,
        loro_test_id,
        encoder_type,
        simclr_config["use_attention"],
    )

    encoder_config = simclr_config["encoder_config"]
    sweep_id = create_sweep(
        sweep_config_path, adapter_dim=adapter_dim
    )

    def train_fn():
        with wandb.init():
            config = dict(wandb.config)

            for key in ENCODER_KEYS:
                if key in encoder_config:
                    config[key] = encoder_config[key]

            config["use_mixup"] = False
            config["label_smoothing"] = 0.0
            config["contrastive_pretraining"] = simclr_config

            if adapter_dim is not None:
                if (
                    "adapter_dim" in config
                    and int(config["adapter_dim"]) != int(adapter_dim)
                ):
                    raise ValueError(
                        "Config and CLI adapter_dim must agree"
                    )
                config["adapter_dim"] = adapter_dim
            elif "adapter_dim" not in config:
                config["adapter_dim"] = 0

            wandb.config.update(
                config,
                allow_val_change=True,
            )

            pipeline = ClassificationPipeline(
                patient_dataset=patient_dataset,
                loro_test_id=loro_test_id,
                config=config,
                device=device,
                enable_wandb=True,
                encoder_type=encoder_type,
                freeze_encoder=True,
                use_attention=simclr_config["use_attention"],
                attn_hidden_dim=encoder_config.get("attn_hidden_dim"),
            )

            pipeline.tune()

    wandb.agent(
        sweep_id,
        function=train_fn,
        count=sweep_count,
    )

    return sweep_id


def run_simclr_sweep(
    patient_id,
    encoder_type,
    patient_dataset,
    loro_test_id,
    device,
    encoder_sweep_config,
    classifier_sweep_config,
    encoder_sweep_count=50,
    classifier_sweep_count=50,
    visual_response_ms=500,
    mask_indices=None,
    use_attention=None,
    bandpass_filter=None,
    classifier_adapter_dim=None,
    output_path=None,
    unlabeled_data_dir="./data/visual_decoding/processed/unlabeled",
    config_dir="./config",
):
    if output_path is None:
        raise ValueError(
            "A fold-specific output_path is required"
        )

    if mask_indices is None or len(mask_indices) == 0:
        raise ValueError(
            "Explicit selected channel indices are required"
        )

    simclr_config = {
        "config_dir": str(config_dir),
        "unlabeled_data_dir": str(unlabeled_data_dir),
        "protocol": PROTOCOLS["transductive"],
        "patient_id": patient_id,
        "encoder_type": encoder_type,
        "outer_test_recording": recording_id(loro_test_id),
        "training_recordings": outer_training_recordings(
            patient_dataset,
            loro_test_id,
        ),
        "visual_response_ms": visual_response_ms,
        "mask_indices": list(mask_indices),
        "bandpass_filter": (
            list(bandpass_filter)
            if bandpass_filter is not None
            else None
        ),
        "use_attention": bool(use_attention),
    }

    output_path = Path(output_path)
    encoder_output = output_path.with_name(
        output_path.name.replace(
            "_best.json",
            "_encoder_best.json",
        )
    )

    encoder_sweep_id = run_encoder_sweep(
        simclr_config,
        device,
        encoder_sweep_config,
        encoder_sweep_count,
    )

    encoder_config = save_best_run(
        encoder_sweep_id,
        metric="cv_val_loss",
        greater_is_better=False,
        output_path=str(encoder_output),
    )

    if encoder_config is None:
        raise RuntimeError(
            "No successful encoder sweep trials"
        )

    simclr_config["encoder_config"] = encoder_config

    # Save the encoder settings and calibration protocol together.
    encoder_output.write_text(
        json.dumps(simclr_config, indent=4) + "\n"
    )

    classifier_sweep_id = run_classifier_sweep(
        patient_dataset,
        loro_test_id,
        device,
        sweep_config_path=classifier_sweep_config,
        sweep_count=classifier_sweep_count,
        encoder_type=encoder_type,
        adapter_dim=classifier_adapter_dim,
        simclr_config=simclr_config,
    )

    classifier_config = save_best_run(
        classifier_sweep_id,
        metric="cv_val_loss",
        greater_is_better=False,
        output_path=str(output_path),
    )

    if classifier_config is None:
        raise RuntimeError(
            "No successful classifier sweep trials"
        )

    return {
        "encoder_config": encoder_config,
        "classifier_config": classifier_config,
        "encoder_config_path": str(encoder_output),
        "classifier_config_path": str(output_path),
        "protocol": simclr_config["protocol"],
    }