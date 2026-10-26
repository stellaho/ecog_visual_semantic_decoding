import json
from pathlib import Path

import numpy as np
import wandb

from meta_learning.protonet import build_protonet
from scripts.utils.splits import get_loro_splits
from scripts.visual_decoding.protonet_trainer import ProtoNetTrainer
from sweep.utils import create_sweep, save_best_run

FIXED_N_EPISODES = 2000


def run_protonet_sweep(
    patient_dataset,
    loro_test_id,
    device,
    sweep_config_path,
    sweep_count=50,
    encoder_type="tcn",
    use_attention=False,
):
    sweep_id = create_sweep(sweep_config_path, use_attention=use_attention)
    _, cv_folds = get_loro_splits(
        patient_dataset.recordings,
        loro_test_id,
    )

    first_train_set, _ = cv_folds[0]
    sample_x, _ = first_train_set[0]
    n_inputs = sample_x.shape[0]
    n_time = sample_x.shape[-1]

    def train_fn():
        with wandb.init() as run:
            config = dict(wandb.config)
            fold_val_accs = []
            fold_episode_sizes = []

            for fold_idx, (train_set, val_set) in enumerate(cv_folds):
                model = build_protonet(
                    encoder_type=encoder_type,
                    n_inputs=n_inputs,
                    n_time=n_time,
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

                _, best_val_acc = trainer.train_fold(
                    train_dataset=train_set,
                    val_dataset=val_set,
                    n_episodes=config.get("n_episodes", FIXED_N_EPISODES),
                    k_shot=config["k_shot"],
                    n_query=config["n_query"],
                    lr=config["learning_rate"],
                    weight_decay=config["weight_decay"],
                    label_smoothing=config["label_smoothing"],
                    patience=config["patience"],
                )

                fold_val_accs.append(best_val_acc)
                fold_episode_sizes.append(trainer.episode_sizes)

                wandb.log(
                    {f"fold_{fold_idx + 1}_val_acc": best_val_acc}
                )

            run.summary["fold_episode_sizes"] = fold_episode_sizes

            mean_val_acc = np.mean(fold_val_accs)
            wandb.log({"val_query_acc": mean_val_acc})

    wandb.agent(
        sweep_id,
        function=train_fn,
        count=sweep_count,
    )

    return sweep_id


def run_and_save_protonet_sweep(
    patient_dataset,
    loro_test_id,
    device,
    sweep_config_path,
    output_path,
    sweep_count=50,
    encoder_type="tcn",
    mask_indices=None,
    visual_response_ms=500,
    use_attention=False,
):
    sweep_id = run_protonet_sweep(
        patient_dataset=patient_dataset,
        loro_test_id=loro_test_id,
        device=device,
        sweep_config_path=sweep_config_path,
        sweep_count=sweep_count,
        encoder_type=encoder_type,
        use_attention=use_attention,
    )

    best_config = save_best_run(
        sweep_id=sweep_id,
        metric="val_query_acc",
        greater_is_better=True,
        output_path=output_path,
    )

    if best_config is None:
        raise RuntimeError("No completed ProtoNet sweep runs")

    test_recording = Path(loro_test_id).stem

    best_config["protonet_tuning"] = {
        "patient_id": patient_dataset.patient_id,
        "encoder_type": encoder_type,
        "outer_test_recording": test_recording,
        "training_recordings": [
            Path(name).stem
            for name in patient_dataset.recordings
            if Path(name).stem not in {test_recording, "Val01"}
        ],
        "use_attention": use_attention,
        "visual_response_ms": visual_response_ms,
        "mask_indices": mask_indices,
        "bandpass_filter": patient_dataset.bandpass_filter,
    }

    best_config.setdefault("n_episodes", FIXED_N_EPISODES)

    Path(output_path).write_text(
        json.dumps(best_config, indent=2) + "\n"
    )

    return best_config