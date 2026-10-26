import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from dataset.visual_decoding.clf_dataset import PatientDataset
from scripts.utils.regions import parse_region_indices, resolve_selected_channels
from scripts.utils.results import build_mode_name, save_results_to_csv
from scripts.visual_decoding.pipeline import (
    ENCODERS,
    ClassificationPipeline,
    validate_config,
)


SEEDS = [42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def prepare_config(args, path):
    config = json.loads(path.read_text())

    if args.mode == "simclr":
        from scripts.visual_decoding.simclr_protocol import ENCODER_KEYS

        simclr_config = config["contrastive_pretraining"]
        encoder_config = simclr_config["encoder_config"]

        config.update(
            {
                key: encoder_config[key]
                for key in ENCODER_KEYS
                if key in encoder_config
            }
        )

    if args.adapter_dim is not None:
        if (
            "adapter_dim" in config
            and int(config["adapter_dim"]) != args.adapter_dim
        ):
            raise ValueError("Config and CLI adapter_dim must agree")

        config["adapter_dim"] = args.adapter_dim

    config["use_mixup"] = args.mode == "mixup"

    if not config["use_mixup"]:
        config["label_smoothing"] = 0.0

    if args.use_attention and config.get("attn_hidden_dim") is None:
        raise KeyError(
            "Attention pooling requires attn_hidden_dim in config"
        )

    if (
        config.get("final_fit_epochs") is None
        and config.get("fold_best_epochs") is None
    ):
        raise KeyError(
            "Evaluation requires final_fit_epochs or fold_best_epochs from HPO"
        )

    if "loro_tuning" not in config:
        raise ValueError(
            "Run a sweep to save outer-holdout metadata before evaluation"
        )

    return validate_config(config, args.encoder)


def select_folds(dataset, requested):
    available = [
        Path(name).stem
        for name in sorted(dataset.recordings)
        if name.startswith("Trn")
    ][:6]

    if requested:
        folds = list(
            dict.fromkeys(
                Path(name.strip()).stem
                for name in requested.split(",")
                if name.strip()
            )
        )
    else:
        folds = available

    if not folds or not set(folds).issubset(available):
        raise ValueError(
            f"Choose LORO recordings from {available}"
        )

    return folds


def save_checkpoint(
    args,
    pipeline,
    model,
    config_path,
    fold,
    seed,
    regions,
    indices,
    run_key,
):
    directory = (
        Path(args.save_checkpoints_dir)
        / run_key
        / args.config_mode
        / fold
        / f"seed_{seed}"
    )
    directory.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": pipeline.trained_config,
            "pretraining_provenance": getattr(
                model,
                "pretraining_provenance",
                None,
            ),
            "config_path": str(config_path),
            "fold_id": fold,
            "seed": seed,
            "run_args": vars(args).copy(),
            "region_indices": regions,
            "indices_to_keep": indices,
            **pipeline.fit_results,
        },
        directory / "model.pt",
    )

    scaler = getattr(model, "scaler", None)

    if scaler is not None:
        torch.save(
            {
                "mode": scaler.mode,
                "center": scaler.center,
                "scale": scaler.scale,
                "fitted": scaler.fitted,
            },
            directory / "scaler.pt",
        )


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate visual decoding configs"
    )

    parser.add_argument("--patient_id", required=True)
    parser.add_argument(
        "--encoder",
        default="resnet",
        choices=ENCODERS,
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["baseline", "mixup", "simclr"],
    )

    parser.add_argument(
        "--config_mode",
        default="fixed",
        choices=["fixed", "per_fold"],
    )
    parser.add_argument(
        "--config",
        help="JSON path for fixed-config evaluation",
    )
    parser.add_argument(
        "--config_dir",
        help="Directory containing fold-specific configs",
    )

    parser.add_argument(
        "--num_seeds",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--seed_values",
        help="Comma-separated seeds; overrides --num_seeds",
    )
    parser.add_argument(
        "--loro_recordings",
        help="Comma-separated folds, e.g. Trn01,Trn03",
    )

    parser.add_argument(
        "--visual_response_ms",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--enable_wandb",
        action="store_true",
    )
    parser.add_argument(
        "--use_attention",
        action="store_true",
    )
    parser.add_argument(
        "--scaler_mode",
        default="standard",
        choices=["standard", "robust"],
    )
    parser.add_argument(
        "--region_indices",
        help="Comma-separated brain-region indices",
    )
    parser.add_argument(
        "--bandpass_filter",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        help="Load tensors preprocessed for this frequency band",
    )
    parser.add_argument(
        "--mode_suffix",
        default="",
    )
    parser.add_argument(
        "--adapter_dim",
        type=int,
    )
    parser.add_argument(
        "--results_path",
        default="./results/visual_decoding/experiment_results.csv",
    )
    parser.add_argument(
        "--save_checkpoints_dir",
        help="Optional directory for model/scaler checkpoints",
    )

    args = parser.parse_args()

    if args.mode == "simclr":
        from scripts.visual_decoding.simclr_protocol import protocol_suffix

        try:
            args.mode_suffix = protocol_suffix(args.mode_suffix)
        except ValueError as exc:
            parser.error(str(exc))

    if args.config_mode == "fixed" and not args.config:
        parser.error(
            "--config is required for fixed-config evaluation"
        )

    if args.adapter_dim is not None and args.adapter_dim < 0:
        parser.error("--adapter_dim must be nonnegative")

    if args.visual_response_ms < 1:
        parser.error("--visual_response_ms must be positive")

    try:
        if args.seed_values:
            seeds = list(
                dict.fromkeys(
                    int(seed.strip())
                    for seed in args.seed_values.split(",")
                    if seed.strip()
                )
            )
        else:
            if not 1 <= args.num_seeds <= len(SEEDS):
                raise ValueError(
                    f"--num_seeds must be between 1 and {len(SEEDS)}"
                )
            seeds = SEEDS[: args.num_seeds]

        if not seeds or any(
            seed < 0 or seed >= 2**32
            for seed in seeds
        ):
            raise ValueError(
                "Provide seeds in [0, 2**32)"
            )

        if args.bandpass_filter:
            low, high = args.bandpass_filter

            if not 0 < low < high < 250:
                raise ValueError(
                    "Filter cutoffs must satisfy "
                    "0 < low < high < 250 Hz"
                )

        regions = parse_region_indices(
            args.region_indices
        )

        indices, _, _ = resolve_selected_channels(
            args.patient_id,
            regions,
        )

    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))

    dataset = PatientDataset(
        patient_id=args.patient_id,
        mask_indices=indices,
        visual_response=args.visual_response_ms,
        bandpass_filter=args.bandpass_filter,
    )

    mode_name = build_mode_name(
        args.mode,
        use_attention=args.use_attention,
        mode_suffix=args.mode_suffix,
    )

    band_name = ""

    if args.bandpass_filter:
        low, high = args.bandpass_filter
        band_name = f"_bp{int(low)}-{int(high)}"

    run_key = (
        f"{args.patient_id}_"
        f"{args.encoder}_"
        f"{mode_name}"
        f"{band_name}"
    )

    config_dir = Path(
        args.config_dir
        or (
            f"config/visual_decoding/subjects/"
            f"{args.patient_id}/{args.encoder}"
        )
    )

    try:
        fixed_config = None

        if args.config_mode == "fixed":
            fixed_config = prepare_config(
                args,
                Path(args.config),
            )

        requested = args.loro_recordings

        if requested is None and fixed_config is not None:
            requested = fixed_config[
                "loro_tuning"
            ]["outer_test_recording"]

        folds = select_folds(
            dataset,
            requested,
        )

        configurations = {}

        for fold in folds:
            if args.config_mode == "fixed":
                path = Path(args.config)
                config = fixed_config
            else:
                path = (
                    config_dir
                    / f"{run_key}_{fold}_best.json"
                )
                config = prepare_config(
                    args,
                    path,
                )

            if (
                config["loro_tuning"]["outer_test_recording"]
                != fold
            ):
                raise ValueError(
                    f"Configuration was not tuned "
                    f"for outer holdout {fold}"
                )

            if args.mode == "simclr":
                from scripts.visual_decoding.simclr_protocol import (
                    validate_simclr_config,
                )

                validate_simclr_config(
                    config["contrastive_pretraining"],
                    dataset,
                    fold,
                    args.encoder,
                    args.use_attention,
                )

            configurations[fold] = (
                path,
                config,
            )

    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    results = []

    for fold in folds:
        path, config = configurations[fold]

        for seed in seeds:
            set_seed(seed)

            pipeline = ClassificationPipeline(
                patient_dataset=dataset,
                loro_test_id=fold,
                config=config,
                device=device,
                encoder_type=args.encoder,
                enable_wandb=args.enable_wandb,
                use_attention=args.use_attention,
                attn_hidden_dim=config.get(
                    "attn_hidden_dim"
                ),
                scaler_mode=config.get(
                    "scaler_mode",
                    args.scaler_mode,
                ),
                wandb_run_name=(
                    f"{run_key}_"
                    f"{args.config_mode}_"
                    f"{fold}_seed{seed}"
                ),
            )

            model, criterion = pipeline.fit()
            metrics = pipeline.test(
                model,
                criterion,
            )

            metrics.update(
                config_mode=args.config_mode,
                config_path=str(path),
                **pipeline.fit_results,
            )

            results.append(
                {
                    "loro_test_id": fold,
                    "seed": seed,
                    **metrics,
                }
            )

            save_results_to_csv(
                subject_id=args.patient_id,
                encoder=args.encoder,
                random_seed=seed,
                loro_test_id=fold,
                visual_response_ms=args.visual_response_ms,
                results=metrics,
                mode=args.mode,
                use_attention=args.use_attention,
                region_indices=regions,
                adapter_dim=config["adapter_dim"] or None,
                bandpass_filter=args.bandpass_filter,
                mode_suffix=args.mode_suffix,
                results_path=args.results_path,
            )

            if args.save_checkpoints_dir:
                save_checkpoint(
                    args,
                    pipeline,
                    model,
                    path,
                    fold,
                    seed,
                    regions,
                    indices,
                    run_key,
                )

    return results


if __name__ == "__main__":
    main()