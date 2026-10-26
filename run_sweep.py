import argparse
from pathlib import Path

import torch

from dataset.visual_decoding.clf_dataset import PatientDataset
from scripts.utils.regions import (
    parse_region_indices,
    region_indices_to_text,
    resolve_selected_channels,
)
from scripts.utils.results import build_mode_name
from sweep.visual_decoding.baseline_sweep import run_and_save_baseline_sweep
from sweep.visual_decoding.mixup_sweep import run_and_save_mixup_sweep
from sweep.visual_decoding.protonet_sweep import run_and_save_protonet_sweep
from sweep.visual_decoding.simclr_sweep import run_simclr_sweep

NYQUIST_HZ = 250.0

ENCODERS = (
    "tcn",
    "resnet",
    "lsnet",
    "lstm",
    "transformer",
    "eegnet",
)

MODES = (
    "baseline",
    "mixup",
    "simclr",
    "protonet",
)


def get_sweep_config(mode, encoder_type="tcn"):
    return (
        f"./config/visual_decoding/sweeps/{mode}/"
        f"{mode}_{encoder_type}_sweep_config.json"
    )


def validate_frequency_band(low, high, name):
    if low <= 0 or high <= 0:
        raise ValueError(
            f"{name} frequencies must be positive, got ({low}, {high})"
        )
    if low >= high:
        raise ValueError(
            f"{name} requires low < high, got ({low}, {high})"
        )
    if high >= NYQUIST_HZ:
        raise ValueError(
            f"{name} high cutoff must be < {NYQUIST_HZ} Hz, got {high}"
        )


def validate_args(parser, args):
    if args.sweep_count < 1:
        parser.error("--sweep_count must be >= 1")

    if args.visual_response_ms <= 0:
        parser.error("--visual_response_ms must be positive")

    if args.adapter_dim is not None and args.adapter_dim < 0:
        parser.error("--adapter_dim must be >= 0")

    if args.mode == "simclr":
        from scripts.visual_decoding.simclr_protocol import protocol_suffix

        try:
            args.mode_suffix = protocol_suffix(
                args.mode_suffix,
                mode="transductive",
            )
        except ValueError as exc:
            parser.error(str(exc))

    if args.bandpass_filter is not None:
        try:
            validate_frequency_band(
                args.bandpass_filter[0],
                args.bandpass_filter[1],
                "Bandpass filter",
            )
        except ValueError as exc:
            parser.error(str(exc))


def resolve_mode_name(args):
    return build_mode_name(
        args.mode,
        use_attention=args.use_attention,
        mode_suffix=args.mode_suffix,
    )


def build_best_config_path(args, mode_name):
    parts = [
        args.patient_id,
        args.encoder,
        mode_name,
    ]

    if args.bandpass_filter:
        low, high = args.bandpass_filter
        parts.append(f"bp{int(low)}-{int(high)}")

    parts.append(Path(args.loro_test_id).stem)

    filename = "_".join(parts) + "_best.json"

    return (
        Path("./config/visual_decoding/subjects")
        / args.patient_id
        / args.encoder
        / filename
    )


def load_patient_dataset(args):
    region_indices = parse_region_indices(args.region_indices)

    indices_to_keep, _, _ = resolve_selected_channels(
        args.patient_id,
        region_indices,
    )

    patient_dataset = PatientDataset(
        patient_id=args.patient_id,
        mask_indices=indices_to_keep,
        visual_response=args.visual_response_ms,
        bandpass_filter=args.bandpass_filter,
    )

    return patient_dataset, indices_to_keep


def run_selected_sweep(
    args,
    patient_dataset,
    indices_to_keep,
    device,
    sweep_config,
    output_path,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path = str(output_path)

    if args.mode == "baseline":
        return run_and_save_baseline_sweep(
            patient_dataset=patient_dataset,
            loro_test_id=args.loro_test_id,
            device=device,
            sweep_config_path=sweep_config,
            output_path=output_path,
            sweep_count=args.sweep_count,
            encoder_type=args.encoder,
            use_attention=args.use_attention,
            adapter_dim=args.adapter_dim,
        )

    if args.mode == "mixup":
        return run_and_save_mixup_sweep(
            patient_dataset=patient_dataset,
            loro_test_id=args.loro_test_id,
            device=device,
            sweep_config_path=sweep_config,
            output_path=output_path,
            sweep_count=args.sweep_count,
            encoder_type=args.encoder,
            use_attention=args.use_attention,
            adapter_dim=args.adapter_dim,
        )

    if args.mode == "simclr":
        classifier_name = (
            "simclr_classifier_eegnet_sweep_config.json"
            if args.encoder == "eegnet"
            else "simclr_classifier_sweep_config.json"
        )
        classifier_sweep_config = (
            f"./config/visual_decoding/sweeps/simclr/{classifier_name}"
        )

        return run_simclr_sweep(
            patient_id=args.patient_id,
            encoder_type=args.encoder,
            patient_dataset=patient_dataset,
            loro_test_id=args.loro_test_id,
            device=device,
            encoder_sweep_config=sweep_config,
            classifier_sweep_config=classifier_sweep_config,
            encoder_sweep_count=args.sweep_count,
            classifier_sweep_count=args.sweep_count,
            visual_response_ms=args.visual_response_ms,
            mask_indices=indices_to_keep,
            use_attention=args.use_attention,
            bandpass_filter=args.bandpass_filter,
            classifier_adapter_dim=args.adapter_dim,
            output_path=output_path,
        )

    if args.mode == "protonet":
        return run_and_save_protonet_sweep(
            patient_dataset=patient_dataset,
            loro_test_id=args.loro_test_id,
            device=device,
            sweep_config_path=sweep_config,
            output_path=output_path,
            sweep_count=args.sweep_count,
            encoder_type=args.encoder,
            mask_indices=indices_to_keep,
            visual_response_ms=args.visual_response_ms,
            use_attention=args.use_attention,
        )

    raise ValueError(f"Unsupported mode: {args.mode}")


def main():
    parser = argparse.ArgumentParser(
        description="ECoG Visual Decoding - Hyperparameter Sweep"
    )

    parser.add_argument(
        "--patient_id",
        required=True,
        help="Participant identifier, e.g. E02",
    )
    parser.add_argument(
        "--encoder",
        default="tcn",
        choices=ENCODERS,
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=MODES,
    )
    parser.add_argument(
        "--sweep_count",
        type=int,
        default=50,
        help="Number of W&B sweep trials",
    )
    parser.add_argument(
        "--loro_test_id",
        default="Trn06",
        help="Outer test recording, e.g. Trn06",
    )
    parser.add_argument(
        "--visual_response_ms",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--use_attention",
        action="store_true",
        help="Use temporal-attention pooling instead of GAP",
    )
    parser.add_argument(
        "--region_indices",
        default=None,
        help=(
            "Comma-separated brain-region indices to include. "
            f"Default: {region_indices_to_text()}"
        ),
    )
    parser.add_argument(
        "--bandpass_filter",
        type=float,
        nargs=2,
        default=None,
        metavar=("LOW", "HIGH"),
        help="Load tensors preprocessed for this frequency band",
    )
    parser.add_argument(
        "--mode_suffix",
        default="",
        help="Optional result-name suffix",
    )
    parser.add_argument(
        "--adapter_dim",
        type=int,
        default=None,
        help="Fix adapter dimension (0 disables it); otherwise use the sweep config",
    )

    args = parser.parse_args()
    validate_args(parser, args)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    sweep_config = get_sweep_config(
        args.mode,
        args.encoder,
    )

    mode_name = resolve_mode_name(args)
    output_path = build_best_config_path(
        args,
        mode_name,
    )

    patient_dataset, indices_to_keep = load_patient_dataset(args)

    return run_selected_sweep(
        args=args,
        patient_dataset=patient_dataset,
        indices_to_keep=indices_to_keep,
        device=device,
        sweep_config=sweep_config,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()