import argparse
import json
from pathlib import Path

import torch

from dataset.visual_decoding.clf_dataset import PatientDataset
from scripts.utils.regions import (
    parse_region_indices,
    region_indices_to_text,
    resolve_selected_channels,
)
from scripts.visual_decoding.protonet_trainer import evaluate_loro_all_recordings


SEEDS = [42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144]

ENCODERS = (
    "tcn",
    "resnet",
    "lsnet",
    "lstm",
    "transformer",
    "eegnet",
)


def main():
    parser = argparse.ArgumentParser(
        description="ProtoNet LORO evaluation"
    )

    parser.add_argument(
        "--patient_id",
        default="E02",
    )
    parser.add_argument(
        "--encoder",
        default="resnet",
        choices=ENCODERS,
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to best ProtoNet config",
    )
    parser.add_argument(
        "--num_seeds",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--visual_response_ms",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--use_attention",
        action="store_true",
    )
    parser.add_argument(
        "--bandpass_filter",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        help="Load tensors preprocessed for this frequency band",
    )
    parser.add_argument(
        "--region_indices",
        default=None,
        help=(
            "Comma-separated brain-region indices. "
            f"Default: {region_indices_to_text()}"
        ),
    )

    args = parser.parse_args()

    if not 1 <= args.num_seeds <= len(SEEDS):
        parser.error(
            f"--num_seeds must be between 1 and {len(SEEDS)}"
        )

    if args.visual_response_ms <= 0:
        parser.error("--visual_response_ms must be positive")

    if args.bandpass_filter:
        low, high = args.bandpass_filter
        if not 0 < low < high < 250:
            parser.error(
                "Filter cutoffs must satisfy 0 < low < high < 250 Hz"
            )

    config = json.loads(
        Path(args.config).read_text()
    )

    regions = parse_region_indices(
        args.region_indices
    )

    mask_indices, _, _ = resolve_selected_channels(
        args.patient_id,
        regions,
    )

    patient_dataset = PatientDataset(
        patient_id=args.patient_id,
        mask_indices=mask_indices,
        visual_response=args.visual_response_ms,
        bandpass_filter=args.bandpass_filter,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    seeds = SEEDS[: args.num_seeds]

    return evaluate_loro_all_recordings(
        patient_dataset=patient_dataset,
        config=config,
        encoder_type=args.encoder,
        device=device,
        seeds=seeds,
        mask_indices=mask_indices,
        visual_response_ms=args.visual_response_ms,
        save_to_csv=True,
        use_attention=args.use_attention,
    )


if __name__ == "__main__":
    main()