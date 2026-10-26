import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
)


def compute_metrics(predictions, labels, test_loss=0.0):
    predictions = np.asarray(predictions)
    labels = np.asarray(labels)

    classes = np.union1d(labels, predictions)
    cm = confusion_matrix(labels, predictions, labels=classes)

    return {
        "balanced_accuracy": balanced_accuracy_score(labels, predictions),
        "kappa": cohen_kappa_score(labels, predictions),
        "test_loss": test_loss,
        "n_classes": len(classes),
        "confusion_matrix": cm.tolist(),
    }


def build_mode_name(
    mode="baseline",
    use_attention=None,
    adapter_dim=None,
    bandpass_filter=None,
    mode_suffix=None,
):
    if mode == "simclr":
        pooling = "att" if use_attention else "gap"
        # Preserve existing result identifiers.
        name = f"pretrain_simclr_{pooling}"
    elif use_attention is not None:
        pooling = "att" if use_attention else "gap"
        name = f"{mode}_{pooling}"
    else:
        name = mode

    if adapter_dim is not None and adapter_dim > 0:
        name += "_adapter"

    if bandpass_filter is not None:
        low, high = bandpass_filter
        name += f"_bp{int(low)}-{int(high)}"

    if mode_suffix:
        suffix = str(mode_suffix).strip()
        if suffix:
            name += f"_{suffix}"

    return name


def save_results_to_csv(
    subject_id,
    encoder,
    random_seed,
    loro_test_id,
    visual_response_ms,
    results,
    mode="baseline",
    use_attention=None,
    region_indices=None,
    adapter_dim=None,
    bandpass_filter=None,
    mode_suffix=None,
    results_path="./results/visual_decoding/experiment_results.csv",
):
    results_path = Path(results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    full_mode = build_mode_name(
        mode=mode,
        use_attention=use_attention,
        adapter_dim=adapter_dim,
        bandpass_filter=bandpass_filter,
        mode_suffix=mode_suffix,
    )

    def round_metric(value):
        if isinstance(value, (int, float, np.integer, np.floating)):
            return round(value, 4)
        return value

    regions = (
        ",".join(map(str, region_indices))
        if region_indices
        else "default"
    )

    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "subject_id": subject_id,
        "mode": full_mode,
        "seed": random_seed,
        "encoder": encoder,
        "test_recording": loro_test_id,
        "visual_ms": visual_response_ms,
        "region_indices": regions,
        "balanced_acc": round_metric(results.get("balanced_accuracy")),
        "kappa": round_metric(results.get("kappa")),
        "test_loss": round_metric(results.get("test_loss")),
        "n_classes": results.get("n_classes"),
        "confusion_matrix": json.dumps(
            results.get("confusion_matrix", [])
        ),
    }

    if results_path.exists():
        df = pd.read_csv(results_path)

        mask = (
            (df["subject_id"] == subject_id)
            & (df["encoder"] == encoder)
            & (df["mode"] == full_mode)
            & (df["seed"] == random_seed)
            & (df["test_recording"] == loro_test_id)
            & (df["visual_ms"] == visual_response_ms)
            & (df["region_indices"] == regions)
        )

        df = df[~mask]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df = df.sort_values("timestamp")
    df.to_csv(results_path, index=False)

    return row