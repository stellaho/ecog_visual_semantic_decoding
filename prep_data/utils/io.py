import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from prep_data.utils.constants import INT_TO_LABEL


def load_global_config(project_root):
    config_path = Path(project_root) / "config" / "global.json"

    with config_path.open() as file:
        return json.load(file)


def load_labeled_images(labels_csv):
    labels_df = pd.read_csv(labels_csv)

    # Val01 is not part of the training recordings.
    return labels_df[labels_df["folder_name"] != "Val01"]


def save_labeled_segments(
    save_dir,
    folder_name,
    segments,
    labels,
    ids,
):
    x_tensor = torch.tensor(
        np.stack(segments),
        dtype=torch.float32,
    )
    y_tensor = torch.tensor(
        np.array(labels),
        dtype=torch.long,
    )

    save_path = Path(save_dir) / f"{folder_name}.pt"
    torch.save(
        {
            "X": x_tensor,
            "y": y_tensor,
            "ids": ids,
        },
        save_path,
    )

    return save_path, x_tensor


def save_unlabeled_segments(
    save_dir,
    folder_name,
    segments,
    ids,
):
    x_tensor = torch.tensor(
        np.stack(segments),
        dtype=torch.float32,
    )

    save_path = Path(save_dir) / f"{folder_name}.pt"
    torch.save(
        {
            "X": x_tensor,
            "ids": ids,
        },
        save_path,
    )

    return save_path, x_tensor


def class_distribution(labels):
    unique, counts = np.unique(labels, return_counts=True)

    return {
        INT_TO_LABEL[label]: count
        for label, count in zip(unique, counts)
    }