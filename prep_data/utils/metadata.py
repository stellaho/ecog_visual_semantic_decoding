import json
import logging
from pathlib import Path

import numpy as np


def electrode_region_index_to_list(region_idx_mat):
    region_idx = []

    # Keep missing values so electrode indices stay aligned with the signal channels.
    for value in np.asarray(region_idx_mat).squeeze():
        if np.isnan(value):
            region_idx.append(None)
        else:
            region_idx.append(int(value))

    return region_idx


def ensure_electrode_info(patient_id, project_root):
    project_root = Path(project_root)
    sub_info_path = project_root / "config" / "sub_info.json"

    with sub_info_path.open() as file:
        sub_info = json.load(file)

    feature_mat_path = (
        project_root
        / "data"
        / "visual_decoding"
        / "features"
        / f"feature_{patient_id}.mat"
    )

    # Use the existing metadata if the feature file is unavailable.
    if not feature_mat_path.exists():
        if patient_id in sub_info and "elec_region_idx" in sub_info[patient_id]:
            return

        logging.getLogger(__name__).warning(
            f"Feature file not found: {feature_mat_path}"
        )
        return

    from scipy.io import loadmat

    mat = loadmat(feature_mat_path)
    region_idx = electrode_region_index_to_list(
        mat["electrode_region_index"]
    )

    subject_info = sub_info.setdefault(patient_id, {})

    # Avoid rewriting the config when the stored metadata is already up to date.
    if (
        subject_info.get("elec_region_idx") == region_idx
        and subject_info.get("n_chls") == len(region_idx)
    ):
        return

    subject_info["elec_region_idx"] = region_idx
    subject_info["n_chls"] = len(region_idx)

    with sub_info_path.open("w") as file:
        json.dump(sub_info, file, indent=2)