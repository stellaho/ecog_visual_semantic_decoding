import json
from pathlib import Path

import torch
from torch.utils.data import TensorDataset

from scripts.utils.regions import validate_channel_selection


class PatientDataset:
    def __init__(
        self,
        patient_id,
        data_dir="./data/visual_decoding/processed/labeled",
        mask_indices=None,
        visual_response=500,
        bandpass_filter=None,
        config_dir="./config",
    ):
        self.patient_id = patient_id
        self.mask_indices = mask_indices
        self.visual_response = visual_response
        self.bandpass_filter = bandpass_filter
        self.recordings = self._load_data(Path(data_dir), Path(config_dir))

    def _load_data(self, data_dir, config_dir):
        folder = data_dir / self.patient_id / f"{self.visual_response}ms"
        
        # Band-specific signals are preprocessed and stored separately.
        if self.bandpass_filter is not None:
            low, high = self.bandpass_filter
            folder = folder.with_name(
                f"{folder.name}_bp{int(low)}-{int(high)}"
            )

        files = sorted(folder.glob("*.pt"))
        if not files:
            raise FileNotFoundError(f"No tensor files found in {folder}")

        expected_channels = None
        if self.mask_indices is not None:
            with (config_dir / "sub_info.json").open() as file:
                regions = json.load(file)[self.patient_id]["elec_region_idx"]
            expected_channels = len(regions)

        recordings = {}
        
        for path in files:
            data = torch.load(
                path, 
                map_location="cpu", 
                weights_only=False,
            )
            X, y = data["X"], data["y"]
            
            if X.ndim != 3 or y.ndim != 1 or len(X) != len(y) or X.numel() == 0:
                raise ValueError(
                    f"{path}: expected X [trials, channels, time] and y [trials]"
                )

            if self.mask_indices is not None:
                # Keep only electrodes selected for the current analysis
                indices = validate_channel_selection(
                    self.mask_indices,
                    X.shape[1],
                    expected_n_channels=expected_channels,
                    source=str(path),
                )
                
                if not indices:
                    raise ValueError(
                        f"No selected channels for {path.name}"
                    )
                
                X = X[:, indices, :]

            recording_id = path.name.removesuffix(".pt")
            recordings[recording_id] = TensorDataset(X, y)

        return recordings

    def get_num_classes(self):
        labels = torch.cat(
            [recording.tensors[1] for recording in self.recordings.values()]
        )
        return labels.unique().numel()
