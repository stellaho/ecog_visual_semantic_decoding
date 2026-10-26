import copy
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset

from model.components import ContrastiveHead, ECoGAugmenter, TemporalAttention
from model.encoder import build_encoder
from scripts.utils.regions import load_region_metadata, validate_channel_selection
from scripts.utils.splits import split_recordings
from scripts.visual_decoding.simclr import SimCLRTrainer

PROTOCOL = "transductive_nested_loro"
PROTOCOLS = {
    "transductive": PROTOCOL,
    "inductive": "inductive_nested_loro",
}

ENCODER_KEYS = (
    "eegnet_f1",
    "eegnet_depth_multiplier",
    "eegnet_f2",
    "eegnet_temporal_kernel",
    "eegnet_separable_kernel",
    "eegnet_pool1",
    "eegnet_pool2",
    "eegnet_spatial_max_norm",
    "base_channels",
    "n_layers",
    "kernel_size",
    "dilation",
    "dilation_base",
    "dropout",
    "n_heads",
    "pos_encoding",
    "attn_hidden_dim",
    "scaler_mode",
)


def recording_id(value):
    return value.removesuffix(".pt")


def simclr_mode(simclr_config):
    for mode, protocol in PROTOCOLS.items():
        if simclr_config.get("protocol") == protocol:
            return mode

    raise ValueError("Unknown SimCLR pretraining protocol")


def protocol_suffix(suffix, mode="transductive"):
    if mode not in PROTOCOLS:
        raise ValueError(f"Unknown pretraining mode: {mode}")

    suffix = str(suffix or "").strip()

    for label in PROTOCOLS:
        if suffix == label or suffix.startswith(f"{label}_"):
            if label != mode:
                raise ValueError(
                    "Mode suffix and pretraining mode must agree"
                )
            return suffix

    return f"{mode}_{suffix}" if suffix else mode


def validate_recording_access(simclr_config, recordings):
    mode = simclr_mode(simclr_config)

    outer_test = recording_id(
        simclr_config["outer_test_recording"]
    )

    training = [
        recording_id(key)
        for key in simclr_config["training_recordings"]
    ]

    requested = [
        recording_id(key)
        for key in recordings
    ]

    if outer_test in training or "Val01" in training:
        raise ValueError(
            "Outer-test and Val01 recordings cannot enter "
            "the training/scaler manifest"
        )

    allowed = set(training)

    if mode == "transductive":
        allowed.add(outer_test)

    if not requested or len(requested) != len(set(requested)):
        raise ValueError(
            "Pretraining recording list must be non-empty and unique"
        )

    if "Val01" in requested or not set(requested) <= allowed:
        raise ValueError(
            "Pretraining requested a recording outside the allowed manifest"
        )

    return requested


def representation_recordings(simclr_config, training_recordings):
    training = validate_recording_access(
        simclr_config,
        training_recordings,
    )

    recordings = list(training)

    if simclr_mode(simclr_config) == "transductive":
        recordings.append(
            recording_id(simclr_config["outer_test_recording"])
        )

    return sorted(recordings)


def outer_training_recordings(patient_dataset, test_id):
    train, _ = split_recordings(
        patient_dataset.recordings,
        test_id,
    )

    return sorted(
        recording_id(name)
        for name in train
    )


def validate_simclr_config(
    simclr_config,
    patient_dataset,
    test_id,
    encoder_type,
    use_attention,
):
    expected = outer_training_recordings(
        patient_dataset,
        test_id,
    )

    validate_recording_access(
        simclr_config,
        simclr_config["training_recordings"],
    )

    if (
        simclr_config.get("outer_test_recording")
        != recording_id(test_id)
    ):
        raise ValueError(
            "Pretraining config belongs to a different outer test recording"
        )

    if simclr_config.get("training_recordings") != expected:
        raise ValueError(
            "Pretraining recording manifest differs from outer-training data"
        )

    for key, actual in (
        ("patient_id", patient_dataset.patient_id),
        ("encoder_type", encoder_type),
        ("use_attention", bool(use_attention)),
        ("visual_response_ms", patient_dataset.visual_response),
        ("mask_indices", patient_dataset.mask_indices),
        ("bandpass_filter", patient_dataset.bandpass_filter),
    ):
        saved = simclr_config.get(key)

        if key in {"mask_indices", "bandpass_filter"}:
            saved = list(saved) if saved is not None else None
            actual = list(actual) if actual is not None else None

        if saved != actual:
            raise ValueError(
                f"Pretraining {key} does not match classifier data/settings"
            )

    epochs = simclr_config["encoder_config"].get(
        "final_fit_epochs"
    )

    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
        raise ValueError(
            "Encoder config must contain positive final_fit_epochs from inner CV"
        )


def build_simclr(simclr_config, device, seed):
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    config = simclr_config["encoder_config"]

    encoder = build_encoder(
        encoder_type=simclr_config["encoder_type"],
        n_channels=len(simclr_config["mask_indices"]),
        n_time=simclr_config["visual_response_ms"] // 2,
        params=config,
        pool_output=not simclr_config["use_attention"],
    )

    attention = None

    if simclr_config["use_attention"]:
        attention = TemporalAttention(
            n_channels=encoder.n_outputs,
            hidden_dim=config["attn_hidden_dim"],
        ).to(device)

    head = ContrastiveHead(
        n_inputs=encoder.n_outputs,
        proj_dim=config["proj_dim"],
    ).to(device)

    augmenter = ECoGAugmenter(
        channel_dropout_p=config["aug_channel_dropout"],
        time_shift_max=config["aug_time_shift"],
        noise_std=config["aug_noise_std"],
    ).to(device)

    return SimCLRTrainer(
        encoder,
        head,
        augmenter,
        attention,
        device,
    )


def load_recordings(simclr_config, recordings):
    recordings = validate_recording_access(
        simclr_config,
        recordings,
    )

    patient_id = simclr_config["patient_id"]
    window_ms = simclr_config["visual_response_ms"]
    channels = simclr_config["mask_indices"]

    data_dir = simclr_config.get(
        "unlabeled_data_dir",
        "./data/visual_decoding/processed/unlabeled",
    )

    folder = Path(data_dir) / patient_id / f"{window_ms}ms"

    if simclr_config["bandpass_filter"] is not None:
        low, high = simclr_config["bandpass_filter"]
        folder = folder.with_name(
            f"{folder.name}_bp{int(low)}-{int(high)}"
        )

    sub_info, _ = load_region_metadata(
        simclr_config.get("config_dir", "./config")
    )

    expected_channels = len(
        sub_info[patient_id]["elec_region_idx"]
    )

    tensors = []
    indices = {}
    offset = 0

    for recording in recordings:
        path = folder / f"{recording}.pt"
        X = torch.load(path, weights_only=False)["X"]

        if X.ndim != 3 or X.shape[0] < 2:
            raise ValueError(
                f"{path}: expected X [trials, channels, time] "
                "with at least two trials"
            )

        selected = validate_channel_selection(
            channels,
            X.shape[1],
            expected_n_channels=expected_channels,
            source=path,
        )

        if not selected or X.shape[2] != window_ms // 2:
            raise ValueError(
                f"{path}: unlabeled channel/time shape differs "
                "from classifier input"
            )

        tensors.append(X[:, selected, :])

        indices[recording] = list(
            range(offset, offset + len(X))
        )

        offset += len(X)

    return torch.cat(tensors, dim=0), indices


class SimCLRFit:
    def __init__(
        self,
        simclr_config,
        patient_dataset,
        test_id,
        encoder_type,
        use_attention,
        device,
    ):
        self.simclr_config = copy.deepcopy(
            simclr_config
        )

        validate_simclr_config(
            simclr_config,
            patient_dataset,
            test_id,
            encoder_type,
            use_attention,
        )

        self.recordings = patient_dataset.recordings
        self.device = device
        self.seed = torch.initial_seed()

    def __call__(self, model, train_set):
        if not isinstance(train_set, ConcatDataset):
            raise ValueError(
                "Recording-scoped pretraining requires LORO recording datasets"
            )

        if not model.freeze_encoder:
            raise ValueError(
                "SimCLR downstream encoder must remain frozen"
            )

        training_ids = []

        for dataset in train_set.datasets:
            matches = [
                recording_id(key)
                for key, value in self.recordings.items()
                if value is dataset
            ]

            if len(matches) != 1:
                raise ValueError(
                    "Cannot identify the pretraining recording "
                    "for a training dataset"
                )

            training_ids.extend(matches)

        training_ids = sorted(training_ids)

        representation_ids = representation_recordings(
            self.simclr_config,
            training_ids,
        )

        config = self.simclr_config["encoder_config"]

        devices = (
            list(range(torch.cuda.device_count()))
            if torch.cuda.is_available()
            else []
        )

        with torch.random.fork_rng(devices=devices):
            simclr_trainer = build_simclr(
                self.simclr_config,
                self.device,
                self.seed,
            )

            X, indices = load_recordings(
                self.simclr_config,
                representation_ids,
            )

            scaler_indices = [
                index
                for recording in training_ids
                for index in indices[recording]
            ]

            simclr_trainer.fit(
                X,
                batch_size=config["batch_size"],
                n_epochs=config["final_fit_epochs"],
                lr=config["lr"],
                weight_decay=config["weight_decay"],
                temperature=config["temperature"],
                scaler_mode=config["scaler_mode"],
                scaler_fit_indices=scaler_indices,
            )

        model.encoder.load_state_dict(
            simclr_trainer.encoder.state_dict()
        )

        model.encoder.requires_grad_(False)
        model.encoder.eval()

        if model.attention is not None:
            model.attention.load_state_dict(
                simclr_trainer.attention.state_dict()
            )

        model.scaler = copy.deepcopy(
            simclr_trainer.scaler
        )

        model.pretraining_provenance = {
            "protocol": self.simclr_config["protocol"],
            "training_recordings": training_ids,
            "pretraining_recordings": representation_ids,
            "scaler_fit_recordings": training_ids,
            "test_time_adaptation": False,
            "outer_test_recording": self.simclr_config[
                "outer_test_recording"
            ],
            "seed": self.seed,
            "epochs": config["final_fit_epochs"],
        }