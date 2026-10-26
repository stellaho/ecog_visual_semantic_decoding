import json
from pathlib import Path

import h5py

from sig_prep.downsample import downsample_signal
from sig_prep.filter import butter_bandpass_filter, highpass, notch_filter


def resolve_signal_path(project_root, patient_id, video_id):
    project_root = Path(project_root)
    signal_path = (
        project_root
        / "data"
        / "raw"
        / "ecog_1khz"
        / f"{patient_id}_mat"
        / f"{patient_id}_{video_id}.mat"
    )
    
    if not signal_path.is_file():
        raise FileNotFoundError(f"Signal file not found: {signal_path}")
    
    return signal_path


def load_and_preprocess_signal(
    patient_id,
    video_id,
    config,
    project_root,
    bandpass_filter=None,
):
    sig_path = resolve_signal_path(
        project_root,
        patient_id,
        video_id,
    )
    
    with h5py.File(sig_path, "r") as hf:
        sig = hf.get("/signals")[:]
        sig_frm_onsets = hf["trigger_scene_onsets"][:]
    
    trg_fs = config["ECoG"]["trg_fs_hz"]
    src_fs = config["ECoG"]["src_fs_hz"]
    lowcut = config["ECoG"]["lowcut_hz"]
    
    with (Path(project_root) / "config" / "sub_info.json").open() as file:
        sub_info = json.load(file)
    
    line_hz = sub_info[patient_id]["line_hz"]
    
    sig = notch_filter(sig, src_fs, line_hz, Q=50)
    sig = highpass(sig, src_fs=src_fs, lowcut=lowcut)
    sig = downsample_signal(sig, trg_fs=trg_fs, src_fs=src_fs)
    
    if bandpass_filter:
        low_hz, high_hz = bandpass_filter
        sig = butter_bandpass_filter(
            sig, 
            low_hz, 
            high_hz, 
            trg_fs, 
            order=5,
        )
    
    return sig, trg_fs, sig_frm_onsets, src_fs
