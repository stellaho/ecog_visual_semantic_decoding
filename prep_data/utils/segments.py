import numpy as np

from prep_data.utils.constants import FOLDER_RANGES, LABEL_TO_INT


def _onset_to_target_index(onset, src_fs, trg_fs):
    if hasattr(onset, "item"):
        onset = onset.item()

    # Convert onset from the original sampling rate to the resampled signal.
    onset_sec = float(onset) / src_fs
    return int(onset_sec * trg_fs)


def extract_segments_labeled(
    sig,
    sig_frm_onsets,
    src_fs,
    trg_fs,
    labels_df,
    video_id,
    window_ms=500,
):
    recording_labels = labels_df[
        labels_df["folder_name"] == video_id
    ]

    if recording_labels.empty:
        return [], [], []
    
    start_id, _ = FOLDER_RANGES[video_id]
    n_samples = int(window_ms / 1000 * trg_fs)
    
    segments = []
    labels = []
    ids = []
    
    for _, row in recording_labels.iterrows():
        global_id = row["ID"]
        label = row["label"]
        local_idx = global_id - start_id
        
        if local_idx < 0 or local_idx >= len(sig_frm_onsets):
            continue
        
        onset_idx = _onset_to_target_index(
            sig_frm_onsets[local_idx], 
            src_fs, 
            trg_fs,
        )
        end_idx = onset_idx + n_samples
        
        # Skip incomplete windows near the end of the recording
        if end_idx > sig.shape[1]:
            continue
        
        segment = sig[:, onset_idx:end_idx]
        if segment.shape[1] != n_samples:
            continue
        
        segments.append(segment)
        labels.append(LABEL_TO_INT[label])
        ids.append(global_id)
    
    return segments, labels, ids


def extract_segments_unlabeled(
    sig,
    sig_frm_onsets,
    src_fs,
    trg_fs,
    window_ms=500,
):
    n_samples = int(window_ms / 1000 * trg_fs)
    
    segments = []
    ids = []
    
    for idx, onset in enumerate(sig_frm_onsets):
        onset_idx = _onset_to_target_index(onset, src_fs, trg_fs)
        end_idx = onset_idx + n_samples
        
        # Ignore windows that extend beyond the available signal.
        if end_idx > sig.shape[1]:
            continue
        
        segment = sig[:, onset_idx:end_idx]
        if segment.shape[1] != n_samples:
            continue
        
        segments.append(segment)
        ids.append(idx)
    
    return segments, ids
