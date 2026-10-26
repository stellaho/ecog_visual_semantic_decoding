from pathlib import Path


def default_save_dir(mode):
    if mode == "labeled":
        return "./data/visual_decoding/processed/labeled"
    return "./data/visual_decoding/processed/unlabeled"


def build_output_dir(
    save_dir,
    patient_id,
    window_ms,
    bandpass_filter=None,
):
    folder_name = f"{window_ms}ms"
    
    if bandpass_filter is not None:
        low, high = bandpass_filter
        folder_name += f"_bp{int(low)}-{int(high)}"

    return Path(save_dir) / patient_id / folder_name
