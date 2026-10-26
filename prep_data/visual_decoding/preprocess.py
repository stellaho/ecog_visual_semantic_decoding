import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from prep_data.utils.io import (
    load_global_config,
    load_labeled_images,
    save_labeled_segments,
    save_unlabeled_segments,
)
from prep_data.utils.metadata import ensure_electrode_info
from prep_data.utils.paths import build_output_dir, default_save_dir
from prep_data.utils.segments import (
    extract_segments_labeled,
    extract_segments_unlabeled,
)
from prep_data.utils.signal import load_and_preprocess_signal


def process_recording(
    video_id,
    args,
    config,
    labels_df,
    save_dir,
):
    sig, trg_fs, sig_frm_onsets, src_fs = load_and_preprocess_signal(
        args.patient_id,
        video_id,
        config,
        PROJECT_ROOT,
        bandpass_filter=args.bandpass_filter,
    )
    
    if args.mode == "labeled":
        segments, labels, ids = extract_segments_labeled(
            sig,
            sig_frm_onsets,
            src_fs,
            trg_fs,
            labels_df,
            video_id,
            args.window_ms,
        )

        if not segments:
            return

        save_labeled_segments(
            save_dir,
            video_id,
            segments,
            labels,
            ids,
        )
    
    else:
        segments, ids = extract_segments_unlabeled(
            sig,
            sig_frm_onsets,
            src_fs,
            trg_fs,
            args.window_ms,
        )

        if not segments:
            return

        save_unlabeled_segments(
            save_dir,
            video_id,
            segments,
            ids,
        )
    

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess ECoG signals"
    )

    parser.add_argument(
        "--patient_id",
        default="E02",
        help="Patient ID",
    )
    parser.add_argument(
        "--mode",
        choices=["labeled", "unlabeled"],
        default="labeled",
        help="Preprocess labeled or unlabeled data",
    )
    parser.add_argument(
        "--folders",
        nargs="+",
        default=["Trn01", "Trn02", "Trn03", "Trn04", "Trn05", "Trn06"],
        help="Recordings to process",
    )
    parser.add_argument(
        "--window_ms",
        type=int,
        default=500,
        help="Post-onset window in milliseconds",
    )
    parser.add_argument(
        "--labels_csv",
        default="./data/visual_decoding/labels/image_labels_50.csv",
        help="Image labels for labeled preprocessing",
    )
    parser.add_argument(
        "--save_dir",
        default=None,
        help="Output directory",
    )
    parser.add_argument(
        "--bandpass_filter",
        type=float,
        nargs=2,
        default=None,
        help="Bandpass range, e.g. 80 150",
    )

    args = parser.parse_args()
    
    os.chdir(PROJECT_ROOT)
    
    if args.save_dir is None:
        args.save_dir = default_save_dir(args.mode)
        
    config = load_global_config(PROJECT_ROOT)
    ensure_electrode_info(args.patient_id, PROJECT_ROOT)
    
    labels_df = None
    if args.mode == "labeled":
        labels_df = load_labeled_images(args.labels_csv)
        
    save_dir = build_output_dir(
        args.save_dir,
        args.patient_id,
        args.window_ms,
        args.bandpass_filter,
    )
    save_dir.mkdir(parents=True, exist_ok=True)
    
    for video_id in args.folders:
        process_recording(
            video_id,
            args,
            config,
            labels_df,
            save_dir,
        )


if __name__ == "__main__":
    main()
