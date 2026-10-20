#Command-line entry point for training and using the rPPG detector.

from __future__ import annotations

import argparse
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
MIN_MEDIAN_SNR_DB = 3.0


def _create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Explore pulse signals in face videos to estimate real or fake."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("train", help="build the dataset and train/evaluate the model")
    predict_parser = commands.add_parser("predict", help="classify one video")
    predict_parser.add_argument("video", type=Path, help="path to a video file")
    plot_parser = commands.add_parser("plot", help="plot pulse signals from one video")
    plot_parser.add_argument("video", type=Path, help="path to a video file")
    return parser


def _plot_video(video_path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.signal import welch

    from src.extract_signal import extract_rgb_traces, pos_algorithm
    from src.features import compute_features
    from src.filters import bandpass, detrend

    traces, fps = extract_rgb_traces(video_path)
    pulse_signals = {
        name: bandpass(detrend(pos_algorithm(rgb, fps)), fps)
        for name, rgb in traces.items()
    }
    features, feature_names = compute_features(pulse_signals, fps)

    roi_names = ("forehead", "left_cheek", "right_cheek")
    figure, axes = plt.subplots(3, 2, figsize=(12, 10))
    time = np.arange(len(pulse_signals["forehead"])) / fps
    for row, roi_name in enumerate(roi_names):
        signal = pulse_signals[roi_name]
        axes[row, 0].plot(time, signal, linewidth=0.8)
        axes[row, 0].set(
            title=f"{roi_name.replace('_', ' ').title()} pulse",
            xlabel="Time (seconds)",
            ylabel="Filtered signal",
        )

        frequencies, power = welch(
            signal,
            fs=fps,
            nperseg=min(len(signal), max(512, int(round(fps * 8)))),
        )
        show_band = (frequencies >= 0.0) & (frequencies <= min(4.0, fps / 2.0))
        axes[row, 1].plot(frequencies[show_band], power[show_band])
        frequency_name = f"{roi_name}_dominant_frequency_hz"
        dominant_frequency = features[feature_names.index(frequency_name)]
        bpm_name = f"{roi_name}_estimated_bpm"
        bpm = features[feature_names.index(bpm_name)]
        if dominant_frequency > 0:
            axes[row, 1].axvline(
                dominant_frequency,
                color="red",
                linestyle="--",
                label=f"Peak: {bpm:.0f} BPM",
            )
            axes[row, 1].legend()
        axes[row, 1].set(
            title=f"{roi_name.replace('_', ' ').title()} power spectrum",
            xlabel="Frequency (Hz)",
            ylabel="Power",
        )

    figure.suptitle(f"rPPG signals: {video_path.name}")
    figure.tight_layout()
    results_dir = PROJECT_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / f"{video_path.stem}_rppg.png"
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def main() -> int:
    args = _create_parser().parse_args()
    try:
        if args.command == "train":
            from src.classifier import build_dataset, train_and_evaluate

            X, y, video_names = build_dataset(
                PROJECT_DIR / "data" / "real",
                PROJECT_DIR / "data" / "fake",
            )
            if len(y) < 10:
                print(
                    f"Only {len(video_names)} usable videos were found. "
                    "Add at least 5 usable videos to each class folder."
                )
                return 1
            train_and_evaluate(X, y)
            return 0

        video_path = args.video
        if not video_path.is_file():
            print(f"Error: video file not found: {video_path}")
            return 1

        if args.command == "predict":
            from src.classifier import extract_video_features, predict_from_features

            features, feature_names = extract_video_features(video_path)
            snr_indices = [
                feature_names.index(f"{roi}_spectral_snr_db")
                for roi in ("forehead", "left_cheek", "right_cheek")
            ]
            median_snr = float(sorted(features[snr_indices])[1])
            if median_snr < MIN_MEDIAN_SNR_DB:
                print(
                    f"uncertain: signal too weak "
                    f"(median ROI SNR {median_snr:.1f} dB; "
                    f"need at least {MIN_MEDIAN_SNR_DB:.1f} dB)"
                )
                return 0

            label, probability = predict_from_features(features)
            print(f"{label} (confidence: {probability:.1%})")
            return 0

        output_path = _plot_video(video_path)
        print(f"Saved pulse plot to {output_path}")
        return 0
    except ModuleNotFoundError as error:
        print(
            f"Error: a required package is missing ({error.name}). "
            "Install the project dependencies with 'pip install -r requirements.txt'."
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        print(f"Error: {error}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
