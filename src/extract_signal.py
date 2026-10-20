# Extract skin-color traces and pulse signals from face videos.

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from src.face_detection import FaceDetector

ROI_NAMES = ("forehead", "left_cheek", "right_cheek")
MAX_MISSING_FRACTION = 0.30
POS_WINDOW_SECONDS = 1.6


def _interpolate_missing(values: NDArray[np.float64]) -> NDArray[np.float64]:
    filled = values.copy()
    positions = np.arange(len(values))

    for channel in range(values.shape[1]):
        valid = np.isfinite(values[:, channel])
        if not np.any(valid):
            raise ValueError("No usable face-color samples were found.")
        filled[:, channel] = np.interp(
            positions, positions[valid], values[valid, channel]
        )

    return filled


def extract_rgb_traces(
    video_path: str | Path,
) -> tuple[dict[str, NDArray[np.float64]], float]:
    path = Path(video_path)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    detector: FaceDetector | None = None
    samples: dict[str, list[NDArray[np.float64]]] = {
        name: [] for name in ROI_NAMES
    }
    missing_frames = 0
    frame_count = 0

    try:
        detector = FaceDetector()
        while True:
            ok, frame_bgr = capture.read()
            if not ok:
                break

            frame_count += 1
            masks = detector.get_roi_masks(frame_bgr)
            if masks is None:
                missing_frames += 1
                for name in ROI_NAMES:
                    samples[name].append(np.full(3, np.nan))
                continue

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            for name in ROI_NAMES:
                mask = masks.get(name)
                if mask is None or not np.any(mask):
                    samples[name].append(np.full(3, np.nan))
                    continue
                pixels = frame_rgb[np.asarray(mask) > 0]
                samples[name].append(pixels.mean(axis=0, dtype=np.float64))
    finally:
        capture.release()
        if detector is not None:
            detector.close()

    if frame_count == 0:
        raise ValueError(f"The video contains no readable frames: {path}")
    if fps <= 0 or not np.isfinite(fps):
        raise ValueError(f"The video does not report a valid frame rate: {path}")
    missing_fraction = missing_frames / frame_count
    if missing_fraction > MAX_MISSING_FRACTION:
        raise ValueError(
            f"Face was missing in {missing_fraction:.0%} of frames "
            f"(maximum allowed: {MAX_MISSING_FRACTION:.0%})."
        )

    traces = {
        name: _interpolate_missing(np.asarray(values, dtype=np.float64))
        for name, values in samples.items()
    }
    return traces, fps


def pos_algorithm(
    rgb_trace: NDArray[np.float64], fps: float
) -> NDArray[np.float64]:
    rgb = np.asarray(rgb_trace, dtype=np.float64)
    if rgb.ndim != 2 or rgb.shape[1] != 3:
        raise ValueError("rgb_trace must have shape (frames, 3).")
    if len(rgb) < 2:
        raise ValueError("At least two RGB samples are needed to estimate a pulse.")
    if fps <= 0 or not np.isfinite(fps):
        raise ValueError("fps must be a positive, finite number.")
    if not np.all(np.isfinite(rgb)):
        raise ValueError("rgb_trace contains missing or non-finite values.")

    window_length = max(2, int(round(POS_WINDOW_SECONDS * fps)))
    window_length = min(window_length, len(rgb))
    step = max(1, window_length // 2)
    last_start = len(rgb) - window_length
    starts = list(range(0, last_start + 1, step))
    if starts[-1] != last_start:
        starts.append(last_start)

    pulse_sum = np.zeros(len(rgb), dtype=np.float64)
    overlap_count = np.zeros(len(rgb), dtype=np.float64)

    for start in starts:
        end = start + window_length
        window = rgb[start:end]
        channel_means = window.mean(axis=0)
        normalized = window / np.maximum(channel_means, 1e-8) - 1.0
        red, green, blue = normalized.T

        signal_one = green - red
        signal_two = green + red - 2.0 * blue
        spread_two = np.std(signal_two)
        if spread_two < 1e-8:
            continue
        projection_scale = np.std(signal_one) / spread_two
        window_pulse = signal_one + projection_scale * signal_two

        pulse_sum[start:end] += window_pulse
        overlap_count[start:end] += 1.0

    return np.divide(
        pulse_sum,
        overlap_count,
        out=np.zeros_like(pulse_sum),
        where=overlap_count > 0,
    )


def green_channel_algorithm(rgb_trace: NDArray[np.float64],) -> NDArray[np.float64]:
    rgb = np.asarray(rgb_trace, dtype=np.float64)
    if rgb.ndim != 2 or rgb.shape[1] != 3:
        raise ValueError("rgb_trace must have shape (frames, 3).")
    if len(rgb) == 0 or not np.all(np.isfinite(rgb)):
        raise ValueError("rgb_trace must contain finite samples.")
    green = rgb[:, 1]
    mean_green = float(np.mean(green))
    if abs(mean_green) < 1e-8:
        return green - mean_green
    return green / mean_green - 1.0
