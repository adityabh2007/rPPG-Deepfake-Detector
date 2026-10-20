#Turn cleaned ROI pulse signals into a fixed-length feature vector.

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from numpy.typing import NDArray
from scipy.signal import welch

if __package__:
    from src.filters import DEFAULT_HIGH_HZ, DEFAULT_LOW_HZ
else:
    from filters import DEFAULT_HIGH_HZ, DEFAULT_LOW_HZ

ROI_NAMES = ("forehead", "left_cheek", "right_cheek")
ROI_FEATURES = (
    "dominant_frequency_hz",
    "estimated_bpm",
    "spectral_snr_db",
    "peak_power_ratio",
    "spectral_entropy",
)
ROI_PAIRS = (
    ("forehead", "left_cheek"),
    ("forehead", "right_cheek"),
    ("left_cheek", "right_cheek"),
)


def _clean_signal(signal: NDArray[np.float64] | None) -> NDArray[np.float64]:
    if signal is None:
        return np.zeros(0, dtype=np.float64)
    values = np.asarray(signal, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("Each pulse signal must be one-dimensional.")
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)


def _spectrum_features(
    signal: NDArray[np.float64], fps: float
) -> tuple[float, float, float, float, float]:
    if len(signal) < 4 or np.std(signal) < 1e-10 or fps <= 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    upper_hz = min(DEFAULT_HIGH_HZ, fps / 2.0)
    if upper_hz <= DEFAULT_LOW_HZ:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    frequencies, power = welch(
        signal,
        fs=fps,
        nperseg=min(len(signal), max(512, int(round(fps * 8)))),
    )
    in_band = (frequencies >= DEFAULT_LOW_HZ) & (frequencies <= upper_hz)
    band_frequencies = frequencies[in_band]
    band_power = power[in_band]
    if len(band_power) == 0 or not np.any(band_power > 0):
        return 0.0, 0.0, 0.0, 0.0, 0.0

    peak_index = int(np.argmax(band_power))
    peak_frequency = float(band_frequencies[peak_index])
    total_power = float(np.sum(band_power))
    if total_power <= 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    resolution = float(np.mean(np.diff(frequencies))) if len(frequencies) > 1 else 0
    neighborhood_hz = max(0.1, 1.5 * resolution)
    pulse_bins = np.abs(band_frequencies - peak_frequency) <= neighborhood_hz
    harmonic = peak_frequency * 2.0
    if harmonic <= upper_hz:
        pulse_bins |= np.abs(band_frequencies - harmonic) <= neighborhood_hz

    pulse_power = float(np.sum(band_power[pulse_bins]))
    noise_power = max(total_power - pulse_power, 1e-20)
    snr_db = 10.0 * np.log10(max(pulse_power, 1e-20) / noise_power)
    peak_ratio = float(band_power[peak_index] / total_power)

    probabilities = band_power / total_power
    nonzero_probabilities = probabilities[probabilities > 0]
    if len(probabilities) <= 1:
        entropy = 0.0
    else:
        raw_entropy = -float(np.sum(nonzero_probabilities * np.log(nonzero_probabilities)))
        entropy = raw_entropy / np.log(len(probabilities))

    return (
        peak_frequency,
        peak_frequency * 60.0,
        float(snr_db),
        peak_ratio,
        float(entropy),
    )


def compute_features(
    pulse_signals: Mapping[str, NDArray[np.float64]], fps: float
) -> tuple[NDArray[np.float64], list[str]]:
    if fps <= 0 or not np.isfinite(fps):
        raise ValueError("fps must be a positive, finite number.")

    signals = {
        name: _clean_signal(pulse_signals.get(name)) for name in ROI_NAMES
    }
    roi_measurements = {
        name: _spectrum_features(signals[name], fps) for name in ROI_NAMES
    }

    feature_names: list[str] = []
    values: list[float] = []
    for name in ROI_NAMES:
        feature_names.extend(f"{name}_{feature}" for feature in ROI_FEATURES)
        values.extend(roi_measurements[name])

    frequencies = {
        name: roi_measurements[name][0] for name in ROI_NAMES
    }
    for first, second in ROI_PAIRS:
        first_signal = signals[first]
        second_signal = signals[second]
        common_length = min(len(first_signal), len(second_signal))
        correlation = 0.0
        if common_length >= 2:
            first_aligned = first_signal[:common_length]
            second_aligned = second_signal[:common_length]
            if np.std(first_aligned) >= 1e-10 and np.std(second_aligned) >= 1e-10:
                correlation_value = np.corrcoef(first_aligned, second_aligned)[0, 1]
                if np.isfinite(correlation_value):
                    correlation = float(correlation_value)

        feature_names.append(f"{first}_{second}_correlation")
        values.append(correlation)
        feature_names.append(f"{first}_{second}_frequency_difference_hz")
        values.append(abs(frequencies[first] - frequencies[second]))

    return np.asarray(values, dtype=np.float64), feature_names


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    sample_rate = 30.0
    time = np.arange(30 * 10) / sample_rate
    synthetic_signals = {
        name: np.sin(2 * np.pi * 1.2 * time + phase)
        + 0.15 * rng.normal(size=len(time))
        for name, phase in zip(ROI_NAMES, (0.0, 0.1, -0.1))
    }
    test_features, test_names = compute_features(synthetic_signals, sample_rate)
    bpm_index = test_names.index("forehead_estimated_bpm")
    recovered_bpm = test_features[bpm_index]
    assert abs(recovered_bpm - 72) <= 3, f"Recovered {recovered_bpm:.1f} BPM"
    print(f"Feature self-test passed: recovered {recovered_bpm:.1f} BPM.")
