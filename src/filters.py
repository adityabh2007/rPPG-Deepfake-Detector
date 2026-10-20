#Small signal-cleaning helpers for rPPG pulse traces.

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, detrend as scipy_detrend, filtfilt

FILTER_ORDER = 3
DEFAULT_LOW_HZ = 0.7
DEFAULT_HIGH_HZ = 4.0


def _as_finite_signal(signal: NDArray[np.float64]) -> NDArray[np.float64]:
    values = np.asarray(signal, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("signal must be one-dimensional.")
    if len(values) == 0:
        raise ValueError("signal must not be empty.")
    if not np.all(np.isfinite(values)):
        raise ValueError("signal must contain only finite values.")
    return values


def bandpass(
    signal: NDArray[np.float64],
    fps: float,
    low: float = DEFAULT_LOW_HZ,
    high: float = DEFAULT_HIGH_HZ,
) -> NDArray[np.float64]:
    values = _as_finite_signal(signal)
    if fps <= 0 or not np.isfinite(fps):
        raise ValueError("fps must be a positive, finite number.")
    if low <= 0 or high <= low:
        raise ValueError("Cutoffs must satisfy 0 < low < high.")
    if high >= fps / 2.0:
        raise ValueError(
            f"fps={fps:g} is too low: the high cutoff must be below "
            f"the Nyquist frequency ({fps / 2.0:g} Hz)."
        )

    numerator, denominator = butter(
        FILTER_ORDER, (low, high), btype="bandpass", fs=fps
    )
    minimum_length = 3 * max(len(numerator), len(denominator))
    if len(values) <= minimum_length:
        raise ValueError(
            f"Signal is too short for filtering; need more than "
            f"{minimum_length} samples."
        )
    return filtfilt(numerator, denominator, values)


def detrend(signal: NDArray[np.float64]) -> NDArray[np.float64]:
    values = _as_finite_signal(signal)
    if len(values) < 2:
        return np.zeros_like(values)
    return scipy_detrend(values, type="linear")


if __name__ == "__main__":
    from scipy.signal import welch

    rng = np.random.default_rng(42)
    sample_rate = 30.0
    time = np.arange(30 * 10) / sample_rate
    noisy_pulse = np.sin(2 * np.pi * 1.2 * time) + 0.25 * rng.normal(
        size=len(time)
    )
    cleaned = bandpass(detrend(noisy_pulse), sample_rate)
    frequencies, power = welch(cleaned, fs=sample_rate, nperseg=300)
    in_pulse_band = (frequencies >= DEFAULT_LOW_HZ) & (
        frequencies <= DEFAULT_HIGH_HZ
    )
    recovered_bpm = frequencies[in_pulse_band][
        np.argmax(power[in_pulse_band])
    ] * 60
    assert abs(recovered_bpm - 72) <= 3, f"Recovered {recovered_bpm:.1f} BPM"
    print(f"Filter self-test passed: recovered {recovered_bpm:.1f} BPM.")
