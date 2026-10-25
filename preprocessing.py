"""
Signal Preprocessing and Windowing Module for HAR
Handles:
1. Missing value interpolation and outlier rejection.
2. Signal denoising: Median filtering and Butterworth 3rd-order low-pass filter (20 Hz).
3. Gravity and Body acceleration component separation (0.3 Hz Butterworth filter).
4. Sliding window segmentation with configurable overlap.
5. Signal normalization.
"""

import numpy as np
import pandas as pd
from scipy import signal
from typing import Tuple, List, Optional


def handle_missing_values(data: np.ndarray) -> np.ndarray:
    """
    Handle missing values (NaN or Inf) using linear interpolation and forward/backward filling.
    Expects 2D (T, C) or 3D (N, T, C) arrays.
    """
    cleaned = np.copy(data)
    if not np.any(np.isnan(cleaned)) and not np.any(np.isinf(cleaned)):
        return cleaned

    cleaned[np.isinf(cleaned)] = np.nan
    
    if cleaned.ndim == 2:
        df = pd.DataFrame(cleaned)
        df = df.interpolate(method="linear", limit_direction="both").bfill().ffill()
        return df.to_numpy()
    elif cleaned.ndim == 3:
        for i in range(cleaned.shape[0]):
            df = pd.DataFrame(cleaned[i])
            df = df.interpolate(method="linear", limit_direction="both").bfill().ffill()
            cleaned[i] = df.to_numpy()
        return cleaned
    return cleaned


def apply_median_filter(data: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Apply median filter along time axis to remove transient spikes / sensor glitches.
    Works on 1D, 2D (T, C), or 3D (N, T, C) arrays.
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    if data.ndim == 1:
        return signal.medfilt(data, kernel_size=kernel_size)
    elif data.ndim == 2:
        filtered = np.zeros_like(data)
        for c in range(data.shape[1]):
            filtered[:, c] = signal.medfilt(data[:, c], kernel_size=kernel_size)
        return filtered
    elif data.ndim == 3:
        filtered = np.zeros_like(data)
        for i in range(data.shape[0]):
            for c in range(data.shape[2]):
                filtered[i, :, c] = signal.medfilt(data[i, :, c], kernel_size=kernel_size)
        return filtered
    return data


def butter_lowpass_filter(
    data: np.ndarray,
    cutoff_hz: float = 20.0,
    sampling_rate_hz: float = 50.0,
    order: int = 3
) -> np.ndarray:
    """
    Apply zero-phase Butterworth low-pass filter (filtfilt) to remove high-frequency noise.
    """
    nyquist = 0.5 * sampling_rate_hz
    normal_cutoff = min(cutoff_hz / nyquist, 0.99)
    b, a = signal.butter(order, normal_cutoff, btype="low", analog=False)

    if data.ndim == 1:
        return signal.filtfilt(b, a, data)
    elif data.ndim == 2:
        filtered = np.zeros_like(data)
        for c in range(data.shape[1]):
            filtered[:, c] = signal.filtfilt(b, a, data[:, c])
        return filtered
    elif data.ndim == 3:
        filtered = np.zeros_like(data)
        for i in range(data.shape[0]):
            for c in range(data.shape[2]):
                filtered[i, :, c] = signal.filtfilt(b, a, data[i, :, c])
        return filtered
    return data


def separate_gravity_and_body_acc(
    acc_data: np.ndarray,
    cutoff_hz: float = 0.3,
    sampling_rate_hz: float = 50.0,
    order: int = 3
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Separate gravitational component from body acceleration using a 0.3 Hz Butterworth filter.
    Acc_Total = Acc_Body + Acc_Gravity
    Returns:
    - body_acc: dynamic acceleration from user movement
    - gravity_acc: static gravitational constant
    """
    gravity_acc = butter_lowpass_filter(
        acc_data, cutoff_hz=cutoff_hz, sampling_rate_hz=sampling_rate_hz, order=order
    )
    body_acc = acc_data - gravity_acc
    return body_acc, gravity_acc


def preprocess_sensor_signals(
    X_raw: np.ndarray,
    sampling_rate_hz: float = 50.0,
    noise_cutoff_hz: float = 20.0
) -> np.ndarray:
    """
    Complete signal preprocessing pipeline:
    1. Handle missing values
    2. Median filtering for impulsive noise
    3. 3rd-order Butterworth low-pass filter at 20 Hz
    """
    cleaned = handle_missing_values(X_raw)
    med_filtered = apply_median_filter(cleaned, kernel_size=3)
    filtered = butter_lowpass_filter(
        med_filtered, cutoff_hz=noise_cutoff_hz, sampling_rate_hz=sampling_rate_hz, order=order if 'order' in locals() else 3
    )
    return filtered


def segment_continuous_stream(
    stream_df: pd.DataFrame,
    channel_cols: List[str] = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"],
    label_col: Optional[str] = "activity_id",
    window_size: int = 128,
    overlap_ratio: float = 0.5
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Sliding window segmentation on continuous smartphone IMU streams.

    Parameters:
    - stream_df: DataFrame containing sensor channels
    - channel_cols: list of column names for the 6 sensor channels
    - label_col: optional column name for activity ground truth
    - window_size: number of samples per window (e.g., 128)
    - overlap_ratio: overlap percentage (0.5 = 50% overlap, step=64)

    Returns:
    - X_windows: (N, window_size, len(channel_cols))
    - y_windows: (N,) or None
    """
    step_size = int(window_size * (1.0 - overlap_ratio))
    total_len = len(stream_df)
    
    if total_len < window_size:
        raise ValueError(f"Stream length ({total_len}) is shorter than window_size ({window_size})")

    sensor_values = stream_df[channel_cols].to_numpy()
    has_labels = label_col is not None and label_col in stream_df.columns
    label_values = stream_df[label_col].to_numpy() if has_labels else None

    windows = []
    labels = []

    for start_idx in range(0, total_len - window_size + 1, step_size):
        end_idx = start_idx + window_size
        windows.append(sensor_values[start_idx:end_idx, :])
        if has_labels:
            # Assign majority label in window
            win_labels = label_values[start_idx:end_idx]
            vals, counts = np.unique(win_labels, return_counts=True)
            labels.append(vals[np.argmax(counts)])

    X_windows = np.array(windows, dtype=np.float32)
    y_windows = np.array(labels, dtype=np.int32) if has_labels else None
    return X_windows, y_windows
