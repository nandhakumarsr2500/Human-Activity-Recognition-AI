"""
Feature Extraction Module for Human Activity Recognition
Extracts time-domain, frequency-domain, and multi-axis kinematic features
from windowed 3-axis accelerometer and gyroscope sensor signals.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.fft import rfft, rfftfreq
from typing import List, Tuple


CHANNEL_NAMES = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]


def compute_spectral_entropy(psd: np.ndarray) -> float:
    """Compute normalized Shannon spectral entropy of power spectral density."""
    psd_norm = psd / (np.sum(psd) + 1e-12)
    psd_norm = psd_norm[psd_norm > 0]
    if len(psd_norm) == 0:
        return 0.0
    entropy = -np.sum(psd_norm * np.log2(psd_norm))
    norm_factor = np.log2(len(psd)) if len(psd) > 1 else 1.0
    return float(entropy / norm_factor)


def extract_window_features(
    window: np.ndarray,
    sampling_rate_hz: float = 50.0
) -> Tuple[np.ndarray, List[str]]:
    """
    Extract a comprehensive feature vector from a single (window_size, 6) matrix.
    Channels: [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
    """
    feature_vals = []
    feature_names = []
    
    # 1. Per-channel statistical and frequency features
    for c_idx, col_name in enumerate(CHANNEL_NAMES):
        sig = window[:, c_idx]
        
        # Time-domain
        mean_val = float(np.mean(sig))
        std_val = float(np.std(sig))
        var_val = float(np.var(sig))
        max_val = float(np.max(sig))
        min_val = float(np.min(sig))
        peak_to_peak = max_val - min_val
        median_val = float(np.median(sig))
        mad_val = float(np.median(np.abs(sig - median_val)))
        q75, q25 = np.percentile(sig, [75, 25])
        iqr_val = float(q75 - q25)
        energy_val = float(np.mean(sig ** 2))
        skew_val = float(stats.skew(sig))
        kurt_val = float(stats.kurtosis(sig))
        
        # Frequency-domain (FFT)
        N = len(sig)
        fft_vals = np.abs(rfft(sig))
        fft_freqs = rfftfreq(N, 1.0 / sampling_rate_hz)
        
        # Ignore DC component (0 Hz) for dominant frequency
        if len(fft_vals) > 1:
            dom_idx = np.argmax(fft_vals[1:]) + 1
            dom_freq = float(fft_freqs[dom_idx])
        else:
            dom_freq = 0.0
            
        spectral_energy = float(np.sum(fft_vals ** 2) / N)
        spec_entropy = compute_spectral_entropy(fft_vals ** 2)
        
        feats = [
            (f"{col_name}_mean", mean_val),
            (f"{col_name}_std", std_val),
            (f"{col_name}_var", var_val),
            (f"{col_name}_max", max_val),
            (f"{col_name}_min", min_val),
            (f"{col_name}_ptp", peak_to_peak),
            (f"{col_name}_median", median_val),
            (f"{col_name}_mad", mad_val),
            (f"{col_name}_iqr", iqr_val),
            (f"{col_name}_energy", energy_val),
            (f"{col_name}_skew", skew_val),
            (f"{col_name}_kurt", kurt_val),
            (f"{col_name}_dom_freq", dom_freq),
            (f"{col_name}_spec_energy", spectral_energy),
            (f"{col_name}_spec_entropy", spec_entropy),
        ]
        
        for k, v in feats:
            feature_names.append(k)
            feature_vals.append(v)
            
    # 2. Composite Cross-Axis Features (Kinematic Magnitudes & SMA)
    acc = window[:, 0:3]
    gyro = window[:, 3:6]
    
    # Signal Vector Magnitude (Euclidean Norm)
    acc_mag = np.linalg.norm(acc, axis=1)
    gyro_mag = np.linalg.norm(gyro, axis=1)
    
    comp_feats = [
        ("acc_mag_mean", float(np.mean(acc_mag))),
        ("acc_mag_std", float(np.std(acc_mag))),
        ("acc_mag_max", float(np.max(acc_mag))),
        ("acc_mag_min", float(np.min(acc_mag))),
        ("acc_mag_energy", float(np.mean(acc_mag ** 2))),
        ("gyro_mag_mean", float(np.mean(gyro_mag))),
        ("gyro_mag_std", float(np.std(gyro_mag))),
        ("gyro_mag_max", float(np.max(gyro_mag))),
        ("gyro_mag_energy", float(np.mean(gyro_mag ** 2))),
        # Signal Magnitude Area (SMA)
        ("acc_sma", float(np.mean(np.sum(np.abs(acc), axis=1)))),
        ("gyro_sma", float(np.mean(np.sum(np.abs(gyro), axis=1)))),
        # Inter-axis Pearson Correlations
        ("acc_corr_xy", float(np.corrcoef(acc[:, 0], acc[:, 1])[0, 1]) if np.std(acc[:, 0]) > 1e-6 and np.std(acc[:, 1]) > 1e-6 else 0.0),
        ("acc_corr_xz", float(np.corrcoef(acc[:, 0], acc[:, 2])[0, 1]) if np.std(acc[:, 0]) > 1e-6 and np.std(acc[:, 2]) > 1e-6 else 0.0),
        ("acc_corr_yz", float(np.corrcoef(acc[:, 1], acc[:, 2])[0, 1]) if np.std(acc[:, 1]) > 1e-6 and np.std(acc[:, 2]) > 1e-6 else 0.0),
        ("gyro_corr_xy", float(np.corrcoef(gyro[:, 0], gyro[:, 1])[0, 1]) if np.std(gyro[:, 0]) > 1e-6 and np.std(gyro[:, 1]) > 1e-6 else 0.0),
        ("gyro_corr_xz", float(np.corrcoef(gyro[:, 0], gyro[:, 2])[0, 1]) if np.std(gyro[:, 0]) > 1e-6 and np.std(gyro[:, 2]) > 1e-6 else 0.0),
        ("gyro_corr_yz", float(np.corrcoef(gyro[:, 1], gyro[:, 2])[0, 1]) if np.std(gyro[:, 1]) > 1e-6 and np.std(gyro[:, 2]) > 1e-6 else 0.0),
    ]
    
    for k, v in comp_feats:
        feature_names.append(k)
        # Handle potential NaNs from zero division
        feature_vals.append(0.0 if np.isnan(v) or np.isinf(v) else v)
        
    return np.array(feature_vals, dtype=np.float32), feature_names


def extract_features_dataset(
    X_windows: np.ndarray,
    sampling_rate_hz: float = 50.0
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Extract features for all windows in dataset.
    Returns:
    - feature_df: pandas DataFrame containing extracted features
    - feature_names: list of feature names
    """
    N = X_windows.shape[0]
    first_vec, feature_names = extract_window_features(X_windows[0], sampling_rate_hz)
    
    D = len(first_vec)
    features_mat = np.zeros((N, D), dtype=np.float32)
    features_mat[0] = first_vec
    
    for i in range(1, N):
        vec, _ = extract_window_features(X_windows[i], sampling_rate_hz)
        features_mat[i] = vec
        
    df = pd.DataFrame(features_mat, columns=feature_names)
    # Fill any edge case nan/inf
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df, feature_names
