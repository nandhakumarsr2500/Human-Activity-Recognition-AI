"""
Real-Time Sensor Simulator and Streaming Utility
Generates live 50 Hz accelerometer and gyroscope packets mimicking smartphone motion.
Allows real-time testing, demonstration, and exporting of continuous IMU data.
"""

import time
import numpy as np
import pandas as pd
from typing import Generator, Dict, Any, Optional

from src.data_loader import ACTIVITY_LABELS


class SensorSimulator:
    """
    Simulates continuous 50 Hz smartphone 6-axis IMU readings.
    """
    def __init__(self, sampling_rate_hz: float = 50.0):
        self.sampling_rate_hz = sampling_rate_hz
        self.dt = 1.0 / sampling_rate_hz
        self.t = 0.0

    def get_sample(self, activity_name: str) -> Dict[str, float]:
        """
        Generate a single instantaneous IMU packet (acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z).
        """
        t = self.t
        self.t += self.dt

        if activity_name == "WALKING":
            freq = 1.8
            acc_x = 0.25 * np.sin(2 * np.pi * freq * t) + np.random.normal(0, 0.04)
            acc_y = 1.0 + 0.35 * np.cos(2 * np.pi * freq * t) + np.random.normal(0, 0.04)
            acc_z = 0.15 * np.sin(4 * np.pi * freq * t) + np.random.normal(0, 0.03)
            gyro_x = 0.8 * np.sin(2 * np.pi * freq * t + 0.3) + np.random.normal(0, 0.06)
            gyro_y = 0.4 * np.cos(2 * np.pi * freq * t) + np.random.normal(0, 0.05)
            gyro_z = 0.3 * np.sin(2 * np.pi * freq * t + 1.0) + np.random.normal(0, 0.05)

        elif activity_name == "WALKING_UPSTAIRS":
            freq = 1.5
            acc_x = 0.35 * np.sin(2 * np.pi * freq * t) + np.random.normal(0, 0.05)
            acc_y = 1.15 + 0.55 * np.sin(2 * np.pi * freq * t) + np.random.normal(0, 0.06)
            acc_z = -0.2 + 0.25 * np.cos(2 * np.pi * freq * t) + np.random.normal(0, 0.04)
            gyro_x = 1.1 * np.sin(2 * np.pi * freq * t) + np.random.normal(0, 0.08)
            gyro_y = 0.5 * np.cos(2 * np.pi * freq * t) + np.random.normal(0, 0.07)
            gyro_z = 0.4 * np.sin(2 * np.pi * freq * t) + np.random.normal(0, 0.07)

        elif activity_name == "WALKING_DOWNSTAIRS":
            freq = 2.1
            acc_x = 0.4 * np.sin(2 * np.pi * freq * t) + np.random.normal(0, 0.07)
            acc_y = 0.9 + 0.7 * np.sin(2 * np.pi * freq * t) + np.random.normal(0, 0.09)
            acc_z = 0.3 * np.cos(2 * np.pi * freq * t) + np.random.normal(0, 0.06)
            gyro_x = 1.3 * np.sin(2 * np.pi * freq * t + 0.5) + np.random.normal(0, 0.1)
            gyro_y = 0.6 * np.cos(2 * np.pi * freq * t) + np.random.normal(0, 0.08)
            gyro_z = 0.5 * np.sin(2 * np.pi * freq * t) + np.random.normal(0, 0.08)

        elif activity_name == "SITTING":
            acc_x = 0.12 + np.random.normal(0, 0.012)
            acc_y = 0.65 + np.random.normal(0, 0.012)
            acc_z = 0.72 + np.random.normal(0, 0.012)
            gyro_x = np.random.normal(0, 0.008)
            gyro_y = np.random.normal(0, 0.008)
            gyro_z = np.random.normal(0, 0.008)

        elif activity_name == "STANDING":
            acc_x = -0.05 + np.random.normal(0, 0.015)
            acc_y = 0.98 + np.random.normal(0, 0.015)
            acc_z = 0.10 + np.random.normal(0, 0.015)
            gyro_x = np.random.normal(0, 0.01)
            gyro_y = np.random.normal(0, 0.01)
            gyro_z = np.random.normal(0, 0.01)

        elif activity_name == "LAYING":
            acc_x = -0.15 + np.random.normal(0, 0.01)
            acc_y = 0.05 + np.random.normal(0, 0.01)
            acc_z = -0.98 + np.random.normal(0, 0.01)
            gyro_x = np.random.normal(0, 0.006)
            gyro_y = np.random.normal(0, 0.006)
            gyro_z = np.random.normal(0, 0.006)
        else:
            acc_x = np.random.normal(0, 0.02)
            acc_y = 1.0 + np.random.normal(0, 0.02)
            acc_z = np.random.normal(0, 0.02)
            gyro_x = np.random.normal(0, 0.01)
            gyro_y = np.random.normal(0, 0.01)
            gyro_z = np.random.normal(0, 0.01)

        return {
            "timestamp": round(self.t, 3),
            "acc_x": float(acc_x),
            "acc_y": float(acc_y),
            "acc_z": float(acc_z),
            "gyro_x": float(gyro_x),
            "gyro_y": float(gyro_y),
            "gyro_z": float(gyro_z),
            "ground_truth": activity_name
        }

    def generate_window(self, activity_name: str, window_size: int = 128) -> np.ndarray:
        """
        Generate a complete window matrix of shape (window_size, 6).
        """
        arr = np.zeros((window_size, 6), dtype=np.float32)
        for i in range(window_size):
            s = self.get_sample(activity_name)
            arr[i, 0] = s["acc_x"]
            arr[i, 1] = s["acc_y"]
            arr[i, 2] = s["acc_z"]
            arr[i, 3] = s["gyro_x"]
            arr[i, 4] = s["gyro_y"]
            arr[i, 5] = s["gyro_z"]
        return arr


def stream_generator(
    duration_seconds: float = 10.0,
    activity_sequence: Optional[list] = None
) -> Generator[Dict[str, Any], None, None]:
    """
    Generator that yields real-time 50 Hz IMU readings.
    """
    sim = SensorSimulator(sampling_rate_hz=50.0)
    if activity_sequence is None:
        activity_sequence = ["STANDING", "WALKING", "WALKING_UPSTAIRS", "SITTING"]
        
    num_steps = int(duration_seconds * 50.0)
    chunk = max(1, num_steps // len(activity_sequence))
    
    for i in range(num_steps):
        act_idx = min(i // chunk, len(activity_sequence) - 1)
        current_act = activity_sequence[act_idx]
        sample = sim.get_sample(current_act)
        yield sample


if __name__ == "__main__":
    sim = SensorSimulator()
    win = sim.generate_window("WALKING", 128)
    print("Generated real-time test window shape:", win.shape)
    print("Mean Accel:", np.mean(win[:, :3], axis=0))
    print("Mean Gyro:", np.mean(win[:, 3:], axis=0))
