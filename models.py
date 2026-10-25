"""
Model Architecture and Training Module for Human Activity Recognition
Implements:
1. Classical Machine Learning:
   - Random Forest Classifier
   - Support Vector Classifier (SVM - RBF)
   - Decision Tree Classifier
   - Gradient Boosting Classifier
2. Deep Learning:
   - 1D Convolutional Neural Network (1D-CNN) for raw sequential sensor windows
"""

import os
import joblib
import numpy as np
from typing import Dict, Any, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler


def get_ml_models() -> Dict[str, Any]:
    """
    Instantiate classical ML classifiers with optimized hyperparameters.
    """
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=16,
            min_samples_split=3,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),
        "Support Vector Machine": SVC(
            C=10.0,
            kernel="rbf",
            gamma="scale",
            probability=True,
            class_weight="balanced",
            random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=12,
            min_samples_split=4,
            class_weight="balanced",
            random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
    }
    return models


def build_1d_cnn(
    input_shape: Tuple[int, int] = (128, 6),
    num_classes: int = 6
):
    """
    Build a 1D Convolutional Neural Network for end-to-end classification
    directly from raw multi-channel IMU windows (without manual feature engineering).
    """
    try:
        import tensorflow as tf
        from tensorflow.keras import layers, models
    except ImportError:
        print("TensorFlow is not available. Skipping 1D-CNN build.")
        return None

    model = models.Sequential([
        layers.Input(shape=input_shape),
        # Conv Block 1
        layers.Conv1D(filters=64, kernel_size=5, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.2),

        # Conv Block 2
        layers.Conv1D(filters=128, kernel_size=5, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.25),

        # Conv Block 3
        layers.Conv1D(filters=128, kernel_size=3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling1D(),

        # Dense Classifier
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def save_artifact(obj: Any, filepath: str) -> None:
    """Save trained model or transformer object using joblib."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(obj, filepath)
    print(f"Saved artifact to {filepath}")


def load_artifact(filepath: str) -> Any:
    """Load saved model or transformer object."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Artifact not found at {filepath}")
    return joblib.load(filepath)
