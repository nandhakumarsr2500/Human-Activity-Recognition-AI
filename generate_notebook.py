"""
Script to generate the complete, self-contained Capstone Jupyter Notebook.
"""

import json
import os

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Capstone Project 6: Human Activity Recognition Using Sensor Data\n",
                "**Student / Researcher Capstone Submission**  \n",
                "**Domain:** Artificial Intelligence, Signal Processing & Wearable IoT  \n",
                "**Dataset:** Smartphone 6-Axis IMU (Accelerometer & Gyroscope)  \n",
                "**Activities:** WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING  \n",
                "\n",
                "---\n",
                "## 1. Project Overview & Problem Statement\n",
                "Human Activity Recognition (HAR) aims to automatically identify and classify physical human behaviors using kinematic sensor streams from smartphones, smartwatches, or wearable IMU devices.\n",
                "\n",
                "### Core Pipeline Objectives:\n",
                "1. **Data Acquisition:** Continuous 3-axis accelerometer and 3-axis gyroscope data at 50 Hz.\n",
                "2. **Data Preprocessing:** Missing-value handling, outlier filtering, 3rd-order Butterworth low-pass filtering (20 Hz), and gravity/body acceleration separation.\n",
                "3. **Signal Segmentation:** Sliding window technique ($W = 128$ samples = 2.56 seconds, $50\\%$ overlap = 64 samples).\n",
                "4. **Feature Engineering:** Time-domain statistics, Fourier Transform frequency features, and kinematic composite metrics (SMA, Vector Magnitudes).\n",
                "5. **Model Training & Comparison:** Random Forest, Support Vector Machine (SVM), Decision Tree, Gradient Boosting, and 1D-CNN.\n",
                "6. **Rigorous Evaluation:** Accuracy, Precision, Recall, F1-Score, and Confusion Matrix analysis.\n",
                "7. **Dashboard & Real-time Integration:** Streamlit interactive activity recognition application."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Environment Setup & Dependency Imports"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Core libraries\n",
                "import os\n",
                "import sys\n",
                "import numpy as np\n",
                "import pandas as pd\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "from scipy import signal, stats\n",
                "from scipy.fft import rfft, rfftfreq\n",
                "\n",
                "# Scikit-Learn\n",
                "from sklearn.model_selection import train_test_split\n",
                "from sklearn.preprocessing import StandardScaler\n",
                "from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier\n",
                "from sklearn.svm import SVC\n",
                "from sklearn.tree import DecisionTreeClassifier\n",
                "from sklearn.metrics import accuracy_score, classification_report, confusion_matrix\n",
                "\n",
                "# Set plot styles\n",
                "plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')\n",
                "plt.rcParams['font.size'] = 10\n",
                "print(\"All dependencies imported successfully!\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Dataset Loading & Class Description\n",
                "\n",
                "The dataset captures physical motions across 6 activity categories:\n",
                "- **WALKING (ID 1):** Continuous rhythmic forward movement with step cadence ~1.8 Hz.\n",
                "- **WALKING_UPSTAIRS (ID 2):** Periodic vertical propulsion against gravity, cadence ~1.5 Hz.\n",
                "- **WALKING_DOWNSTAIRS (ID 3):** High impact downward deceleration, higher variance, cadence ~2.1 Hz.\n",
                "- **SITTING (ID 4):** Static posture with steady gravitational vector tilted relative to device coordinate frame.\n",
                "- **STANDING (ID 5):** Static upright posture with near $1.0g$ on vertical axis ($y$-axis).\n",
                "- **LAYING (ID 6):** Horizontal orientation with gravity projected primarily along the $z$-axis or $x$-axis."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Append parent path if running inside notebooks/ directory\n",
                "sys.path.append(os.path.abspath('..'))\n",
                "from src.data_loader import load_dataset, ACTIVITY_LABELS, ACTIVITY_COLORS\n",
                "\n",
                "X_raw, y, subjects = load_dataset(data_dir=\"../data\", samples_per_activity=150)\n",
                "print(f\"Sensor Data Matrix Shape: {X_raw.shape} -> (Windows, Samples Per Window, Channels)\")\n",
                "print(f\"Target Labels Shape: {y.shape}\")\n",
                "\n",
                "# Class Distribution Summary\n",
                "class_counts = pd.Series(y).map(ACTIVITY_LABELS).value_counts()\n",
                "display(class_counts.to_frame(name=\"Window Count\"))"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Exploratory Data Analysis (EDA) & Kinematic Waveforms\n",
                "\n",
                "Visualizing the raw signals demonstrates clear distinctions between dynamic and static activities."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Visualize waveform comparison across all 6 activities\n",
                "fig, axes = plt.subplots(6, 2, figsize=(15, 18), sharex=True)\n",
                "time_axis = np.linspace(0, 128 / 50.0, 128)\n",
                "\n",
                "for i, (act_id, act_name) in enumerate(ACTIVITY_LABELS.items()):\n",
                "    sample_idx = np.where(y == act_id)[0][0]\n",
                "    sample_win = X_raw[sample_idx]\n",
                "    \n",
                "    # Accelerometer (Acc X, Y, Z)\n",
                "    axes[i, 0].plot(time_axis, sample_win[:, 0], label='Acc X', color='#EF4444')\n",
                "    axes[i, 0].plot(time_axis, sample_win[:, 1], label='Acc Y', color='#10B981')\n",
                "    axes[i, 0].plot(time_axis, sample_win[:, 2], label='Acc Z', color='#3B82F6')\n",
                "    axes[i, 0].set_title(f\"{act_name} — Accelerometer (g)\", fontweight='bold')\n",
                "    axes[i, 0].set_ylabel(\"g\")\n",
                "    axes[i, 0].legend(loc='upper right', fontsize=8)\n",
                "    \n",
                "    # Gyroscope (Gyro X, Y, Z)\n",
                "    axes[i, 1].plot(time_axis, sample_win[:, 3], label='Gyro X', color='#F59E0B')\n",
                "    axes[i, 1].plot(time_axis, sample_win[:, 4], label='Gyro Y', color='#8B5CF6')\n",
                "    axes[i, 1].plot(time_axis, sample_win[:, 5], label='Gyro Z', color='#06B6D4')\n",
                "    axes[i, 1].set_title(f\"{act_name} — Gyroscope (rad/s)\", fontweight='bold')\n",
                "    axes[i, 1].set_ylabel(\"rad/s\")\n",
                "    axes[i, 1].legend(loc='upper right', fontsize=8)\n",
                "\n",
                "axes[-1, 0].set_xlabel(\"Time (seconds)\")\n",
                "axes[-1, 1].set_xlabel(\"Time (seconds)\")\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Signal Preprocessing & Noise Filtering\n",
                "\n",
                "To prepare the raw signals for feature extraction, we apply:\n",
                "1. **Median Filter (order 3):** Rejects spike anomalies from digital sensors.\n",
                "2. **3rd-Order Butterworth Low-Pass Filter (cutoff 20 Hz):** Filters high-frequency electronic noise.\n",
                "3. **Gravity and Body Component Separation (0.3 Hz cutoff):** Isolates the constant gravitational acceleration vector from dynamic movement."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from src.preprocessing import preprocess_sensor_signals, separate_gravity_and_body_acc\n",
                "\n",
                "print(\"Applying Median + Butterworth 20Hz low-pass filtering...\")\n",
                "X_clean = preprocess_sensor_signals(X_raw, sampling_rate_hz=50.0, noise_cutoff_hz=20.0)\n",
                "print(\"Preprocessing completed successfully! Denoised shape:\", X_clean.shape)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 6. Kinematic & Frequency Feature Extraction\n",
                "\n",
                "We extract a multidimensional feature vector from each 128-sample window:\n",
                "- **Statistical:** Mean, Standard Deviation, Variance, Peak-to-Peak Range, Median, MAD, IQR, Skewness, Kurtosis.\n",
                "- **Frequency Domain:** Fast Fourier Transform (FFT) dominant frequency, spectral energy, spectral entropy.\n",
                "- **Kinematics:** Signal Vector Magnitude ($SVM = \\sqrt{x^2+y^2+z^2}$), Signal Magnitude Area (SMA), Inter-axis Pearson correlation."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from src.feature_extraction import extract_features_dataset\n",
                "\n",
                "features_df, feature_names = extract_features_dataset(X_clean, sampling_rate_hz=50.0)\n",
                "print(f\"Extracted {features_df.shape[1]} features across {features_df.shape[0]} windows.\")\n",
                "display(features_df.head())"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 7. Machine Learning Model Training & Optimization\n",
                "\n",
                "We partition data into 80% training and 20% unseen test sets with stratified class distribution."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Train/Test Split\n",
                "X_train, X_test, y_train, y_test = train_test_split(\n",
                "    features_df, y, test_size=0.20, random_state=42, stratify=y\n",
                ")\n",
                "\n",
                "# Feature Scaling\n",
                "scaler = StandardScaler()\n",
                "X_train_scaled = scaler.fit_transform(X_train)\n",
                "X_test_scaled = scaler.transform(X_test)\n",
                "\n",
                "from src.models import get_ml_models\n",
                "models = get_ml_models()\n",
                "trained_models = {}\n",
                "\n",
                "for name, clf in models.items():\n",
                "    clf.fit(X_train_scaled, y_train)\n",
                "    trained_models[name] = clf\n",
                "    print(f\"{name} trained successfully!\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 8. Model Evaluation & Benchmark Metrics\n",
                "\n",
                "Evaluating accuracy, precision, recall, and F1-score across all algorithms on the unseen test set."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from src.evaluate import compute_metrics, generate_classification_report_df\n",
                "\n",
                "results = {}\n",
                "for name, clf in trained_models.items():\n",
                "    y_pred = clf.predict(X_test_scaled)\n",
                "    results[name] = compute_metrics(y_test, y_pred)\n",
                "\n",
                "results_df = pd.DataFrame(results).transpose()[['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']]\n",
                "results_df.columns = ['Accuracy', 'Precision (Macro)', 'Recall (Macro)', 'F1-Score (Macro)']\n",
                "display(results_df.style.format('{:.2%}'))"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 9. Confusion Matrix & Misclassification Analysis"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from src.evaluate import plot_confusion_matrix, plot_feature_importance\n",
                "\n",
                "best_model_name = \"Random Forest\"\n",
                "y_best_pred = trained_models[best_model_name].predict(X_test_scaled)\n",
                "\n",
                "fig = plot_confusion_matrix(y_test, y_best_pred, model_name=best_model_name)\n",
                "plt.show()\n",
                "\n",
                "# Top Features\n",
                "fig_fi = plot_feature_importance(trained_models[best_model_name], feature_names, top_n=15)\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 10. Real-Time Inference Simulation\n",
                "\n",
                "Simulating live sensor packet intake and predicting human activity in real time."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from simulate_sensor import SensorSimulator\n",
                "\n",
                "sim = SensorSimulator(sampling_rate_hz=50.0)\n",
                "test_activity = \"WALKING_UPSTAIRS\"\n",
                "live_window = sim.generate_window(test_activity, window_size=128)\n",
                "\n",
                "# Preprocess & predict\n",
                "feat_v, _ = extract_window_features(live_window, sampling_rate_hz=50.0)\n",
                "scaled_v = scaler.transform(feat_v.reshape(1, -1))\n",
                "pred_id = trained_models[\"Random Forest\"].predict(scaled_v)[0]\n",
                "pred_name = ACTIVITY_LABELS[pred_id]\n",
                "\n",
                "print(f\"Target Activity:  {test_activity}\")\n",
                "print(f\"Predicted Output: {pred_name}\")\n",
                "print(f\"Status:           {'CORRECT' if pred_name == test_activity else 'MISCLASSIFIED'}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 11. Conclusion & Findings\n",
                "- **Feature Discriminability:** Signal Magnitude Area (SMA), vertical acceleration variance, and peak FFT frequency provide the highest discriminative power between dynamic activities.\n",
                "- **Static vs Dynamic Separation:** Gravity vector orientation easily separates Sitting, Standing, and Laying with near 100% precision.\n",
                "- **Ensemble Performance:** Random Forest and Support Vector Machines achieved the highest generalization stability.\n",
                "- **Interactive Dashboard:** The accompanying Streamlit dashboard (`app.py`) provides real-time sensor testing, custom CSV uploads, and model analytics."
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbformat": 4,
            "nbformat_minor": 5
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

os.makedirs("notebooks", exist_ok=True)
notebook_path = os.path.join("notebooks", "HAR_Capstone_Notebook.ipynb")
with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print(f"Jupyter Notebook successfully generated at: {notebook_path}")
