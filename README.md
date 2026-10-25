# Human Activity Recognition (HAR) Using Sensor Data
### AI Capstone Project 6 — Smartphone & Wearable Motion Recognition

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![TensorFlow](https://img.shields.io/badge/Deep%20Learning-TensorFlow-yellow.svg)](https://tensorflow.org/)

---

## 📖 Project Overview
This project delivers a complete AI-based Human Activity Recognition (HAR) system that uses 6-axis inertial sensor data (3-axis Accelerometer + 3-axis Gyroscope) sampled at **50 Hz** from smartphones or wearable devices to automatically recognize six physical human activities:
- 🚶 **WALKING**
- 🧗 **WALKING_UPSTAIRS**
- 🏃 **WALKING_DOWNSTAIRS**
- 🪑 **SITTING**
- 🧍 **STANDING**
- 🛌 **LAYING**

---

## 🗂️ Repository Structure

```
vac human project 1/
├── app.py                     # Interactive Streamlit AI Dashboard
├── simulate_sensor.py         # Real-time 50 Hz IMU sensor streaming engine
├── generate_notebook.py       # Notebook generation utility
├── requirements.txt           # Project dependencies
├── PROJECT_REPORT.md          # Comprehensive Capstone Academic Report
├── README.md                  # System Documentation & Quickstart
├── data/
│   └── raw/
│       ├── har_dataset.npz               # Cached HAR dataset
│       └── sample_realtime_stream.csv    # Continuous stream export for mobile testing
├── models/
│   ├── random_forest_model.joblib        # Trained Random Forest classifier
│   ├── support_vector_machine_model.joblib # Trained Support Vector Machine (RBF)
│   ├── decision_tree_model.joblib        # Trained Decision Tree classifier
│   ├── gradient_boosting_model.joblib    # Trained Gradient Boosting classifier
│   ├── cnn_har_model.keras               # Trained 1D-CNN Deep Learning model
│   ├── scaler.joblib                     # Feature StandardScaler
│   └── feature_names.joblib              # 77 Extracted feature names
├── notebooks/
│   └── HAR_Capstone_Notebook.ipynb       # Complete step-by-step Jupyter / Colab Notebook
├── reports/
│   ├── evaluation_metrics.json           # Quantitative benchmark scores (JSON)
│   └── figures/                          # Exported high-resolution plots
│       ├── confusion_matrix_random_forest.png
│       ├── confusion_matrix_support_vector_machine.png
│       ├── confusion_matrix_decision_tree.png
│       ├── confusion_matrix_gradient_boosting.png
│       ├── confusion_matrix_1d_cnn.png
│       ├── model_comparison_bar.png
│       └── top_feature_importance.png
└── src/
    ├── __init__.py
    ├── data_loader.py                    # Dataset loading & synthetic IMU generator
    ├── preprocessing.py                  # Denoising (Median + Butterworth LPF), windowing
    ├── feature_extraction.py             # Time-frequency & kinematic feature engineering
    ├── models.py                         # ML and 1D-CNN architecture definitions
    ├── train.py                          # Master training and evaluation pipeline
    └── evaluate.py                       # Evaluation metrics and visualization generation
```

---

## 🚀 Quickstart Guide

### 1. Installation
All required dependencies are listed in `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 2. Launch the Streamlit Interactive Dashboard
Run the dashboard with:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.  
The dashboard features:
- **Real-Time Activity Predictor:** Generate synthetic motion packets, adjust noise, and observe live predictions with confidence scores.
- **Sensor EDA & Waveforms:** Inspect 3-axis accelerometer and gyroscope time-series waveforms.
- **Model Benchmark & Metrics:** View confusion matrices, feature importance graphs, and metric leaderboards.
- **Upload Custom Sensor Stream:** Ingest CSV files recorded from mobile apps (*Phyphox*, *Physics Toolbox*).
- **Architecture & Methodology:** In-depth technical breakdown of signal processing and feature engineering.

### 3. Run the Jupyter / Google Colab Notebook
Open and execute the step-by-step notebook:
```bash
jupyter notebook notebooks/HAR_Capstone_Notebook.ipynb
```

### 4. Re-Train Models from Scratch
To run the end-to-end data processing, feature extraction, and model training:
```bash
python src/train.py
```

---

## 📊 Benchmark Results

Evaluated on an independent, unseen 20% test dataset (Stratified 6-class split):

| Algorithm | Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) |
| :--- | :---: | :---: | :---: | :---: |
| **Random Forest** | **100.00%** | **100.00%** | **100.00%** | **100.00%** |
| **Support Vector Machine (SVM)** | **100.00%** | **100.00%** | **100.00%** | **100.00%** |
| **Gradient Boosting** | **100.00%** | **100.00%** | **100.00%** | **100.00%** |
| **Decision Tree** | **100.00%** | **100.00%** | **100.00%** | **100.00%** |
| **1D-CNN (Raw Windows)** | 66.67% | 50.00% | 66.67% | 55.56% |

---

## 🔬 Signal Processing & Feature Engineering

1. **Preprocessing:**
   - **Median Filter ($k=3$):** Rejects spike glitches.
   - **3rd-Order Butterworth Low-Pass Filter (20 Hz):** Eliminates high-frequency sensor noise.
   - **Gravity & Body Separation (0.3 Hz cutoff):** Separates static gravity from dynamic body acceleration.
   - **Sliding Window:** 128 samples ($2.56 \, \text{s}$) with 50% overlap ($1.28 \, \text{s}$ step).
2. **77 Extracted Features:**
   - **Statistical:** Mean, Standard Deviation, Variance, Min, Max, Range, Median, MAD, IQR, Skewness, Kurtosis.
   - **Spectral:** FFT Dominant Frequency, Spectral Energy, Spectral Entropy.
   - **Kinematic:** Signal Magnitude Area (SMA), Signal Vector Magnitude (SVM / Euclidean norm), Inter-Axis Pearson Correlation.

---

## 📱 Mobile App Real-Time Collection Guide
You can log real sensor data using any smartphone sensor app:
1. **Phyphox (Free, Open Source):**
   - Open Phyphox $\rightarrow$ Select **Acceleration with g** and **Gyroscope**.
   - Record motion for 10-30 seconds.
   - Export as CSV and upload directly into the **Upload Custom Sensor Stream** tab in `app.py`.
2. **Physics Toolbox Sensor Suite:**
   - Record Multi-Sensor (Accelerometer + Gyroscope) at 50 Hz.
   - Export CSV and test live classification.
