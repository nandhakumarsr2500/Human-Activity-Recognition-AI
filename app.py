"""
Human Activity Recognition (HAR) — Interactive AI Sensor Dashboard
Built with Streamlit, Scikit-Learn, TensorFlow, and Matplotlib.
"""

import os
import sys
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Setup paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import ACTIVITY_LABELS, ACTIVITY_COLORS
from src.preprocessing import preprocess_sensor_signals, segment_continuous_stream
from src.feature_extraction import extract_window_features, CHANNEL_NAMES
from simulate_sensor import SensorSimulator

# Page Configuration
st.set_page_config(
    page_title="Human Activity Recognition AI System",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px 32px;
        margin-bottom: 24px;
        color: white;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.2);
    }
    
    .metric-card {
        background: #FFFFFF;
        border-radius: 12px;
        padding: 18px 22px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    
    .activity-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.95rem;
        letter-spacing: 0.5px;
        color: white;
    }
    
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_all_models():
    """Load trained models and scaler."""
    models_dir = os.path.join(PROJECT_ROOT, "models")
    artifacts = {}
    try:
        artifacts["scaler"] = joblib.load(os.path.join(models_dir, "scaler.joblib"))
        artifacts["feature_names"] = joblib.load(os.path.join(models_dir, "feature_names.joblib"))
        artifacts["Random Forest"] = joblib.load(os.path.join(models_dir, "random_forest_model.joblib"))
        artifacts["Support Vector Machine"] = joblib.load(os.path.join(models_dir, "support_vector_machine_model.joblib"))
        artifacts["Decision Tree"] = joblib.load(os.path.join(models_dir, "decision_tree_model.joblib"))
        artifacts["Gradient Boosting"] = joblib.load(os.path.join(models_dir, "gradient_boosting_model.joblib"))
    except Exception as e:
        st.error(f"Error loading models: {e}. Please ensure models are trained.")
    return artifacts


models_cache = load_all_models()


# Sidebar Navigation
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/smartwatch.png", width=64)
    st.title("HAR System Studio")
    st.caption("AI-Powered Smartphone & Wearable Motion Analytics")
    st.divider()

    selected_page = st.radio(
        "Navigation Menu",
        [
            "🎯 Real-Time Activity Predictor",
            "📈 Sensor EDA & Waveforms",
            "🧠 Model Benchmark & Metrics",
            "📂 Upload Custom Sensor Stream",
            "📋 Architecture & Methodology"
        ],
        index=0
    )
    
    st.divider()
    st.subheader("Selected Classifier")
    active_model_name = st.selectbox(
        "Model Engine",
        ["Random Forest", "Support Vector Machine", "Gradient Boosting", "Decision Tree"],
        index=0
    )
    st.info("💡 Sampling Rate: **50 Hz**\n\nWindow Size: **128 samples (2.56s)**\n\nChannels: **3-Axis Accel + 3-Axis Gyro**")


# Header Banner
st.markdown("""
<div class="main-header">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <h1 style="margin: 0; font-size: 1.9rem; font-weight: 800; letter-spacing: -0.5px;">
                Human Activity Recognition (HAR) AI Platform
            </h1>
            <p style="margin: 6px 0 0 0; color: #94A3B8; font-size: 1.0rem;">
                Tri-axial Accelerometer & Gyroscope Kinematic Feature Extraction & Machine Learning Classification
            </p>
        </div>
        <div style="text-align: right;">
            <span style="background: rgba(37, 99, 235, 0.2); border: 1px solid #3B82F6; color: #60A5FA; padding: 6px 14px; border-radius: 20px; font-weight: 600; font-size: 0.85rem;">
                ● System Ready
            </span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ---------------- PAGE 1: REAL-TIME ACTIVITY PREDICTOR ----------------
if selected_page == "🎯 Real-Time Activity Predictor":
    st.subheader("Interactive Motion Simulation & Prediction Studio")
    st.markdown("Test the trained AI model against simulated real-time 50 Hz motion data or choose an activity scenario.")

    col_sim1, col_sim2 = st.columns([1, 2])
    
    with col_sim1:
        st.markdown("### 1. Motion Scenario")
        scenario = st.selectbox(
            "Select Physical Activity to Simulate:",
            list(ACTIVITY_LABELS.values()),
            index=0
        )
        noise_level = st.slider("Environmental Sensor Noise", 0.0, 1.0, 0.15, step=0.05)
        
        test_btn = st.button("🚀 Generate & Predict Activity", type="primary", use_container_width=True)
        
    sim = SensorSimulator(sampling_rate_hz=50.0)
    
    # Generate test window
    win_data = sim.generate_window(scenario, window_size=128)
    if noise_level > 0.15:
        win_data += np.random.normal(0, (noise_level - 0.15) * 0.2, win_data.shape)

    # Feature extraction and prediction
    feat_vec, feat_names = extract_window_features(win_data, sampling_rate_hz=50.0)
    scaler = models_cache["scaler"]
    model = models_cache[active_model_name]
    
    feat_df = pd.DataFrame([feat_vec], columns=models_cache["feature_names"])
    feat_scaled = scaler.transform(feat_df)
    pred_class_id = int(model.predict(feat_scaled)[0])
    pred_activity = ACTIVITY_LABELS[pred_class_id]
    
    # Class probabilities if available
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(feat_scaled)[0]
    else:
        probs = np.zeros(len(ACTIVITY_LABELS))
        probs[pred_class_id - 1] = 1.0

    with col_sim2:
        st.markdown("### 2. AI Inference Output")
        color = ACTIVITY_COLORS.get(pred_activity, "#2563EB")
        
        c_res1, c_res2, c_res3 = st.columns(3)
        with c_res1:
            st.metric("Predicted Activity", pred_activity)
        with c_res2:
            st.metric("Model Confidence", f"{np.max(probs) * 100:.1f}%")
        with c_res3:
            st.metric("Ground Truth", scenario)
            
        # Match indicator
        if pred_activity == scenario:
            st.success("✅ **Accurate Classification:** Sensor signature correctly identified!")
        else:
            st.warning(f"⚠️ Ambiguity detected between {scenario} and {pred_activity}")

    st.divider()
    
    # Visualizations
    st.markdown("### 3. Sensor Waveforms (Time-Series Window = 2.56s)")
    t_axis = np.linspace(0, 2.56, 128)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4), dpi=150)
    
    # Accelerometer Plot
    ax1.plot(t_axis, win_data[:, 0], label="Acc X (Lateral)", color="#EF4444", lw=1.6)
    ax1.plot(t_axis, win_data[:, 1], label="Acc Y (Vertical)", color="#10B981", lw=1.6)
    ax1.plot(t_axis, win_data[:, 2], label="Acc Z (Forward)", color="#3B82F6", lw=1.6)
    ax1.set_title("Tri-Axial Accelerometer (g)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Time (seconds)", fontsize=9)
    ax1.set_ylabel("Acceleration (g)", fontsize=9)
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(loc="upper right", fontsize=8)
    
    # Gyroscope Plot
    ax2.plot(t_axis, win_data[:, 3], label="Gyro X (Roll)", color="#F59E0B", lw=1.6)
    ax2.plot(t_axis, win_data[:, 4], label="Gyro Y (Pitch)", color="#8B5CF6", lw=1.6)
    ax2.plot(t_axis, win_data[:, 5], label="Gyro Z (Yaw)", color="#06B6D4", lw=1.6)
    ax2.set_title("Tri-Axial Gyroscope (rad/s)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Time (seconds)", fontsize=9)
    ax2.set_ylabel("Angular Velocity (rad/s)", fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(loc="upper right", fontsize=8)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Probability distribution bar
    st.markdown("### 4. Class Probability Distribution")
    prob_df = pd.DataFrame({
        "Activity": [ACTIVITY_LABELS[i] for i in sorted(ACTIVITY_LABELS.keys())],
        "Probability (%)": probs * 100.0
    })
    st.bar_chart(prob_df.set_index("Activity"))


# ---------------- PAGE 2: SENSOR EDA & WAVEFORMS ----------------
elif selected_page == "📈 Sensor EDA & Waveforms":
    st.subheader("Exploratory Data Analysis (EDA) of Human Motion")
    st.markdown("Explore biomechanical kinematic signatures across static and dynamic activities.")

    eda_act = st.selectbox("Inspect Biomechanical Signature for Activity:", list(ACTIVITY_LABELS.values()))
    
    sim = SensorSimulator(sampling_rate_hz=50.0)
    eda_win = sim.generate_window(eda_act, window_size=256)
    t_axis = np.linspace(0, 5.12, 256)
    
    col_eda1, col_eda2 = st.columns(2)
    
    with col_eda1:
        st.markdown("#### Tri-Axial Accelerometer Waveform")
        fig1, ax1 = plt.subplots(figsize=(7, 4), dpi=150)
        ax1.plot(t_axis, eda_win[:, 0], label="Acc X", color="#EF4444", lw=1.5)
        ax1.plot(t_axis, eda_win[:, 1], label="Acc Y", color="#10B981", lw=1.5)
        ax1.plot(t_axis, eda_win[:, 2], label="Acc Z", color="#3B82F6", lw=1.5)
        ax1.set_title(f"Acceleration — {eda_act}", fontsize=11, fontweight="bold")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("g (9.81 m/s²)")
        ax1.legend()
        ax1.grid(True, linestyle="--", alpha=0.4)
        st.pyplot(fig1)

    with col_eda2:
        st.markdown("#### Tri-Axial Gyroscope Waveform")
        fig2, ax2 = plt.subplots(figsize=(7, 4), dpi=150)
        ax2.plot(t_axis, eda_win[:, 3], label="Gyro X", color="#F59E0B", lw=1.5)
        ax2.plot(t_axis, eda_win[:, 4], label="Gyro Y", color="#8B5CF6", lw=1.5)
        ax2.plot(t_axis, eda_win[:, 5], label="Gyro Z", color="#06B6D4", lw=1.5)
        ax2.set_title(f"Angular Velocity — {eda_act}", fontsize=11, fontweight="bold")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("rad/s")
        ax2.legend()
        ax2.grid(True, linestyle="--", alpha=0.4)
        st.pyplot(fig2)

    st.divider()
    st.markdown("#### Kinematic Insights")
    if eda_act in ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS"]:
        st.info("🚶 **Dynamic Activity:** Characterized by strong periodicity, high variance, significant angular rotation, and step harmonic peaks around 1.5 - 2.2 Hz.")
    else:
        st.info("🪑 **Static Posture:** Near-zero gyro variance. Accelerometer magnitude remains ~1.0g, with gravity projection determining exact posture (Y-axis: Standing, Z-axis: Laying).")


# ---------------- PAGE 3: MODEL BENCHMARK & METRICS ----------------
elif selected_page == "🧠 Model Benchmark & Metrics":
    st.subheader("Model Evaluation & Benchmark Leaderboard")
    st.markdown("Comprehensive performance metrics evaluated on unseen sensor test data.")
    
    # Load metrics json if available
    metrics_path = os.path.join(PROJECT_ROOT, "reports", "evaluation_metrics.json")
    if os.path.exists(metrics_path):
        import json
        with open(metrics_path, "r") as f:
            metrics_data = json.load(f)
            
        summary_rows = []
        for name, m in metrics_data.items():
            summary_rows.append({
                "Algorithm": name,
                "Accuracy": f"{m['accuracy'] * 100:.2f}%",
                "Precision (Macro)": f"{m['precision_macro'] * 100:.2f}%",
                "Recall (Macro)": f"{m['recall_macro'] * 100:.2f}%",
                "F1-Score (Macro)": f"{m['f1_macro'] * 100:.2f}%"
            })
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)
    
    st.divider()
    
    col_cm, col_fi = st.columns(2)
    fig_dir = os.path.join(PROJECT_ROOT, "reports", "figures")
    
    with col_cm:
        st.markdown("#### Confusion Matrix (Random Forest)")
        cm_path = os.path.join(fig_dir, "confusion_matrix_random_forest.png")
        if os.path.exists(cm_path):
            st.image(cm_path, use_container_width=True)
            
    with col_fi:
        st.markdown("#### Top Discriminative Features")
        fi_path = os.path.join(fig_dir, "top_feature_importance.png")
        if os.path.exists(fi_path):
            st.image(fi_path, use_container_width=True)

    st.markdown("#### Comparative Model Performance")
    bar_path = os.path.join(fig_dir, "model_comparison_bar.png")
    if os.path.exists(bar_path):
        st.image(bar_path, use_container_width=True)


# ---------------- PAGE 4: UPLOAD CUSTOM SENSOR STREAM ----------------
elif selected_page == "📂 Upload Custom Sensor Stream":
    st.subheader("Upload & Classify Custom Sensor Stream")
    st.markdown("Upload continuous CSV recordings from smartphone logging apps (Physics Toolbox, Phyphox, etc.).")
    
    sample_csv_path = os.path.join(PROJECT_ROOT, "data", "raw", "sample_realtime_stream.csv")
    if os.path.exists(sample_csv_path):
        with open(sample_csv_path, "rb") as f:
            st.download_button(
                "📥 Download Sample Sensor CSV Recording",
                f,
                file_name="sample_sensor_stream.csv",
                mime="text/csv"
            )

    uploaded_file = st.file_uploader("Upload CSV containing acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z:", type=["csv"])
    
    target_stream_df = None
    if uploaded_file is not None:
        target_stream_df = pd.read_csv(uploaded_file)
    elif os.path.exists(sample_csv_path):
        st.caption("ℹ️ Showing demonstration with pre-recorded smartphone sensor data:")
        target_stream_df = pd.read_csv(sample_csv_path)

    if target_stream_df is not None:
        st.dataframe(target_stream_df.head(10), use_container_width=True)
        
        # Segment and predict
        if st.button("⚡ Run Sliding Window Activity Recognition"):
            with st.spinner("Processing continuous stream with 50% overlap sliding windows..."):
                X_wins, y_true_wins = segment_continuous_stream(
                    target_stream_df,
                    channel_cols=["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"],
                    label_col="activity_id" if "activity_id" in target_stream_df.columns else None,
                    window_size=128,
                    overlap_ratio=0.5
                )
                
                # Preprocess
                X_clean = preprocess_sensor_signals(X_wins)
                
                # Extract features & predict
                from src.feature_extraction import extract_features_dataset
                feats_df, _ = extract_features_dataset(X_clean)
                scaled_feats = models_cache["scaler"].transform(feats_df)
                preds = models_cache[active_model_name].predict(scaled_feats)
                
                pred_names = [ACTIVITY_LABELS[p] for p in preds]
                
                timeline_df = pd.DataFrame({
                    "Window Index": range(len(preds)),
                    "Start Time (s)": [round(i * 1.28, 2) for i in range(len(preds))],
                    "End Time (s)": [round(i * 1.28 + 2.56, 2) for i in range(len(preds))],
                    "Predicted Activity": pred_names
                })
                
                if y_true_wins is not None:
                    timeline_df["Ground Truth"] = [ACTIVITY_LABELS[int(y)] for y in y_true_wins]
                    accuracy = np.mean(preds == y_true_wins) * 100.0
                    st.success(f"Stream Recognition Accuracy: **{accuracy:.2f}%**")
                    
                st.dataframe(timeline_df, use_container_width=True)


# ---------------- PAGE 5: ARCHITECTURE & METHODOLOGY ----------------
elif selected_page == "📋 Architecture & Methodology":
    st.subheader("System Architecture & Implementation Pipeline")
    st.markdown(r"""
    ### 🔬 Human Activity Recognition (HAR) AI Framework
    The system follows the standard sensor processing methodology:
    
    1. **Data Acquisition:**
       - Tri-axial Accelerometer ($a_x, a_y, a_z$) in gravitational acceleration units ($g \approx 9.81 \, \text{m/s}^2$).
       - Tri-axial Gyroscope ($\omega_x, \omega_y, \omega_z$) in angular velocity ($\text{rad/s}$).
       - Continuous sampling at **50 Hz** (Nyquist frequency 25 Hz).
    
    2. **Signal Preprocessing & Noise Filtering:**
       - **Median Filter (kernel=3):** Rejects spike anomalies and measurement glitches.
       - **3rd-Order Butterworth Low-Pass Filter (cutoff 20 Hz):** Attenuates high-frequency noise from body vibration and electronics.
       - **Gravity Separation (0.3 Hz cutoff):** Decouples static gravity orientation from dynamic human acceleration.
       - **Sliding Window:** Window size $W = 128$ samples (2.56 seconds) with 50% overlap ($H = 64$ samples).
    
    3. **Kinematic & Frequency Feature Extraction:**
       - **Time-Domain Statistics:** Mean, Standard Deviation, Variance, Min, Max, Range, Median, MAD, IQR, Skewness, Kurtosis.
       - **Frequency-Domain:** Fast Fourier Transform (FFT), Spectral Energy, Dominant Frequency, Spectral Entropy.
       - **Kinematic Composite Features:** Signal Magnitude Area (SMA), Signal Vector Magnitude (SVM), Inter-Axis Pearson Correlation coefficients.
    
    4. **Machine Learning Classifiers:**
       - **Random Forest:** Ensemble of decorrelated decision trees with Gini impurity splitting.
       - **Support Vector Machine (SVM):** Radial Basis Function (RBF) kernel mapping non-linear decision boundaries.
       - **Gradient Boosting & Decision Trees:** Sequential gradient-boosted decision trees.
       - **1D-CNN (Deep Learning):** End-to-end multi-channel temporal convolutions with batch normalization and global average pooling.
    """)

st.caption("Capstone Project 6: Human Activity Recognition Using Sensor Data | Antigravity AI")
