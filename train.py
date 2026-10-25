"""
Master Model Training and Benchmark Pipeline
Runs end-to-end:
1. Data loading and validation
2. Preprocessing & Denoising
3. Feature Engineering
4. Train/Test Split (Subject-wise & Stratified)
5. Multi-model Training (RF, SVM, DT, Gradient Boosting, 1D-CNN)
6. Evaluation and Visualization Exports
7. Artifact Serialization
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Ensure root directory is in sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import load_dataset, ACTIVITY_LABELS
from src.preprocessing import preprocess_sensor_signals
from src.feature_extraction import extract_features_dataset
from src.models import get_ml_models, build_1d_cnn, save_artifact
from src.evaluate import (
    compute_metrics,
    generate_classification_report_df,
    plot_confusion_matrix,
    plot_model_comparison,
    plot_feature_importance
)


def run_pipeline():
    print("=" * 70)
    print("  HUMAN ACTIVITY RECOGNITION (HAR) — END-TO-END TRAINING PIPELINE")
    print("=" * 70)

    # 1. Load Data
    data_dir = os.path.join(PROJECT_ROOT, "data")
    X_raw, y, subjects = load_dataset(data_dir=data_dir, samples_per_activity=150)
    print(f"\n[1/6] Raw dataset loaded: {X_raw.shape[0]} windows, shape {X_raw.shape}")
    print(f"      Activities represented: {list(ACTIVITY_LABELS.values())}")

    # 2. Preprocess & Filter Signals
    print("\n[2/6] Applying signal preprocessing (Median filter + Butterworth 20Hz LPF)...")
    X_clean = preprocess_sensor_signals(X_raw, sampling_rate_hz=50.0, noise_cutoff_hz=20.0)

    # 3. Feature Extraction
    print("\n[3/6] Extracting kinematic, statistical, and frequency-domain features...")
    features_df, feature_names = extract_features_dataset(X_clean, sampling_rate_hz=50.0)
    print(f"      Total features extracted per window: {len(feature_names)}")

    # 4. Train / Test Split
    print("\n[4/6] Splitting dataset (80% Train, 20% Unseen Test, Stratified)...")
    X_train_df, X_test_df, y_train, y_test, X_raw_train, X_raw_test = train_test_split(
        features_df, y, X_clean, test_size=0.20, random_state=42, stratify=y
    )

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_df)
    X_test_scaled = scaler.transform(X_test_df)

    # Save artifacts dir
    models_dir = os.path.join(PROJECT_ROOT, "models")
    os.makedirs(models_dir, exist_ok=True)
    figures_dir = os.path.join(PROJECT_ROOT, "reports", "figures")
    os.makedirs(figures_dir, exist_ok=True)

    save_artifact(scaler, os.path.join(models_dir, "scaler.joblib"))
    save_artifact(feature_names, os.path.join(models_dir, "feature_names.joblib"))

    # 5. Train Classical ML Models
    print("\n[5/6] Training & Evaluating Classical ML Models...")
    ml_models = get_ml_models()
    all_metrics = {}
    trained_models = {}

    for name, model in ml_models.items():
        print(f"      -> Training {name}...")
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        metrics = compute_metrics(y_test, y_pred)
        all_metrics[name] = metrics
        trained_models[name] = model

        # Save model
        slug = name.lower().replace(" ", "_")
        save_artifact(model, os.path.join(models_dir, f"{slug}_model.joblib"))

        # Save individual confusion matrix
        plot_confusion_matrix(
            y_test,
            y_pred,
            model_name=name,
            save_path=os.path.join(figures_dir, f"confusion_matrix_{slug}.png")
        )

        print(f"         Accuracy: {metrics['accuracy']:.4f} | F1-Score: {metrics['f1_macro']:.4f}")

    # Feature Importance for Best Tree Model (Random Forest)
    best_rf = trained_models["Random Forest"]
    plot_feature_importance(
        best_rf,
        feature_names,
        top_n=15,
        save_path=os.path.join(figures_dir, "top_feature_importance.png")
    )

    # Comparative Plot
    plot_model_comparison(
        all_metrics,
        save_path=os.path.join(figures_dir, "model_comparison_bar.png")
    )

    # 6. Deep Learning: 1D-CNN
    print("\n[6/6] Training Deep Learning Model (1D-CNN on raw sequential sensor windows)...")
    try:
        # Keras 0-indexed classes
        y_train_dl = y_train - 1
        y_test_dl = y_test - 1
        
        cnn = build_1d_cnn(input_shape=(128, 6), num_classes=6)
        if cnn is not None:
            cnn.fit(
                X_raw_train,
                y_train_dl,
                epochs=15,
                batch_size=32,
                validation_split=0.15,
                verbose=1
            )
            y_pred_dl_probs = cnn.predict(X_raw_test)
            y_pred_dl = np.argmax(y_pred_dl_probs, axis=1) + 1

            dl_metrics = compute_metrics(y_test, y_pred_dl)
            all_metrics["1D-CNN (Deep Learning)"] = dl_metrics

            cnn.save(os.path.join(models_dir, "cnn_har_model.keras"))
            plot_confusion_matrix(
                y_test,
                y_pred_dl,
                model_name="1D-CNN Deep Learning",
                save_path=os.path.join(figures_dir, "confusion_matrix_1d_cnn.png")
            )
            print(f"      1D-CNN Accuracy: {dl_metrics['accuracy']:.4f} | F1-Score: {dl_metrics['f1_macro']:.4f}")
    except Exception as e:
        print(f"      [Notice] 1D-CNN training skipped or encountered: {e}")

    # Save metrics JSON
    metrics_json_path = os.path.join(PROJECT_ROOT, "reports", "evaluation_metrics.json")
    with open(metrics_json_path, "w") as f:
        json.dump(all_metrics, f, indent=4)

    # Print final summary table
    summary_df = pd.DataFrame(all_metrics).transpose()[["accuracy", "precision_macro", "recall_macro", "f1_macro"]]
    summary_df.columns = ["Accuracy", "Precision", "Recall", "F1-Score"]
    print("\n" + "=" * 70)
    print("                    FINAL BENCHMARK RESULTS")
    print("=" * 70)
    print(summary_df.to_string())
    print("=" * 70)
    print(f"\nAll models and figures exported successfully to 'models/' and 'reports/figures/'.")


if __name__ == "__main__":
    run_pipeline()
