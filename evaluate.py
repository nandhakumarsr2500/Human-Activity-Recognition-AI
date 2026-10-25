"""
Model Evaluation and Metrics Module for Human Activity Recognition
Generates:
1. Accuracy, Precision, Recall, F1-Score (macro and weighted).
2. Confusion Matrix (counts & normalized).
3. Class-wise classification report dataframe.
4. Exportable visualizations for reports and dashboards.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
from typing import Dict, Any, List, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from src.data_loader import ACTIVITY_LABELS


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute comprehensive classification metrics.
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def generate_classification_report_df(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Generate structured pandas DataFrame of classification report per activity.
    """
    if target_names is None:
        target_names = [ACTIVITY_LABELS[i] for i in sorted(ACTIVITY_LABELS.keys())]
        
    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        output_dict=True,
        zero_division=0
    )
    df = pd.DataFrame(report_dict).transpose()
    return df


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Random Forest",
    save_path: Optional[str] = None,
    normalize: bool = True
) -> plt.Figure:
    """
    Plot and save confusion matrix with annotations.
    """
    classes = [ACTIVITY_LABELS[i] for i in sorted(ACTIVITY_LABELS.keys())]
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm_display = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        fmt = ".2%"
    else:
        cm_display = cm
        fmt = "d"

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    if HAS_SEABORN:
        sns.heatmap(
            cm_display,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            xticklabels=classes,
            yticklabels=classes,
            cbar=True,
            ax=ax,
            linewidths=0.5,
            linecolor="#E2E8F0"
        )
    else:
        im = ax.imshow(cm_display, interpolation='nearest', cmap=plt.cm.Blues)
        fig.colorbar(im, ax=ax)
        tick_marks = np.arange(len(classes))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(classes, rotation=45, ha="right")
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(classes)
        for i in range(len(classes)):
            for j in range(len(classes)):
                val_str = f"{cm_display[i, j]:.2%}" if normalize else f"{cm_display[i, j]}"
                ax.text(j, i, val_str, ha="center", va="center",
                        color="white" if cm_display[i, j] > (cm_display.max() / 2.) else "black")
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Activity", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Activity", fontsize=11, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved confusion matrix plot: {save_path}")
        
    return fig


def plot_model_comparison(
    metrics_dict: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot side-by-side bar chart comparing all models across key metrics.
    """
    df = pd.DataFrame(metrics_dict).transpose()
    display_metrics = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    col_labels = ["Accuracy", "Precision (Macro)", "Recall (Macro)", "F1-Score (Macro)"]
    
    plot_df = df[display_metrics] * 100.0
    plot_df.columns = col_labels

    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    plot_df.plot(kind="bar", ax=ax, width=0.8, colormap="viridis")
    ax.set_title("Machine Learning Models Performance Comparison", fontsize=14, fontweight="bold", pad=12)
    ax.set_ylabel("Score (%)", fontsize=11, fontweight="bold")
    ax.set_ylim(70, 102)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.xticks(rotation=0, fontweight="bold")
    plt.legend(frameon=True, facecolor="#F8FAFC", edgecolor="#CBD5E1")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved model comparison plot: {save_path}")

    return fig


def plot_feature_importance(
    model: Any,
    feature_names: List[str],
    top_n: int = 15,
    save_path: Optional[str] = None
) -> Optional[plt.Figure]:
    """
    Plot top N most important features from tree-based models.
    """
    if not hasattr(model, "feature_importances_"):
        return None

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    
    top_feats = [feature_names[i] for i in indices]
    top_scores = importances[indices]

    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    if HAS_SEABORN:
        sns.barplot(x=top_scores, y=top_feats, palette="crest", ax=ax)
    else:
        y_pos = np.arange(len(top_feats))
        ax.barh(y_pos, top_scores, color="#0284C7", align="center")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_feats)
        ax.invert_yaxis()
    ax.set_title(f"Top {top_n} Most Discriminative HAR Features", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Gini Feature Importance", fontsize=11, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved feature importance plot: {save_path}")

    return fig
