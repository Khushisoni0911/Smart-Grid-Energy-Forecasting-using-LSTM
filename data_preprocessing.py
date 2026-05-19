# ============================================================
# FILE 3: evaluate.py
# Full Evaluation: MAE, RMSE, R², MAPE, F1, Confusion Matrix
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    f1_score,
    classification_report,
    confusion_matrix
)
import joblib
import os


# ==============================================================
# HELPER: inverse-scale predictions back to original MW values
# ==============================================================
def inverse_scale(scaler, scaled_values, n_features):
    """
    scaler was fitted on (n_features) columns; target is col 0.
    We pad with zeros for the other columns to do inverse_transform.
    """
    dummy = np.zeros((len(scaled_values), n_features))
    dummy[:, 0] = scaled_values.flatten()
    return scaler.inverse_transform(dummy)[:, 0]


# ==============================================================
# REGRESSION METRICS
# ==============================================================
def regression_metrics(y_true_mw, y_pred_mw):
    """
    Compute and print regression accuracy metrics.

    Args:
        y_true_mw : actual MW values (numpy array)
        y_pred_mw : predicted MW values (numpy array)

    Returns:
        dict of metrics
    """
    mae  = mean_absolute_error(y_true_mw, y_pred_mw)
    rmse = np.sqrt(mean_squared_error(y_true_mw, y_pred_mw))
    r2   = r2_score(y_true_mw, y_pred_mw)

    # MAPE — skip zero actual values to avoid division by zero
    mask = y_true_mw != 0
    mape = np.mean(np.abs((y_true_mw[mask] - y_pred_mw[mask]) / y_true_mw[mask])) * 100

    print("\n" + "="*50)
    print("  REGRESSION METRICS (Original MW Scale)")
    print("="*50)
    print(f"  MAE   : {mae:>10.2f} MW")
    print(f"  RMSE  : {rmse:>10.2f} MW")
    print(f"  MAPE  : {mape:>10.2f} %")
    print(f"  R²    : {r2:>10.4f}  (1.0 = perfect)")
    print("="*50)

    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'R2': r2}


# ==============================================================
# CLASSIFICATION METRICS (binning continuous → discrete labels)
# ==============================================================
def bin_consumption(mw_values, n_bins=5):
    """
    Convert continuous MW values into n_bins discrete labels.
    Labels: 0 = Very Low … (n_bins-1) = Very High
    Uses equal-width bins based on the range of y_true.
    """
    min_val, max_val = mw_values.min(), mw_values.max()
    bin_edges = np.linspace(min_val, max_val, n_bins + 1)
    labels = np.digitize(mw_values, bin_edges[1:-1])   # 0-indexed labels
    return labels, bin_edges


def classification_metrics(y_true_mw, y_pred_mw, n_bins=5):
    """
    Convert regression outputs → bins and compute:
      - F1 score (macro, weighted)
      - Full classification report
      - Confusion matrix (as plot)

    Args:
        y_true_mw : actual MW values
        y_pred_mw : predicted MW values
        n_bins    : number of consumption levels (default 5)

    Returns:
        f1_macro, f1_weighted
    """
    # Bin using TRUE distribution so both share the same scale
    y_true_bins, bin_edges = bin_consumption(y_true_mw, n_bins)
    y_pred_bins, _         = bin_consumption(
        np.clip(y_pred_mw, y_true_mw.min(), y_true_mw.max()), n_bins
    )

    # Label names for readability
    bin_labels = []
    for i in range(n_bins):
        lo = bin_edges[i]
        hi = bin_edges[i + 1]
        bin_labels.append(f"Bin{i}\n({lo:.0f}–{hi:.0f})")

    f1_macro    = f1_score(y_true_bins, y_pred_bins, average='macro',    zero_division=0)
    f1_weighted = f1_score(y_true_bins, y_pred_bins, average='weighted', zero_division=0)

    print("\n" + "="*50)
    print(f"  CLASSIFICATION METRICS  ({n_bins} consumption bins)")
    print("="*50)
    print(f"  F1 Score (Macro)    : {f1_macro:.4f}")
    print(f"  F1 Score (Weighted) : {f1_weighted:.4f}")
    print("\n  Detailed Report:")
    print(classification_report(
        y_true_bins, y_pred_bins,
        target_names=[f"Bin{i}" for i in range(n_bins)],
        zero_division=0
    ))
    print("="*50)

    # ── Confusion Matrix Plot ──────────────────────────
    cm = confusion_matrix(y_true_bins, y_pred_bins)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='YlOrRd',
        xticklabels=[f"Bin{i}" for i in range(n_bins)],
        yticklabels=[f"Bin{i}" for i in range(n_bins)]
    )
    plt.title(f'Confusion Matrix  ({n_bins} Consumption Bins)', fontsize=14, fontweight='bold')
    plt.ylabel('Actual Bin')
    plt.xlabel('Predicted Bin')
    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/confusion_matrix.png', dpi=150)
    plt.show()
    print("[INFO] Confusion matrix saved → results/confusion_matrix.png")

    return f1_macro, f1_weighted


# ==============================================================
# PLOTS
# ==============================================================
def plot_training_history(history):
    """Plot loss and MAE curves during training."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.history['loss'],     label='Train Loss', color='steelblue')
    axes[0].plot(history.history['val_loss'], label='Val Loss',   color='tomato')
    axes[0].set_title('Loss (Huber) over Epochs')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(history.history['mae'],     label='Train MAE', color='steelblue')
    axes[1].plot(history.history['val_mae'], label='Val MAE',   color='tomato')
    axes[1].set_title('MAE over Epochs')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('MAE')
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.suptitle('LSTM Training History', fontsize=15, fontweight='bold')
    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/training_history.png', dpi=150)
    plt.show()
    print("[INFO] Training history saved → results/training_history.png")


def plot_predictions(y_true_mw, y_pred_mw, n_show=500):
    """Plot actual vs predicted MW for a portion of the test set."""
    plt.figure(figsize=(15, 5))
    idx = range(n_show)
    plt.plot(idx, y_true_mw[:n_show], label='Actual',    color='steelblue',  linewidth=1.5)
    plt.plot(idx, y_pred_mw[:n_show], label='Predicted', color='orangered',  linewidth=1.5, linestyle='--')
    plt.title(f'Actual vs Predicted Hourly Energy Consumption (first {n_show} test hours)',
              fontsize=13, fontweight='bold')
    plt.xlabel('Hour')
    plt.ylabel('Energy Consumption (MW)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/actual_vs_predicted.png', dpi=150)
    plt.show()
    print("[INFO] Prediction plot saved → results/actual_vs_predicted.png")


def plot_scatter(y_true_mw, y_pred_mw):
    """Scatter plot of actual vs predicted with ideal line."""
    plt.figure(figsize=(7, 7))
    plt.scatter(y_true_mw, y_pred_mw, alpha=0.3, s=5, color='steelblue')
    lo = min(y_true_mw.min(), y_pred_mw.min())
    hi = max(y_true_mw.max(), y_pred_mw.max())
    plt.plot([lo, hi], [lo, hi], 'r--', linewidth=2, label='Perfect Prediction')
    plt.title('Actual vs Predicted (Scatter)', fontsize=13, fontweight='bold')
    plt.xlabel('Actual MW'); plt.ylabel('Predicted MW')
    plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/scatter_plot.png', dpi=150)
    plt.show()
    print("[INFO] Scatter plot saved → results/scatter_plot.png")


# ==============================================================
# FULL EVALUATION PIPELINE
# ==============================================================
def full_evaluation(model, X_test, y_test_scaled, scaler, history, n_bins=5):
    """
    End-to-end evaluation:
      1. Predict on test set
      2. Inverse-scale to MW
      3. Regression metrics
      4. Classification metrics (F1, confusion matrix)
      5. All plots

    Args:
        model          : trained Keras model
        X_test         : test sequences (scaled)
        y_test_scaled  : true labels (scaled)
        scaler         : fitted MinMaxScaler
        history        : Keras training history object
        n_bins         : number of bins for classification metrics

    Returns:
        dict of all metrics
    """
    n_features = scaler.n_features_in_

    # ── 1. Predict ─────────────────────────────────────
    y_pred_scaled = model.predict(X_test, verbose=1).flatten()

    # ── 2. Inverse scale ───────────────────────────────
    y_true_mw = inverse_scale(scaler, y_test_scaled, n_features)
    y_pred_mw = inverse_scale(scaler, y_pred_scaled,  n_features)

    # ── 3. Regression metrics ──────────────────────────
    reg_metrics = regression_metrics(y_true_mw, y_pred_mw)

    # ── 4. Classification metrics ──────────────────────
    f1_macro, f1_weighted = classification_metrics(y_true_mw, y_pred_mw, n_bins)

    # ── 5. Plots ───────────────────────────────────────
    plot_training_history(history)
    plot_predictions(y_true_mw, y_pred_mw)
    plot_scatter(y_true_mw, y_pred_mw)

    # ── 6. Summary ─────────────────────────────────────
    all_metrics = {**reg_metrics, 'F1_macro': f1_macro, 'F1_weighted': f1_weighted}
    print("\n[SUMMARY]", all_metrics)
    return all_metrics


if __name__ == '__main__':
    print("evaluate.py — import and call full_evaluation() from train.py")