# ==============================================================
# LSTM Hourly Energy Consumption Predictor  — SINGLE FILE
# Run: python train.py
# Dataset: https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption
# ==============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    f1_score, classification_report, confusion_matrix
)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# ==============================================================
# CONFIG  — change CSV_PATH to your downloaded file
# ==============================================================
CSV_PATH   = "AEP_hourly.csv"
SEQ_LENGTH = 24
EPOCHS     = 100
BATCH_SIZE = 64
N_BINS     = 5
TEST_SPLIT = 0.2

os.makedirs("models",  exist_ok=True)
os.makedirs("results", exist_ok=True)

# ==============================================================
# STEP 1 — LOAD & PREPROCESS
# ==============================================================
print("\n" + "="*55)
print("  LSTM Energy Demand Predictor  |  SIH Smart Grid")
print("="*55)
print("\n[1/5] Loading and preprocessing data ...")

df = pd.read_csv(CSV_PATH)
print(f"      Raw shape: {df.shape}")

date_col  = df.columns[0]
value_col = df.columns[1]

df[date_col] = pd.to_datetime(df[date_col])
df = df.sort_values(date_col).reset_index(drop=True)
df = df.dropna()

print(f"      Date range : {df[date_col].min()} --> {df[date_col].max()}")
print(f"      Target col : '{value_col}'")

# Time features
df['hour']       = df[date_col].dt.hour
df['dayofweek']  = df[date_col].dt.dayofweek
df['month']      = df[date_col].dt.month
df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)

# Cyclical encoding
df['hour_sin']  = np.sin(2 * np.pi * df['hour']  / 24)
df['hour_cos']  = np.cos(2 * np.pi * df['hour']  / 24)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

# Lag features
for lag in [1, 2, 24, 48]:
    df[f'lag_{lag}'] = df[value_col].shift(lag)

# Rolling stats
df['rolling_mean_24'] = df[value_col].rolling(24).mean()
df['rolling_std_24']  = df[value_col].rolling(24).std()

df = df.dropna().reset_index(drop=True)
print(f"      Shape after feature engineering: {df.shape}")

# ==============================================================
# STEP 2 — SCALE & CREATE SEQUENCES
# ==============================================================
print("\n[2/5] Scaling and creating sequences ...")

feature_cols = [
    value_col,
    'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
    'dayofweek', 'is_weekend',
    'lag_1', 'lag_2', 'lag_24', 'lag_48',
    'rolling_mean_24', 'rolling_std_24'
]
N_FEATURES = len(feature_cols)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(df[feature_cols].values)
joblib.dump(scaler, "models/scaler.pkl")
print("      Scaler saved --> models/scaler.pkl")

X, y = [], []
for i in range(SEQ_LENGTH, len(scaled)):
    X.append(scaled[i - SEQ_LENGTH:i, :])
    y.append(scaled[i, 0])

X, y = np.array(X), np.array(y)
print(f"      Sequence shape -- X: {X.shape}, y: {y.shape}")

split_idx = int(len(X) * (1 - TEST_SPLIT))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]
print(f"      Train: {len(X_train)} samples | Test: {len(X_test)} samples")

# ==============================================================
# STEP 3 — BUILD & TRAIN LSTM
# ==============================================================
print("\n[3/5] Building LSTM model ...")

model = Sequential([
    Input(shape=(SEQ_LENGTH, N_FEATURES)),

    LSTM(128, return_sequences=True),
    Dropout(0.2),
    BatchNormalization(),

    LSTM(64, return_sequences=False),
    Dropout(0.2),
    BatchNormalization(),

    Dense(32, activation='relu'),
    Dense(1)
])

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss='huber',
    metrics=['mae']
)
model.summary()

callbacks = [
    EarlyStopping(monitor='val_loss', patience=10,
                  restore_best_weights=True, verbose=1),
    ModelCheckpoint('models/best_lstm.keras', monitor='val_loss',
                    save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                      patience=5, min_lr=1e-6, verbose=1)
]

print("\n      Training ...")
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.1,
    callbacks=callbacks,
    shuffle=False,
    verbose=1
)
print("      Training complete.")

# ==============================================================
# STEP 4 — PREDICT & INVERSE SCALE
# ==============================================================
print("\n[4/5] Predicting on test set ...")

y_pred_scaled = model.predict(X_test, verbose=1).flatten()

def inv_scale(vals):
    dummy = np.zeros((len(vals), N_FEATURES))
    dummy[:, 0] = vals
    return scaler.inverse_transform(dummy)[:, 0]

y_true_mw = inv_scale(y_test)
y_pred_mw = inv_scale(y_pred_scaled)

# ==============================================================
# STEP 5 — EVALUATION
# ==============================================================
print("\n[5/5] Evaluating ...")

mae  = mean_absolute_error(y_true_mw, y_pred_mw)
rmse = np.sqrt(mean_squared_error(y_true_mw, y_pred_mw))
r2   = r2_score(y_true_mw, y_pred_mw)
mask = y_true_mw != 0
mape = np.mean(np.abs((y_true_mw[mask] - y_pred_mw[mask]) / y_true_mw[mask])) * 100

print("\n" + "="*45)
print("  REGRESSION METRICS")
print("="*45)
print(f"  MAE   : {mae:>10.2f} MW")
print(f"  RMSE  : {rmse:>10.2f} MW")
print(f"  MAPE  : {mape:>10.2f} %")
print(f"  R2    : {r2:>10.4f}  (1.0 = perfect)")
print("="*45)

min_v, max_v = y_true_mw.min(), y_true_mw.max()
bin_edges    = np.linspace(min_v, max_v, N_BINS + 1)

y_true_bins = np.digitize(y_true_mw, bin_edges[1:-1])
y_pred_clip = np.clip(y_pred_mw, min_v, max_v)
y_pred_bins = np.digitize(y_pred_clip, bin_edges[1:-1])

f1_macro    = f1_score(y_true_bins, y_pred_bins, average='macro',    zero_division=0)
f1_weighted = f1_score(y_true_bins, y_pred_bins, average='weighted', zero_division=0)

print(f"\n  CLASSIFICATION METRICS  ({N_BINS} bins)")
print("="*45)
print(f"  F1 Score (Macro)    : {f1_macro:.4f}")
print(f"  F1 Score (Weighted) : {f1_weighted:.4f}")
print("\n  Detailed Report:")
print(classification_report(
    y_true_bins, y_pred_bins,
    target_names=[f"Bin{i}" for i in range(N_BINS)],
    zero_division=0
))

# Plot 1: Confusion Matrix
cm = confusion_matrix(y_true_bins, y_pred_bins)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd',
            xticklabels=[f"Bin{i}" for i in range(N_BINS)],
            yticklabels=[f"Bin{i}" for i in range(N_BINS)])
plt.title(f'Confusion Matrix -- {N_BINS} Consumption Bins', fontsize=14, fontweight='bold')
plt.ylabel('Actual Bin')
plt.xlabel('Predicted Bin')
plt.tight_layout()
plt.savefig('results/confusion_matrix.png', dpi=150)
plt.show()
print("  Saved --> results/confusion_matrix.png")

# Plot 2: Training History
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(history.history['loss'],     label='Train Loss', color='steelblue')
axes[0].plot(history.history['val_loss'], label='Val Loss',   color='tomato')
axes[0].set_title('Loss over Epochs')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(history.history['mae'],     label='Train MAE', color='steelblue')
axes[1].plot(history.history['val_mae'], label='Val MAE',   color='tomato')
axes[1].set_title('MAE over Epochs')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('MAE')
axes[1].legend(); axes[1].grid(alpha=0.3)

plt.suptitle('LSTM Training History', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('results/training_history.png', dpi=150)
plt.show()
print("  Saved --> results/training_history.png")

# Plot 3: Actual vs Predicted
N_SHOW = min(500, len(y_true_mw))
plt.figure(figsize=(15, 5))
plt.plot(y_true_mw[:N_SHOW], label='Actual',    color='steelblue',  linewidth=1.5)
plt.plot(y_pred_mw[:N_SHOW], label='Predicted', color='orangered',  linewidth=1.5, linestyle='--')
plt.title(f'Actual vs Predicted -- First {N_SHOW} Test Hours', fontsize=13, fontweight='bold')
plt.xlabel('Hour'); plt.ylabel('Energy (MW)')
plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('results/actual_vs_predicted.png', dpi=150)
plt.show()
print("  Saved --> results/actual_vs_predicted.png")

# Plot 4: Scatter
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
print("  Saved --> results/scatter_plot.png")

# ==============================================================
# FINAL SUMMARY
# ==============================================================
print("\n" + "="*55)
print("  FINAL SUMMARY")
print("="*55)
print(f"  MAE            : {mae:.2f} MW")
print(f"  RMSE           : {rmse:.2f} MW")
print(f"  MAPE           : {mape:.2f} %")
print(f"  R2             : {r2:.4f}")
print(f"  F1 (Macro)     : {f1_macro:.4f}")
print(f"  F1 (Weighted)  : {f1_weighted:.4f}")
print("="*55)
print("\n  Plots  --> results/")
print("  Model  --> models/best_lstm.keras")
print("  Done!")