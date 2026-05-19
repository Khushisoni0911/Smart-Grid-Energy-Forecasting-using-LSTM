# ==============================================================
# LSTM Energy Forecast — Input a Date/Time, Get MW Prediction
# Run AFTER train.py has saved: models/best_lstm.keras
#                                models/scaler.pkl
# Usage: python forecast.py
# ==============================================================

import os
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")

from tensorflow.keras.models import load_model

# ==============================================================
# CONFIG  — must match what you used in train.py
# ==============================================================
CSV_PATH   = "AEP_hourly.csv"
SEQ_LENGTH = 24        # lookback window (hours)
MODEL_PATH = "models/best_lstm.keras"
SCALER_PATH= "models/scaler.pkl"

# ==============================================================
# STEP 1 — Load model + scaler
# ==============================================================
print("\n" + "="*55)
print("  LSTM Energy Forecast — Date/Time Predictor")
print("="*55)

if not os.path.exists(MODEL_PATH):
    print(f"\n[ERROR] Model not found at '{MODEL_PATH}'")
    print("        Please run train.py first to train and save the model.")
    exit()

model  = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
N_FEATURES = scaler.n_features_in_
print(f"\n  Model  loaded : {MODEL_PATH}")
print(f"  Scaler loaded : {SCALER_PATH}")
print(f"  Features      : {N_FEATURES}")

# ==============================================================
# STEP 2 — Load historical data (needed to build input window)
# ==============================================================
print(f"\n  Loading historical data from '{CSV_PATH}' ...")

df = pd.read_csv(CSV_PATH)
date_col  = df.columns[0]
value_col = df.columns[1]

df[date_col] = pd.to_datetime(df[date_col])
df = df.sort_values(date_col).reset_index(drop=True)
df = df.dropna()

# Re-build all features (must exactly match train.py)
df['hour']       = df[date_col].dt.hour
df['dayofweek']  = df[date_col].dt.dayofweek
df['month']      = df[date_col].dt.month
df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
df['hour_sin']   = np.sin(2 * np.pi * df['hour']  / 24)
df['hour_cos']   = np.cos(2 * np.pi * df['hour']  / 24)
df['month_sin']  = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos']  = np.cos(2 * np.pi * df['month'] / 12)
for lag in [1, 2, 24, 48]:
    df[f'lag_{lag}'] = df[value_col].shift(lag)
df['rolling_mean_24'] = df[value_col].rolling(24).mean()
df['rolling_std_24']  = df[value_col].rolling(24).std()
df = df.dropna().reset_index(drop=True)

feature_cols = [
    value_col,
    'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
    'dayofweek', 'is_weekend',
    'lag_1', 'lag_2', 'lag_24', 'lag_48',
    'rolling_mean_24', 'rolling_std_24'
]

scaled_all = scaler.transform(df[feature_cols].values)

print(f"  Data loaded   : {len(df)} rows")
print(f"  Date range    : {df[date_col].min()} --> {df[date_col].max()}")

# ==============================================================
# HELPER FUNCTIONS
# ==============================================================

def inv_scale(vals):
    """Inverse-transform scaled MW values back to original units."""
    dummy = np.zeros((len(vals), N_FEATURES))
    dummy[:, 0] = vals
    return scaler.inverse_transform(dummy)[:, 0]


def get_actual_mw(target_dt):
    """Helper to get actual MW if it exists in the dataset."""
    match = df[df[date_col] == target_dt]
    if not match.empty:
        return match.iloc[0][value_col]
    return None


def get_window_for_datetime(target_dt):
    """
    Find the 24-hour window of scaled features ending just before target_dt.
    Returns the window array or None if not enough history.
    """
    # Find the row index at or just before target_dt
    mask = df[date_col] <= target_dt
    if mask.sum() < SEQ_LENGTH:
        return None, None
    idx = df[mask].index[-1]          # last known row before target
    if idx < SEQ_LENGTH - 1:
        return None, None
    window = scaled_all[idx - SEQ_LENGTH + 1 : idx + 1]   # shape (24, N_FEATURES)
    return window, idx


def forecast_single(target_dt):
    """
    Predict energy consumption for a single target datetime.
    Returns predicted MW value.
    """
    window, idx = get_window_for_datetime(target_dt)
    if window is None:
        print(f"  [!] Not enough historical data before {target_dt}")
        return None
    X_input = window.reshape(1, SEQ_LENGTH, N_FEATURES)
    pred_scaled = model.predict(X_input, verbose=0).flatten()
    pred_mw = inv_scale(pred_scaled)[0]
    return pred_mw


def forecast_multi_step(start_dt, n_hours):
    """
    Predict energy consumption for n_hours starting from start_dt.
    Each prediction feeds into the next (true multi-step forecast).
    Returns list of (datetime, predicted_mw, actual_mw) tuples.
    """
    window, idx = get_window_for_datetime(start_dt)
    if window is None:
        print(f"  [!] Not enough historical data before {start_dt}")
        return []

    window = window.copy()   # shape (24, N_FEATURES)
    results = []

    current_dt = start_dt + pd.Timedelta(hours=1)

    for step in range(n_hours):
        X_input     = window.reshape(1, SEQ_LENGTH, N_FEATURES)
        pred_scaled = model.predict(X_input, verbose=0).flatten()[0]
        pred_mw     = inv_scale([pred_scaled])[0]
        actual_mw   = get_actual_mw(current_dt)

        results.append((current_dt, pred_mw, actual_mw))

        # Build next row features using predicted value + time info
        next_hour  = current_dt.hour
        next_dow   = current_dt.dayofweek
        next_month = current_dt.month
        next_row = np.array([
            pred_scaled,
            np.sin(2 * np.pi * next_hour  / 24),
            np.cos(2 * np.pi * next_hour  / 24),
            np.sin(2 * np.pi * next_month / 12),
            np.cos(2 * np.pi * next_month / 12),
            next_dow,
            1 if next_dow >= 5 else 0,
            window[-1, 0],    # lag_1  = previous prediction
            window[-2, 0],    # lag_2
            window[-24, 0] if len(window) >= 24 else window[0, 0],  # lag_24
            window[-24, 0] if len(window) >= 24 else window[0, 0],  # lag_48 (approx)
            np.mean(window[:, 0]),   # rolling_mean_24
            np.std(window[:, 0]),    # rolling_std_24
        ])

        # Slide window forward
        window = np.vstack([window[1:], next_row])
        current_dt += pd.Timedelta(hours=1)

    return results


def consumption_label(mw):
    """Return a human-readable demand level."""
    if   mw < 10000: return "Very Low"
    elif mw < 13000: return "Low"
    elif mw < 16000: return "Medium"
    elif mw < 19000: return "High"
    else:            return "Very High"


def print_forecast_table(results):
    """Pretty-print a forecast table."""
    print("\n" + "-"*78)
    print(f"  {'Date & Time':<23} {'Predicted MW':>14}  {'Actual MW':>14}  {'Demand Level'}")
    print("-" * 78)
    for dt, pred_mw, actual_mw in results:
        label = consumption_label(pred_mw)
        actual_str = f"{actual_mw:>14,.2f}" if actual_mw is not None else f"{'N/A':>14}"
        print(f"  {str(dt):<23} {pred_mw:>14,.2f}  {actual_str}  {label}")
    print("-" * 78)

# ==============================================================
# STEP 3 — Interactive Forecast Loop
# ==============================================================
print("\n" + "="*55)
print("  FORECAST MODE")
print("="*55)
print(f"  Dataset covers: {df[date_col].min().strftime('%Y-%m-%d')} to "
      f"{df[date_col].max().strftime('%Y-%m-%d')}")
print("  Enter a date/time within or near this range.\n")

while True:
    print("\n  Options:")
    print("    1 → Forecast for a single date & time")
    print("    2 → Forecast for next N hours from a date & time")
    print("    3 → Forecast for a full day (24 hours)")
    print("    q → Quit")

    choice = input("\n  Your choice: ").strip().lower()

    if choice == 'q':
        print("\n  Goodbye!\n")
        break

    # ── Single datetime forecast ────────────────────────────
    elif choice == '1':
        raw = input("  Enter date & time (YYYY-MM-DD HH:MM): ").strip()
        try:
            target_dt = pd.to_datetime(raw)
        except Exception:
            print("  [!] Invalid format. Use YYYY-MM-DD HH:MM  e.g. 2017-06-15 14:00")
            continue

        print(f"\n  Forecasting for: {target_dt} ...")
        pred_mw = forecast_single(target_dt)
        if pred_mw is not None:
            actual_mw = get_actual_mw(target_dt)
            label = consumption_label(pred_mw)
            print("\n" + "-"*45)
            print(f"  Date & Time   : {target_dt}")
            print(f"  Predicted MW  : {pred_mw:,.2f} MW")
            if actual_mw is not None:
                print(f"  Actual MW     : {actual_mw:,.2f} MW")
            print(f"  Demand Level  : {label}")
            print("-"*45)

    # ── Multi-step forecast (N hours) ───────────────────────
    elif choice == '2':
        raw = input("  Enter start date & time (YYYY-MM-DD HH:MM): ").strip()
        try:
            start_dt = pd.to_datetime(raw)
        except Exception:
            print("  [!] Invalid format. Use YYYY-MM-DD HH:MM  e.g. 2017-06-15 08:00")
            continue

        try:
            n_hours = int(input("  How many hours to forecast? (e.g. 6, 12, 48): ").strip())
        except ValueError:
            print("  [!] Please enter a whole number.")
            continue

        print(f"\n  Forecasting {n_hours} hours from {start_dt} ...")
        results = forecast_multi_step(start_dt, n_hours)
        if results:
            print_forecast_table(results)

            # Save to CSV
            save = input("\n  Save forecast to CSV? (y/n): ").strip().lower()
            if save == 'y':
                out_df = pd.DataFrame(results, columns=['DateTime', 'Predicted_MW', 'Actual_MW'])
                out_df['Demand_Level'] = out_df['Predicted_MW'].apply(consumption_label)
                fname = f"results/forecast_{start_dt.strftime('%Y%m%d_%H%M')}_{n_hours}h.csv"
                os.makedirs("results", exist_ok=True)
                out_df.to_csv(fname, index=False)
                print(f"  Saved --> {fname}")

    # ── Full day forecast (24 hours) ────────────────────────
    elif choice == '3':
        raw = input("  Enter date (YYYY-MM-DD): ").strip()
        try:
            day_dt = pd.to_datetime(raw + " 00:00")
        except Exception:
            print("  [!] Invalid format. Use YYYY-MM-DD  e.g. 2017-06-15")
            continue

        print(f"\n  Forecasting full day: {raw} (24 hours) ...")
        results = forecast_multi_step(day_dt, 24)
        if results:
            print_forecast_table(results)
            avg_mw = np.mean([mw for _, mw, _ in results])
            peak   = max(results, key=lambda x: x[1])
            low    = min(results, key=lambda x: x[1])
            print(f"\n  Daily Average : {avg_mw:,.2f} MW")
            print(f"  Peak Hour     : {peak[0].strftime('%H:%M')}  -->  {peak[1]:,.2f} MW")
            print(f"  Lowest Hour   : {low[0].strftime('%H:%M')}   -->  {low[1]:,.2f} MW")

            save = input("\n  Save forecast to CSV? (y/n): ").strip().lower()
            if save == 'y':
                out_df = pd.DataFrame(results, columns=['DateTime', 'Predicted_MW', 'Actual_MW'])
                out_df['Demand_Level'] = out_df['Predicted_MW'].apply(consumption_label)
                fname = f"results/forecast_{raw}.csv"
                os.makedirs("results", exist_ok=True)
                out_df.to_csv(fname, index=False)
                print(f"  Saved --> {fname}")

    else:
        print("  [!] Invalid choice. Enter 1, 2, 3 or q.")