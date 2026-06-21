import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import autokeras as ak
from tpot import TPOTRegressor
import os
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

base_dir = r"d:\research paper - Copy\code+data\code+data\dataset\csv_format"
train_batteries = ["B0005", "B0006", "B0007"]
test_battery = "B0018"

def load_data(batteries):
    cycle_data = []
    for b in batteries:
        df = pd.read_csv(os.path.join(base_dir, f"{b}_discharge.csv"))
        agg_df = df.groupby('id_cycle').agg({
            'Voltage_measured': ['max', 'min', 'mean'],
            'Current_measured': ['max', 'min', 'mean'],
            'Temperature_measured': 'mean',
            'Capacity': 'max'
        })
        agg_df.columns = [f"{c[0]}_{c[1]}" for c in agg_df.columns]
        cycle_data.append(agg_df)
    full_df = pd.concat(cycle_data, ignore_index=True)
    
    # Feature X: everything except Capacity
    # Target y: Capacity_max
    X = full_df.drop('Capacity_max', axis=1).values
    y = full_df['Capacity_max'].values
    return X, y

print("Loading Base Paper Train/Test Split...")
X_train_raw, y_train_raw = load_data(train_batteries)
X_test_raw, y_test_raw = load_data([test_battery])

# Normalize
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train = scaler_X.fit_transform(X_train_raw)
X_test = scaler_X.transform(X_test_raw)

y_train = scaler_y.fit_transform(y_train_raw.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test_raw.reshape(-1, 1)).flatten()

def evaluate_metrics(y_true_scaled, y_pred_scaled, name):
    # Inverse transform to get real capacity values for metrics
    y_true = scaler_y.inverse_transform(y_true_scaled.reshape(-1, 1)).flatten()
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"\n--- {name} Baseline Results ---")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2:   {r2:.4f}")
    return mae, rmse, r2

# 1. Base Paper Baseline: AutoKeras
print("\nTraining Baseline AutoKeras...")
reg = ak.StructuredDataRegressor(max_trials=5, overwrite=True, seed=42)
reg.fit(X_train, y_train, epochs=100, verbose=0)
y_pred_ak = reg.predict(X_test, verbose=0).flatten()
evaluate_metrics(y_test_scaled, y_pred_ak, "AutoKeras (Base)")

# 2. Base Paper Baseline: TPOT
print("\nTraining Baseline TPOT...")
tpot = TPOTRegressor(generations=3, population_size=10, random_state=42)
tpot.fit(X_train, y_train)
y_pred_tpot = tpot.predict(X_test)
evaluate_metrics(y_test_scaled, y_pred_tpot, "TPOT (Base)")
