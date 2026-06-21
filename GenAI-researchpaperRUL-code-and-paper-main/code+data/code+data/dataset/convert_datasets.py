import pandas as pd
from scipy.io import loadmat
import numpy as np
import os
import json

def to_df(mat_db):
    cycles_cols = ['type', 'ambient_temperature', 'time']
    features_cols = {
        'charge': ['Voltage_measured', 'Current_measured', 'Temperature_measured',
                'Current_charge', 'Voltage_charge', 'Time'],
        'discharge': ['Voltage_measured', 'Current_measured', 'Temperature_measured',
                    'Current_charge', 'Voltage_charge', 'Time', 'Capacity'],
        'impedance': ['Sense_current', 'Battery_current', 'Current_ratio',
                    'Battery_impedance', 'Rectified_impedance', 'Re', 'Rct']
    }
    df = {key: pd.DataFrame() for key in features_cols.keys()}
    
    # Get every cycle
    cycles = [[row.flat[0] for row in line] for line in mat_db[0][0][0][0]]
    for cycle_id, cycle_data in enumerate(cycles):
        tmp = pd.DataFrame()
        features_x_cycle = cycle_data[-1]
        features = features_cols[cycle_data[0]]
        for feature, data in zip(features, features_x_cycle):
            if len(data[0]) > 1:
                tmp[feature] = data[0]
            else:
                tmp[feature] = data[0][0]
        tmp['id_cycle'] = cycle_id
        for k, col in enumerate(cycles_cols):
            tmp[col] = cycle_data[k]
        cycle_type = cycle_data[0]
        # Use concat instead of append since append is deprecated
        if not tmp.empty:
            df[cycle_type] = pd.concat([df[cycle_type], tmp], ignore_index=True)
            
    return df

def convert_numpy(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(i) for i in obj]
    return obj

def main():
    batteries = ['B0005', 'B0006', 'B0007', 'B0018']
    base_dir = r'd:\research paper - Copy\code+data\code+data\dataset'
    
    out_dir = os.path.join(base_dir, 'csv_format')
    os.makedirs(out_dir, exist_ok=True)
    
    for B in batteries:
        mat_path = os.path.join(base_dir, f'{B}.mat')
        if os.path.exists(mat_path):
            print(f"Converting {B}.mat...")
            try:
                mat_db = loadmat(mat_path)[B]
                dfs = to_df(mat_db)
                for ctype in ['charge', 'discharge', 'impedance']:
                    if not dfs[ctype].empty:
                        out_path = os.path.join(out_dir, f'{B}_{ctype}.csv')
                        # For impedance, cells might contain complex numbers which CSV doesn't handle natively
                        # convert complex numbers to string just in case
                        if ctype == 'impedance':
                            for col in dfs[ctype].columns:
                                dfs[ctype][col] = dfs[ctype][col].apply(lambda x: str(x) if isinstance(x, complex) else x)
                        dfs[ctype].to_csv(out_path, index=False)
                        print(f"  -> Saved {out_path}")
            except Exception as e:
                print(f"  -> Error converting {B}.mat: {e}")
                
    npy_path = os.path.join(base_dir, 'NASA.npy')
    if os.path.exists(npy_path):
        print(f"Converting NASA.npy...")
        try:
            bat_data = np.load(npy_path, allow_pickle=True)
            if hasattr(bat_data, 'item'):
                bat_data = bat_data.item()
            out_json = os.path.join(out_dir, 'NASA.json')
            with open(out_json, 'w') as f:
                json.dump(convert_numpy(bat_data), f)
            print(f"  -> Saved {out_json}")
        except Exception as e:
            print(f"  -> Error converting NASA.npy: {e}")

if __name__ == "__main__":
    main()
