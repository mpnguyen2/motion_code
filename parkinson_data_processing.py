import os, math
from collections import defaultdict
import numpy as np
import pandas as pd


def get_label_to_ids_map(ids_labels_file):
    df = pd.read_csv(ids_labels_file)
    label_to_ids_map = defaultdict(list)
    for _, row in df.iterrows():
        # p = row['subject_id']
        r1, r2, r3 = row['on_off'], row['dyskinesia'], row['tremor']
        id = row['measurement_id']
        if (not math.isnan(r1)) and (not math.isnan(r2)) and (not math.isnan(r3)):
            label = (int(r1), int(r2), int(r3))
            label_to_ids_map[label].append(id)
    return label_to_ids_map

def get_refined_label_to_ids_map(label_to_ids_map,
                                 mode='two_tremor'):
    revised_label_to_ids_map = defaultdict(list)
    label_names = []
    for l, ids in label_to_ids_map.items():
        if mode == 'two_tremor':
            if l == (0, 0, 0):
                label = 0
            elif l == (0, 0, 1):
                label = 1
            else:
                label = -1
            label_names = ['normal', 'light tremor']
        if mode == 'three_tremor':
            if l == (0, 0, 0):
                label = 0
            elif l == (0, 0, 1):
                label = 1
            elif l[2] >= 3:
                label = 2
            else:
                label = -1
            label_names = ['normal', 'light tremor', 'noticeable tremor']
        if label != -1:
            revised_label_to_ids_map[label].extend(label_to_ids_map[l])

    return revised_label_to_ids_map, label_names

def get_parkinson_train_test_data_helper(ids_labels_file,
                                        sensor_data_path,
                                        refine_label_mode='two_tremor',
                                        max_num_series_per_class=20,
                                        use_fft=False,
                                        smoothing_params = {
                                            'type': 'EMA',
                                            'rolling_window': 30,
                                            'alpha_ema': 0.1
                                        },
                                        seed=42,
                                        ):
    # Get label_to_ids map
    label_to_ids_map = get_label_to_ids_map(ids_labels_file)
    revised_label_to_ids_map, label_names = (
        get_refined_label_to_ids_map(label_to_ids_map, refine_label_mode)
    )

    # Get smooth parameters:
    smooth_type = smoothing_params['type']
    rolling_window = smoothing_params['rolling_window']
    alpha_ema = smoothing_params['alpha_ema']
    np.random.seed(seed)

    # Get training data through actual sensor data
    X_train, Y_train, labels_train = [], [], []
    X_test, Y_test, labels_test = [], [], []

    # Go through revised list (revised label is from 0 to num_motion)
    for label, ids in revised_label_to_ids_map.items():
        num_series = len(ids)
        if num_series > max_num_series_per_class:
            sample_indices = np.random.choice(np.arange(num_series),
                                            size=max_num_series_per_class,
                                            replace=False)
        else:
            sample_indices = np.arange(num_series)
        sample_indices = set(sample_indices)

        # Sensor data reading
        for i, id in enumerate(ids):
            df = pd.read_csv(os.path.join(sensor_data_path, id + '.csv'))
            df.ffill(inplace=True)
            if not use_fft:
                df['Z1'] = (df['Z'] - df['Z'].shift(1)).abs()
                df.fillna(0, inplace=True)
                v = df.isna().sum(axis=0).sum()
                if v != 0:
                    print(id, v)             
                if smooth_type == 'SMA':
                    sensor_df = df['Z1'].rolling(window=rolling_window, min_periods=1).mean()
                elif smooth_type == 'CMA':
                    sensor_df = df['Z1'].expanding().mean()
                elif smooth_type == 'EMA':
                    sensor_df = df['Z1'].ewm(alpha=alpha_ema, min_periods=1, adjust=True).mean()
                else:
                    sensor_df = df['Z1'].rolling(window=rolling_window, min_periods=1).std()
            else:
                df['fft'] = np.clip(np.abs(np.fft.fft(df['Z'])), -10, 10)
                df['fft_SMA'] = (
                    df['fft'].rolling(window=rolling_window, min_periods=1).mean()
                )
                sensor_df = (
                    df['fft_SMA'].ewm(alpha=alpha_ema, min_periods=1, adjust=True).mean()
                )
            
            # Add timestamps and sensor values to train/test data.
            time_stamps = df['Timestamp'].values/1200
            sensor_values = sensor_df.values
            if i in sample_indices:
                X_train.append(time_stamps)
                Y_train.append(sensor_values)
                labels_train.append(label)
            else:
                X_test.append(time_stamps)
                Y_test.append(sensor_values)
                labels_test.append(label)

    return label_to_ids_map, revised_label_to_ids_map, label_names,\
            X_train, Y_train, labels_train,\
            X_test, Y_test, labels_test


def get_parkinson_train_test_data(name):
    # Default data input paths
    ids_labels_file = 'data/parkinson/ids_labels_file.csv'
    sensor_data_path = 'data/parkinson/sensor_data/'

    # name is either PD setting 1 or PD setting 2
    if name == 'PD setting 1':
        _, _, _, X_train, Y_train, labels_train,\
        X_test, Y_test, labels_test = get_parkinson_train_test_data_helper(ids_labels_file,
                                        sensor_data_path,
                                        refine_label_mode='two_tremor',
                                        max_num_series_per_class=10,
                                        seed=20)
    else:
        _, _, _, X_train, Y_train, labels_train,\
        X_test, Y_test, labels_test = get_parkinson_train_test_data_helper(ids_labels_file,
                                        sensor_data_path,
                                        refine_label_mode='three_tremor',
                                        max_num_series_per_class=8,
                                        seed=42)
        
    return X_train, Y_train, labels_train, X_test, Y_test, labels_test