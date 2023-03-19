import pickle
import numpy as np
from utils import *
import argparse
import scipy.io.wavfile as wavfile
from sktime.datasets import load_UCR_UEA_dataset

# Color list for plotting
COLOR_LIST = ['red', 'blue', 'green', 'yellow', 'black', 'orange', 'brown', 'grey', 'purple', 'hotpink']

def clear():
    os.system('clear')

def load_UCR_UEA_data(name, mode='train', visualize=False):
    Y, labels= load_UCR_UEA_dataset(name=name, split=mode, return_X_y=True, return_type="numpy3d")
    try:
        labels = np.array(labels, dtype=int)
        labels -= 1
    except:
        return -1, np.array([]), np.array([]), np.array([])
    Y = Y[:, 0, :]
    num_samples = Y.shape[0]; seq_len = Y.shape[1]
    X = np.tile(np.linspace(0, 1, seq_len), (num_samples, 1))
    # Visualization
    if visualize:
        for i in range(num_samples):
            plt.plot(X[i], Y[i], color=COLOR_LIST[int(labels[i])])
        plt.show()
    return num_samples, X, Y, labels

def save_data(X, Y, labels, data_path):
    return np.savez(data_path, X=X, Y=Y, labels=labels)  

## Sound dataset ##
def read_sound_timeseries(file_name, down_sampling_rate=100):
    samplerate, data = wavfile.read(file_name)
    duration = len(data)/samplerate
    time = np.arange(0, 1, 1/(duration*samplerate))
    intervals = np.array(np.arange(0, len(time), len(time)/down_sampling_rate), dtype=int)
    time = time[intervals]; data = data[intervals]
    data = np.abs(data)/np.max(np.abs(data))
    return time, data

def extract_data_from_sound_dataset(input_dir, output_dir):
    cur_label = 0
    for single_dir in os.scandir(input_dir):
        if not single_dir.is_dir() or single_dir.name == 'processed':
            continue
        X, Y, labels = [], [], []
        for sound_file in os.scandir(single_dir):
            # Read current timeseries
            time, data = read_sound_timeseries(sound_file)
            X.append(time), Y.append(data), labels.append(cur_label)
        cur_label += 1
    X, Y, labels = np.array(X), np.array(Y), np.array(labels)
    save_data(X, Y, labels, output_dir)

## Synthetic data ##
def func_factory(coef, arg):
    def func(x):
        return coef[0] * np.sin(x * arg[0] * np.pi) + coef[1] * np.cos(x * arg[1] * np.pi) +  coef[2] * np.sin(x * arg[2] * np.pi) 
    return func

def generate_synthetic_data(funcs, output_dir, num_samples_all=[20, 20, 20], seq_len=10, sigma=0.1):
    num_type = num_samples_all.shape[0]
    num_samples = np.sum(num_samples_all)
    base_X = np.linspace(0, 1, seq_len)
    X = np.tile(base_X, (num_samples, 1))

    Y = np.random.normal(size=num_samples_all[0]) * sigma
    labels = np.zeros * num_samples_all[0]
    for l in range(1, num_type):
        Y = np.concatenate((Y,  funcs[l](base_X) + np.random.normal(size=num_samples_all[l]) * sigma), axis=0)

    save_data(X, Y, labels, output_dir)