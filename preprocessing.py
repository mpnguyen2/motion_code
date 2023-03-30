import numpy as np
from utils import *
import scipy.io.wavfile as wavfile
from sktime.datasets import load_UCR_UEA_dataset
from utils import plot_timeseries

# Color list for plotting
COLOR_LIST = ['red', 'blue', 'green', 'yellow', 'black', 'orange', 'brown', 'grey', 'purple', 'hotpink']

def clear():
    os.system('clear')

## Sound dataset ##
def read_sound_timeseries(file_name, down_sampling_rate=100):
    sample_rate, data = wavfile.read(file_name)
    duration = len(data)/sample_rate
    time = np.arange(0, 1, 1/(duration*sample_rate))
    intervals = np.array(np.arange(0, len(time), len(time)/down_sampling_rate), dtype=int)
    intervals = intervals[:down_sampling_rate]
    data = data[intervals]
    data = np.abs(data)/np.max(np.abs(data))
    return data

def generate_data_from_sound_dataset(input_dir):
    cur_label = 0
    Y, labels = [], []
    for single_dir in os.scandir(input_dir):
        if not single_dir.is_dir():
            continue
        for sound_file in os.scandir(single_dir):
            # Read current timeseries
            data = read_sound_timeseries(sound_file)
            Y.append(data) 
            labels.append(cur_label)
        cur_label += 1
    Y, labels = np.array(Y, dtype=float), np.array(labels, dtype=int)
    return Y.reshape(Y.shape[0], 1, Y.shape[1]), labels+1

## Synthetic data ##
def func_factory(coef, arg):
    def func(x):
        return coef[0] * np.sin(x * arg[0] * np.pi) + coef[1] * np.cos(x * arg[1] * np.pi) +  coef[2] * np.sin(x * arg[2] * np.pi) 
    return func

def generate_synthetic_data(num_samples=np.array([20, 20, 20]), seq_len=10, sigma=0.1): 
    base_X = np.linspace(0, 1, seq_len)

    func1 = func_factory([1.0, 0.3, 0.5], [3, 9, 7])
    func2 = func_factory([0.1, 1, -0.1], [1.5, 6, 7])
    func3 = func_factory([0.5, -1,  0.6], [4.5, 2.5, 9])
    funcs = [func1, func2, func3]

    Y = []; labels = []
    for l in range(0, 3):
        start_ind = 0 if l != 0 else 1
        for _ in range(start_ind, num_samples[l]):
            Y.append(funcs[l](base_X) + np.random.normal(size=seq_len) * sigma)
            labels.append(l)

    Y = np.array(Y); labels = np.array(labels)
    return Y.reshape(Y.shape[0], 1, Y.shape[1]), labels+1

# Load datasets: 
# a set of synthetic data and another set of audio data.
def load_data(name, split='train', add_noise=False):
    '''
    Returns time series data together with corresponding labels. 
    Note we are considering different motions or collections of time series.
    
    Parameters
    ----------
    name: Name of the data set
    split: either train or test data
    '''
    if name == 'Synthetic':
        if split == 'train':
            Y, labels = generate_synthetic_data(num_samples=[30, 30, 30], seq_len=500, sigma=0.1)
        elif split == 'test':
            Y, labels = generate_synthetic_data(num_samples=[20, 20, 20], seq_len=500, sigma=0.1)
    elif name == 'Sound':
        Y, labels = generate_data_from_sound_dataset(input_dir='data/sound')
    else:
        Y, labels= load_UCR_UEA_dataset(name=name, split=split, return_X_y=True, return_type="numpy3d")
    if add_noise:
        Y += np.random.normal(size=Y.shape) * 0.3 * np.max(np.abs(Y))
    return Y, labels

def add_time_variable(Y, labels, visualize=False):
    '''
    Add the time variable X. Reformat labels to zero-indexed. Optinally allow visualize data.
    '''
    try:
        labels = np.array(labels, dtype=int)
        labels_unique = np.sort(np.unique(labels))
        num_motion = labels_unique.shape[0]
        labels_to_indices = {}
        for k in range(num_motion):
            labels_to_indices[labels_unique[k]] = k
        for i in range(labels.shape[0]):
            labels[i] = labels_to_indices[labels[i]]
    except:
        return np.array([]), np.array([]), np.array([])
    Y = Y[:, 0, :]
    num_samples = Y.shape[0]; seq_len = Y.shape[1]
    X = np.tile(np.linspace(0, 1, seq_len), (num_samples, 1))

    # Optional visualization
    if visualize:
        plot_timeseries(X, Y, labels)

    return X, Y, labels