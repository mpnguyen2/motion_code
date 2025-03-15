import numpy as np
import os
from scipy.interpolate import interp1d, UnivariateSpline, Rbf
import scipy.io.wavfile as wavfile
from sktime.datasets import load_UCR_UEA_dataset
from parkinson_data_processing import get_parkinson_train_test_data


def clear():
    os.system('clear')


################################ Proununciation dataset ################################
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


################################ Parkinson disease's sensor data ################################
def _interp(xs, ys, xn, mode='np'):
    if mode == 'np':
        return np.interp(xn, xs, ys)
    else:
        if mode == 'scipy_linear':
            interp_func = interp1d(xs, ys, fill_value='extrapolate')
        elif mode == 'scipy_spline':
            interp_func = UnivariateSpline(xs, ys)
        else:
            interp_func = Rbf(xs, ys)
        return interp_func(xn)

def _prepare_equal_length(X, Y, xn):
    Y_list = []
    for xs, ys in zip(X, Y):
        Y_list.append(_interp(xs, ys, xn))
    return np.array(Y_list, dtype=float)

def interp_parkison_data_for_benchmarking_algorithms(X_train, Y_train, labels_train,
                                                     X_test, Y_test, labels_test):
    xn = np.linspace(0, 1, 1600)
    Y1_train = _prepare_equal_length(X_train, Y_train, xn)
    Y1_test = _prepare_equal_length(X_test, Y_test, xn)

    return Y1_train, np.array(labels_train, dtype=int),\
           Y1_test, np.array(labels_test, dtype=int)


################################ Load/process data for non-Parkinson data ################################
def load_data(name, split='train', add_noise=False):
    '''
    Returns time series data together with corresponding labels. 
    Note we are considering different motions or collections of time series.
    
    Parameters
    ----------
    name: Name of the data set
    split: either train or test data
    '''
    if name == 'Pronunciation Audio':
        Y, labels = generate_data_from_sound_dataset(input_dir='data/audio')
    else:
        Y, labels= load_UCR_UEA_dataset(name=name, split=split, return_X_y=True, return_type="numpy3d")
    if add_noise:
        Y += np.random.normal(size=Y.shape) * 0.3 * np.max(np.abs(Y))
    return Y, labels

def process_data(Y, labels):
    '''
    Simple data processing for collections of time series.
    Make Y 2d array and normalize labels to [0..L-1], where L is the number of labels.
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
        return np.array([]), np.array([])
    
    return Y[:, 0, :], labels

def add_time_variable(Y, labels, visualize=False):
    '''
    Add the time variable X
    '''
    if len(labels) == 0:
        return np.array([]), np.array([]), np.array([])

    num_samples = Y.shape[0]; seq_len = Y.shape[1]
    X = np.tile(np.linspace(0, 1, seq_len), (num_samples, 1))
    return X, Y, labels

def process_data_for_motion_codes(Y, labels):
    '''
    Data processing specifically for MotionCodes as the algorithm also need a generated time variable X.
    '''
    Y, labels = process_data(Y, labels)
    return add_time_variable(Y, labels)

def split_train_test_forecasting(Y, percentage):
    '''
    Split train and test sets for forecasting.
    '''
    seq_length = Y.shape[1]
    train_num_steps = int(percentage*seq_length)
    test_num_steps = seq_length - train_num_steps
    
    return Y[:, :train_num_steps], Y[:, train_num_steps:], train_num_steps, test_num_steps


################################ Single convenient fcts to get data for all algorithms ################################
def get_train_test_data_forecast(name):
    Y, labels = load_data(name, split='train')
    Y, labels = process_data(Y, labels)
    Y_train, Y_test, train_num_steps, test_num_steps = (
        split_train_test_forecasting(Y, percentage=0.8)
    )
    benchmark_data = (Y_train, Y_test, labels, test_num_steps)

    # Data for motion code requires the additional X variable
    X, _, _ = add_time_variable(Y, labels)
    X_train = X[:, :train_num_steps]
    test_time_horizon = X[0, train_num_steps:]
    motion_code_data = (X_train, Y_train, labels, test_time_horizon, Y_test)

    return benchmark_data, motion_code_data

def get_train_test_data_classify(name, load_existing_data, add_noise=True):
    benchmark_data, motion_code_data = None, None
    if name != 'PD setting 1' and name != 'PD setting 2':
        data_path = 'data/basics/' + name
        if name == 'Pronunciation Audio':
            data_path = 'data/audio/' + name
        if load_existing_data:
            data = np.load(data_path + '.npy', allow_pickle=True).item()
            Y_train_bm, labels_train_bm = data.get('Y_train'), data.get('labels_train')
            Y_test_bm, labels_test_bm = data.get('Y_test'), data.get('labels_test')                
        else:
            Y_train_bm, labels_train_bm = load_data(name, split='train', add_noise=add_noise)
            Y_test_bm, labels_test_bm = load_data(name, split='test', add_noise=add_noise)
            
        benchmark_data = (Y_train_bm, labels_train_bm, Y_test_bm, labels_test_bm)
        X_train, Y_train, labels_train = process_data_for_motion_codes(Y_train_bm, labels_train_bm)
        X_test, Y_test, labels_test = process_data_for_motion_codes(Y_test_bm, labels_test_bm)
        motion_code_data = (X_train, Y_train, labels_train, X_test, Y_test, labels_test)
        
    if name == 'PD setting 1' or name == 'PD setting 2':
        X_train, Y_train, labels_train, X_test, Y_test, labels_test = (
            get_parkinson_train_test_data(name)
        )
        motion_code_data = (X_train, Y_train, labels_train, X_test, Y_test, labels_test)
    
        Y_train_bm, labels_train_bm, Y_test_bm, labels_test_bm = (
            interp_parkison_data_for_benchmarking_algorithms(X_train, Y_train, labels_train,
                                                                X_test, Y_test, labels_test)
        )
        benchmark_data = (Y_train_bm, labels_train_bm, Y_test_bm, labels_test_bm)

    return benchmark_data, motion_code_data


################################ Get audio data for notebook experiment ################################
def randomly_remove_data_points(X, Y, low_percent=0.8, high_percent=0.9):
    X_removed = []
    Y_removed = []
    num_series = len(X)
    for s in range(num_series):
        series_len = X[s].shape[0]
        num_remained = np.random.randint(low=int(low_percent*series_len), 
                                         high=int(high_percent*series_len))
        idx = np.sort(np.random.choice(np.arange(series_len), size=num_remained, replace=False))
        X_removed.append(np.copy(X[s][idx]))
        Y_removed.append(np.copy(Y[s][idx]))
    
    return X_removed, Y_removed


def get_pronunciation_audio_data_unequal_lengths():
    Y_train_orig, labels_train = load_data('Pronunciation Audio', split='train')
    Y_test, labels_test = load_data('Pronunciation Audio', split='test')
    X_train_orig, Y_train_orig, labels_train = process_data_for_motion_codes(Y_train_orig, labels_train)
    X_test, Y_test, labels_test = process_data_for_motion_codes(Y_test, labels_test)    
    X_train_orig, X_test, Y_train_orig, Y_test =\
        list(X_train_orig), list(X_test), list(Y_train_orig), list(Y_test)
    np.random.seed(seed=41)
    X_train, Y_train = randomly_remove_data_points(X_train_orig,
                                                   Y_train_orig,
                                                   low_percent=0.8,
                                                   high_percent=0.95)
    return X_train, Y_train, labels_train, X_test, Y_test, labels_test