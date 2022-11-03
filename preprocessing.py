import pickle
import numpy as np
from utils import *
import argparse
import scipy.io.wavfile as wavfile

def clear():
    os.system('clear')

def save_data(prefix, index_to_motion, motion_to_index, motion_names, num_observations, X, Y):
    '''
    Save data to 4 different files:
    1. 3 files for motion to index dict, index to motion array, motion names array
    2. a main data file with X (x of all timeseries concatenated), Y (y of all timeseries concatenated), 
        & number of observations of each data. 
    '''
    # Saving look up dictionaries
    np.savetxt(prefix + '_index_to_motion.txt', index_to_motion, fmt='%s')
    with open(prefix + '_motion_to_index.pickle', 'wb') as handle:
        pickle.dump(motion_to_index, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Saving motion names for all motions/samples
    np.savetxt(prefix + '_motion_names.txt', motion_names, fmt='%s')
    
    # Saving main timeseries data: a big Y=f(X) and accumulated number of observations for these timeseries
    np.savez(prefix + '_data', X=X, Y=Y, num_observations=num_observations)    

class TimeSeriesProcessor:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.index_to_motion = []
        self.motion_names = []
        self.motion_to_index = {}
        self.num_observations = [0]
        self.observation_cnt = 0
        self.X = np.array([])
        self.Y = np.array([])

    def save_data(self):
        # Convert index_to_motion and motion_names to np array of strings
        self.index_to_motion = np.array(self.index_to_motion, dtype=str)
        self.motion_names = np.array(self.motion_names, dtype=str)
        print('Done processing data.\n')

        # General information
        print('General information:')
        print('Number of samples/motions: ', len(self.motion_names))
        print('Number of features in timeseries: ', self.Y.shape[1])

        # Save files: use npz for arrays, txt for str arrays, and pickle for dictionaries
        print('\nSaving data files...')
        save_data(self.output_dir, self.index_to_motion, self.motion_to_index, self.motion_names, self.num_observations, self.X, self.Y)
        print('Data files saved. Data processing done!')

    def add_timeseries(self, time, data, motion_name):
        self.observation_cnt += data.shape[0]
        self.num_observations.append(self.observation_cnt)
        # Update big data array
        if self.Y.shape[0] != 0:
            self.Y = np.concatenate((self.Y, data), axis=0)
            self.X = np.concatenate((self.X, time), axis=0) 
        else:
            self.Y = data
            self.X = time
        # Update motion name (of current timeseries)
        self.motion_names.append(motion_name)
        # Update reference dictionaries if new motion detected
        if motion_name not in self.motion_to_index:
            self.motion_to_index[motion_name] = len(self.index_to_motion)
            self.index_to_motion.append(motion_name)

## Auslan motion data ##
def read_timeseries(file_name):
    """
    Read timeseries and put them together if they are the same motion
    """
    f = open(file_name, mode='r')
    # Each line is a frame contain a fixed number of features
    data = []
    while (line := f.readline()):
        data.append(np.array(line.split(), dtype=float))

    return np.array(data, dtype=float)

def get_all_files_from(dir_name, file_format):
    """
    Return paths of all the files in the directory with given name and format
    """
    file_paths = []
    for file in os.scandir(dir_name):
        if file.is_dir():
            file_paths.extend(get_all_files_from(file.path, file_format))
        else:
            if file.path.endswith(file_format):
                file_paths.append((file.path, file.name))
    
    return file_paths

def extract_data_from_motion_dataset(input_dir, output_dir, file_format='tsd', max_num_motions=10):
    """
    Process a given time series dataset in a directory. motion data that includes:
        1. Dictionaries index to/from motion
        2. List of motion names for each timeseries
        3. data that includes all X-values and Y-values of all timeseries concatenated and
            num_observations array indicating number of observation in each timeseries.
        (X, Y) = (time, data at that time of time-series)

    Parameters
    ----------
    input_dir: Root directory to data files
    output_dir: Directory to save converted data
    file_format: file type for timeseries data
    max_num_motions: Maximum number of motions allowed
    time_step: rate of sampling. Default to 100 frames per second
    """
    processor = TimeSeriesProcessor(output_dir)
    # Find all data files (other than dirs) with given format
    file_paths = get_all_files_from(input_dir, file_format)

    # Iterate over each file and append timeseries to list of appropriate motion
    for file_path, file_name in tqdm(file_paths, leave=False):
        # Find motion names
        motion_name = file_name.split('-')[0]
        data = read_timeseries(file_path)
        time = np.linspace(0, 1, data.shape[0]).reshape(-1, 1)
        processor.add_timeseries(time, data, motion_name)

    processor.save_data()

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
    processor = TimeSeriesProcessor(output_dir)
    for single_dir in os.scandir(input_dir):
        if not single_dir.is_dir() or single_dir.name == 'processed':
            continue
        motion_name = single_dir.name
        for sound_file in os.scandir(single_dir):
            # Read current timeseries
            time, data = read_sound_timeseries(sound_file)
            processor.add_timeseries(time.reshape(-1, 1), data.reshape(-1, 1), motion_name)
    processor.save_data()

## Synthetic data ##
def func_factory(coef, arg):
    def func(x):
        return coef[0] * np.sin(x * arg[0] * np.pi) + coef[1] * np.cos(x * arg[1] * np.pi) +  coef[2] * np.sin(x * arg[2] * np.pi) 
    return func

def generate_gaussian_data(funcs, num_series_with_type, num_pts_per_series=10, sigma=0.1):
    assert(len(funcs) == len(num_series_with_type))
    num_type = len(num_series_with_type) # number of types
    N = num_pts_per_series # Number of points per series
    base_X = np.linspace(0, 1, num=num_pts_per_series) # base times
    motion_names = []; num_observations = [0]

    # Motion dictionary
    index_to_motion = np.array([('motion_' + str(l+1)) for l in range(num_type)], dtype=str)
    motion_to_index = {}
    for l in range(num_type):
        motion_to_index['motion_' + str(l+1)] = l

    # Generate data X, Y, and motion_names according to number of series needed each type
    X = None; Y = None
    for l in range(num_type):
        start_ind = 0
        if X is None:
            X = base_X
            Y = funcs[0](base_X) + np.random.normal(size=N) * sigma
            start_ind = 1
            motion_names.append(index_to_motion[0])
        for _ in range(start_ind, num_series_with_type[l]):
            X = np.concatenate((X, base_X), axis=0)
            Y = np.concatenate((Y,  funcs[l](base_X) + np.random.normal(size=N) * sigma), axis=0)
            motion_names.append(index_to_motion[l])
            num_observations.append(len(num_observations)*base_X.shape[0])
    num_observations.append(len(num_observations)*base_X.shape[0])
    num_observations = np.array(num_observations, dtype=int); motion_names = np.array(motion_names, dtype=str)
    X = X.reshape(-1, 1); Y = Y.reshape(-1, 1)

    return index_to_motion, motion_to_index, motion_names, num_observations, X, Y

def generate_synthetic_data(funcs, num_train, num_test, prefix, num_pts_per_series=10, sigma=0.1):
    # Generate training data
    print('\nGenerating artificial train data...')    
    index_to_motion, motion_to_index, motion_names, num_observations, X, Y = generate_gaussian_data(funcs, [num_train]*len(funcs), num_pts_per_series, sigma)
    print('Saving artificial train data...')
    save_data(prefix+'train', index_to_motion, motion_to_index, motion_names, num_observations, X, Y)
    print('Train data files saved.')

    # Generate testing data
    print('\nGenerating artificial test data...')    
    index_to_motion, motion_to_index, motion_names, num_observations, X, Y = generate_gaussian_data(funcs, [num_test]*len(funcs), num_pts_per_series, sigma)
    print('Saving artificial test data...')
    save_data(prefix+'test', index_to_motion, motion_to_index, motion_names, num_observations, X, Y)
    print('Test data files saved.')

## Main program ##
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CLI arguments')
    parser.add_argument('--mode', type=str, default='artificial', help='Whether to generate artificial or data from dataset')
    # Additional setting for artificial mode
    parser.add_argument('--num_pts', type=int, default=10, help='Number of data points each timeseries')
    parser.add_argument('--num_train', type=int, default=5, help='Number of training timeseries per type')
    parser.add_argument('--num_test', type=int, default=1, help='Number of testing timeseries per type')
    parser.add_argument('--sigma', type=float, default=0.1, help='Sigma, or variance in noise added to original series')
    args = parser.parse_args()

    if args.mode == 'artificial':
        # We choose a fix set of synthetic data's generate functions
        func1 = func_factory([1.0, 0.3, 0.5], [3, 9, 7])
        func2 = func_factory([0.1, 1, -0.1], [1.5, 6, 7])
        func3 = func_factory([0.5, -1,  0.6], [4.5, 2.5, 9])
        funcs = [func1, func2, func3]
        generate_synthetic_data(funcs, num_train=args.num_train, num_test=args.num_test, 
            prefix='data/artificial/', num_pts_per_series = args.num_pts, sigma=args.sigma)
    elif args.mode == 'auslan':
        # Rate: 100 frames per second
        # time_step = 1e-2 
        # Number of maximum allowed motion
        max_num_motions = 2
        # Process training data
        main_dir = 'data/auslan'
        print('Process auslan motion training data...')
        extract_data_from_motion_dataset(input_dir=main_dir+'/train', output_dir=main_dir+'/processed/train', max_num_motions=max_num_motions)
        print('Done processing auslan motion training data.\n')
        print('--------------------------------------------')
        # Process first test set
        print('Process test data...')   
        extract_data_from_motion_dataset(input_dir=main_dir+'/test1', output_dir=main_dir+'/processed/test', max_num_motions=max_num_motions)
        print('Done processing test data.\n')
        print('--------------------------------------------')

    elif args.mode == 'sound':
        main_dir = 'data/sound'
        print('Process pronunciation sound training data...')
        extract_data_from_sound_dataset(input_dir=main_dir, output_dir=main_dir+'/processed/train')
        print('Done processing pronunciation sound training data.\n')
        print('--------------------------------------------')
        # Process first test set
        print('Process test data...')   
        extract_data_from_sound_dataset(input_dir=main_dir, output_dir=main_dir+'/processed/test')
        print('Done processing test data.\n')
        print('--------------------------------------------')