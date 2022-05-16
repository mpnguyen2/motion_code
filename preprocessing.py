import pickle
import numpy as np
from utils import *
import argparse

def clear():
    os.system( 'clear' )

# Saving data: 
# we need 3 files for motion to index dict, index to motion array, motion names array,
# And a main data file with X (x of all timeseries concatenated), Y (y of all timeseries concatenated), 
# & number of observations of each data. 
def save_data(prefix, index_to_motion, motion_to_index, motion_names, num_observations, X, Y):
    # Saving look up dictionaries
    np.savetxt(prefix + '_index_to_motion.txt', index_to_motion, fmt='%s')
    with open(prefix + '_motion_to_index.pickle', 'wb') as handle:
        pickle.dump(motion_to_index, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # Saving motion names for all motions/samples
    np.savetxt(prefix + '_motion_names.txt', motion_names, fmt='%s')
    # Saving main timeseries data: big Y=f(X) and accumulated number of observations for these timeseries
    np.savez(prefix + '_data', X=X, Y=Y, num_observations=num_observations)    

def process_data(dir_name, prefix, file_format='tsd', max_num_motions=10):
    '''
    dir_name: Root directory to data files
    prefix: Directory prefix to save converted data
    file_format: file type for timeseries data
    max_num_motions: Maximum number of motions allowed
    time_step: rate of sampling. Default to 100 frames per second
    '''
    print('Loading data...')
    # Load data and get motion names for each timeseries, together with motion name to index & index to name look-up dictionary
    index_to_motion, motion_to_index, motion_names, Y, num_observations = \
        get_motion_data_from_dir(dir_name=dir_name, file_format=file_format, max_num_motions=max_num_motions)
    print('Loaded data. Preprocessing data...')
    # Create time X arrays: 2 big arrays for Y=f(X) time-series data
    X = np.array([])
    for i in range(len(num_observations)-1):
        current_time_var = get_time_variables(Y[num_observations[i]:num_observations[i+1]])
        if X.shape[0] != 0:
            X = np.concatenate((X, current_time_var), axis=0)
        else:
            X = current_time_var
    # Convert index_to_motion and motion_names to np array of strings
    index_to_motion = np.array(index_to_motion, dtype=str)
    motion_names = np.array(motion_names, dtype=str)
    # Store list of timeseries data into 3 arrays X, Y (), and num_observations
    # Store motion names into numpy txt file, and motion name to index look-up dictionary to pickle file
    print('Done processing data.\n')
    # General information
    print('General information:')
    print('Number of samples/motions: ', len(motion_names))
    #print('First motion types: ', index_to_motion[:4]) 
    print('Number of features in timeseries: ', Y.shape[1])
    # Save files: use npz for arrays, txt for str arrays, and pickle for dictionaries
    print('\nSaving data files...')
    save_data(prefix, index_to_motion, motion_to_index, motion_names, num_observations, X, Y)
    print('Data files saved. Data processing done!')

# Functions seed for generating synthetic data
def func1(x):
    return 1.0 * np.sin(x * 3 * np.pi) + 0.3 * np.cos(x * 9 * np.pi) +  0.5 * np.sin(x * 7 * np.pi) 
def func2(x):
    return 0.1 * np.sin(x * 1.5 * np.pi) + 1 * np.cos(x * 6 * np.pi) - 0.1 * np.sin(x * 7 * np.pi) 
def func3(x):
    return 0.5 * np.sin(x * 4.5 * np.pi) - 1 * np.cos(x * 2.5 * np.pi) + 0.6 * np.sin(x * 9 * np.pi) 

def generate_gaussian_data(funcs, num_series_with_type, num_pts_per_series=10, sigma=0.1):
    assert(len(funcs) == len(num_series_with_type))
    num_type = len(num_series_with_type) # number of types
    N = num_pts_per_series # Number of points per series
    base_X = np.linspace(0, 1, num=num_pts_per_series) # base times
    motion_names = []; num_observations = [0]
    # motion dictionary
    index_to_motion = np.array([('motion_' + str(l+1)) for l in range(num_type)], dtype=str)
    motion_to_index = {}
    for l in range(num_type):
        motion_to_index['motion_' + str(l+1)] = l
    # Initialize X and Y
    X = None; Y = None
    # Generate data X, Y, and motion_names according to number of series needed each type
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
    print('Generating artificial train data...')    
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
        funcs = [func1, func2, func3]
        generate_synthetic_data(funcs, num_train=args.num_train, num_test=args.num_test, 
            prefix='data/artificial/', num_pts_per_series = args.num_pts, sigma=args.sigma)
    else:
        # Rate: 100 frames per second
        # time_step = 1e-2 
        # Number of maximum allowed motion
        max_num_motions = 12
        # Process training data
        main_dir = 'data/auslan'
        print('Process training data...')
        process_data(dir_name=main_dir+'/train', prefix=main_dir+'/processed/train', max_num_motions=max_num_motions)
        print('Done processing training data.\n')
        print('--------------------------------------------')
        # Process first test set
        print('Process test 1 data...')   
        process_data(dir_name=main_dir+'/test1', prefix=main_dir+'/processed/test1', max_num_motions=max_num_motions)
        print('Done processing test 1 data.\n')
        print('--------------------------------------------')
        # Process second test set
        print('Process test 2 data...')
        process_data(dir_name=main_dir+'/test2', prefix=main_dir+'/processed/test2', max_num_motions=max_num_motions)
        print('Done processing test 2 data.')