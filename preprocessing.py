import pickle
import numpy as np
from utils import *
import argparse

def clear():
    os.system( 'clear' )

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
    # Saving look up dictionaries
    np.savetxt(prefix + '_index_to_motion.txt', index_to_motion, fmt='%s')
    with open(prefix + '_motion_to_index.pickle', 'wb') as handle:
        pickle.dump(motion_to_index, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # Saving motion names for all motions/samples
    np.savetxt(prefix + '_motion_names.txt', motion_names, fmt='%s')
    # Saving main timeseries data: big Y=f(X) and accumulated number of observations for these timeseries
    np.savez(prefix + '_data', X=X, Y=Y, num_observations=num_observations)    
    print('Data files saved. Data processing done!')

def func1(x):
    return 1.0 * np.sin(x * 3 * np.pi) + 0.3 * np.cos(x * 9 * np.pi) +  0.5 * np.sin(x * 7 * np.pi) 

def func2(x):
    return 0.1 * np.sin(x * 1.5 * np.pi) + 1 * np.cos(x * 6 * np.pi) - 0.1 * np.sin(x * 7 * np.pi) 

def generate_synthetic_data(prefix, num_points=10, num_sin_series=1, num_cos_series=1, sigma=0.1):
    # To ensure data read/write format consistency, 
    # we need 3 files for motion to index dict, index to motion array, and motion names array
    # Finally a main data file with X (x of all timeseries concatenated), Y (y of all timeseries concatenated), 
    # and number of observations of each data. 
    print('Generating artificial data...')
    motion_names = []; num_observations = [0]
    # base times
    base_X = np.linspace(0, 1, num=num_points)
    # motion dictionary
    index_to_motion = np.array(['first', 'second'], dtype=str); motion_to_index = {'first': 0, 'second': 1}
    # Initialize
    X = base_X
    Y = func1(base_X) + np.random.normal(size=num_points) * sigma
    motion_names.append(index_to_motion[0])
    # Generate data X, Y, and motion_names according to number of series needed each type
    for _ in range(1, num_sin_series):
        X = np.concatenate((X, base_X), axis=0)
        Y = np.concatenate((Y,  func1(base_X) + np.random.normal(size=num_points) * sigma), axis=0)
        motion_names.append(index_to_motion[0])
        num_observations.append(len(num_observations)*base_X.shape[0])
    for _ in range(num_cos_series):
        X = np.concatenate((X, base_X), axis=0)
        Y = np.concatenate((Y, func2(base_X) + np.random.normal(size=num_points) * sigma), axis=0)
        motion_names.append(index_to_motion[1])
        num_observations.append(len(num_observations)*base_X.shape[0])
    num_observations.append(len(num_observations)*base_X.shape[0])
    num_observations = np.array(num_observations, dtype=int); motion_names = np.array(motion_names, dtype=str)
    X = X.reshape(-1, 1); Y = Y.reshape(-1, 1)
    # Saving data
    print('\nSaving artificial data...')
    # Saving look up dictionaries
    np.savetxt(prefix + '_index_to_motion.txt', index_to_motion, fmt='%s')
    with open(prefix + '_motion_to_index.pickle', 'wb') as handle:
        pickle.dump(motion_to_index, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # Saving motion names for all motions/samples
    np.savetxt(prefix + '_motion_names.txt', motion_names, fmt='%s')
    # Saving main timeseries data: big Y=f(X) and accumulated number of observations for these timeseries
    np.savez(prefix + '_data', X=X, Y=Y, num_observations=num_observations)    
    print('Data files saved. Data processing done!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CLI arguments')
    parser.add_argument('--mode', type=str, default='artificial', help='Whether to generate artificial or data from dataset')
    # Additional setting for artificial mode
    parser.add_argument('--num_points_per_series', type=int, default=10, help='Number of data points each timeseries')
    parser.add_argument('--num_sin_series', type=int, default=1, help='Number of first type of timeseries')
    parser.add_argument('--num_cos_series', type=int, default=1, help='Number of second type of timeseries')
    parser.add_argument('--sigma', type=float, default=0.1, help='Sigma, or variance in noise added to original series')
    args = parser.parse_args()
    if args.mode == 'artificial':
        generate_synthetic_data(prefix='data/artificial/artificial', num_points = args.num_points_per_series,
        num_sin_series=args.num_sin_series, num_cos_series=args.num_cos_series, sigma=args.sigma)
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