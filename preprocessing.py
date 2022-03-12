import pickle
import numpy as np
from utils import *

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
    print('First 4 motion types: ', index_to_motion[:4]) 
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

if __name__ == '__main__':
    # Rate: 100 frames per second
    #time_step = 1e-2 
    # Number of maximum allowed motion
    max_num_motions = 10
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