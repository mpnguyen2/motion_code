import os
import numpy as np
from tqdm import tqdm

def read_timeseries(file_name, mode='plain'):
    f = open(file_name, mode='r')
    # Each line is a frame contain a fixed number of features
    data = []
    while (line := f.readline()):
        data.append(np.array(line.split(), dtype=float))

    return np.array(data, dtype=float)

# Get paths of all the files in the directory with name dir_name with the given format file_format
def get_all_files_from(dir_name, file_format):
    file_paths = []
    for file in os.scandir(dir_name):
        if file.is_dir():
            file_paths.extend(get_all_files_from(file.path, file_format))
        else:
            if file.path.endswith(file_format):
                file_paths.append((file.path, file.name))
    
    return file_paths

# Get all motion data from given directory. The maximum number of motions is limited
# The return info include dictionaries index to/from motion, a list of motion names for each timeseries
# The main data include all Y-values of all timeseries concatenated
# And an array of num_observations indicating number of observation in each timeseries
# This information can be used to separate these timeseries from big data array later
def get_motion_data_from_dir(dir_name, file_format='tsd', mode='plain', max_num_motions=10):
    index_to_motion = []
    motion_names = []
    motion_to_index = {}
    num_observations = [0]
    # Find all data files (other than dirs) with given format
    file_paths = get_all_files_from(dir_name, file_format)
    observation_cnt = 0; motion_cnt = 0
    data = np.array([])
    # Iterate over each file and append timeseries to list of appropriate motion
    for file_path, file_name in tqdm(file_paths, leave=False):
        # Find motion names
        motion_name = file_name.split('-')[0]
        if motion_name in motion_to_index or motion_cnt < max_num_motions:
            # Read current timeseries
            cur_timeseries = read_timeseries(file_path, mode=mode)
            # Update observation cnt array
            observation_cnt += cur_timeseries.shape[0]; num_observations.append(observation_cnt)
            # Update big data array
            if data.shape[0] != 0:
                data = np.concatenate((data, cur_timeseries), axis=0)
            else:
                data = cur_timeseries
            # Update motion name (of current timeseries)
            motion_names.append(motion_name)
        # Update reference dictionaries if new motion detected
        if motion_name not in motion_to_index and motion_cnt < max_num_motions:
            index_to_motion.append(motion_name)
            motion_to_index[motion_name] = motion_cnt
            motion_cnt += 1
    
    return index_to_motion, motion_to_index, motion_names, data, np.array(num_observations, dtype=int)

def combine_timeseries(ts_list, time_step):
    '''
    ts_list: list of timeseries, where each is a 2D array with 1st dim being frame number, and 2nd being feature dim
    '''
    ts0_len = ts_list[0].shape[0]
    X = np.linspace(0, time_step*(ts0_len-1), ts0_len)
    Y = ts_list[0]
    for i in range(1, len(ts_list)):
        tsi_len = ts_list[i].shape[0]
        X = np.concatenate((X, np.linspace(0, time_step*(tsi_len-1), tsi_len)))
        Y = np.concatenate((Y, ts_list[i]), axis=0)

    return X, Y

def get_time_variables(Y):
    len_series = Y.shape[0]
    return np.linspace(0, 1, len_series).reshape(-1, 1)