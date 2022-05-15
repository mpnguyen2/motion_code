import os, pickle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

### Read timeseries and put them together if they are the same motion for model training ###
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
        
### Load data from condensed files (npz and small txt and pickle) ###
def load_data(prefix):
    # Load look-up dictionaries and array of motion names
    motion_to_index = pickle.load(open(prefix + '_motion_to_index.pickle', 'rb'))
    index_to_motion = np.loadtxt(prefix+'_index_to_motion.txt', dtype=str, delimiter=' ')
    motion_names = np.loadtxt(prefix+'_motion_names.txt', dtype=str, delimiter=' ')
    # Load main timeseries data and divide them into different samples of timeseries
    # Also record the appropriate indices correponding to their motion names for later training
    data = np.load(prefix+'_data.npz')
    X = data['X']; Y = data['Y']
    num_observations = data['num_observations']
    X_list = []; Y_list = []; indices = []
    for i in range(len(num_observations)-1):
        X_list.append(X[num_observations[i]:num_observations[i+1]])
        Y_list.append(Y[num_observations[i]:num_observations[i+1]])
        indices.append(motion_to_index[motion_names[i]])
   
    return index_to_motion, motion_names, X_list, Y_list, indices


## Metric
# pred and gt are both lists of str
def precision(pred, gt):
    return np.sum(np.array(pred)==np.array(gt))/len(pred)

# Multi-class F-score: TBA
def F_score(pred, gt, num_class):
    pass

## Visualization
def plot_timeseries(index_to_motion, X_list, y_list, indices, colors, explore=True, X_ms=np.array([])):
    num_series = len(y_list)
    for i in range(num_series):
        plt.plot(X_list[i], y_list[i], c=colors[indices[i]], lw=0.5)
    plt.title('Time series')
    patches = []
    L = len(index_to_motion)
    for k in range(L):
        patches.append(mpatches.Patch(color=colors[k], label=index_to_motion[k]))
    plt.legend(handles=patches)
    if not explore:
        for k in range(L):
            plt.scatter(X_ms[k].reshape(-1), np.zeros(X_ms[k].shape[0]), color=colors[k])
    if explore:
        plt.savefig('out/plot.png')
    else:
        plt.savefig('out/inducing_pts.png')

def plot_motion_types(index_to_motion, X_lists, y_lists, X_test, means, covars, colors):
    L = len(X_lists)
    for k in range(L):
        num_series = len(X_lists[k])
        for i in range(num_series):
            plt.plot(X_lists[k][i], y_lists[k][i], c=colors[k], lw=0.5)
        std = np.sqrt(np.diag(covars[k])).reshape(-1)
        mean = means[k].reshape(-1)
        X_test = X_test.reshape(-1)
        plt.plot(X_test, mean, c=colors[(k+1)%len(X_lists)], lw=0.5)
        plt.fill_between(X_test, mean+2*std, mean-2*std,
            color=colors[(k+2)%len(X_lists)], alpha=0.1)
        #plt.scatter(X_ms[k].reshape(-1), np.zeros(X_ms[k].shape[0]), color=colors[k])
        plt.legend(handles=[mpatches.Patch(color=colors[k], label=index_to_motion[k])])
        plt.savefig('out/multiple/' + index_to_motion[k] + '.png')
        plt.clf()