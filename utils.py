import os, pickle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Color list for plotting
COLOR_LIST = ['red', 'blue', 'green', 'yellow', 'black', 'orange', 'brown', 'grey', 'purple', 'hotpink']

## Load timeseries data utilities ##
def load_data(prefix):
    """
    Load data from a given directory including condensed files (npz and small txt and pickle)
    """

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

def find_min_max_from_data_list(l):
    cur_min = np.min(l[0])
    cur_max = np.max(l[0])
    for e in l:
        cur_min = min(cur_min, np.min(e))
        cur_max = max(cur_max, np.max(e))
    return cur_min, cur_max

## Metric ##
def precision(pred, gt):
    """
    Return precision metric from testing
    
    Parameters
    ----------
    pred and gt: lists of str
    """
    return np.sum(np.array(pred)==np.array(gt))/len(pred)

def F_score(pred, gt, num_class):
    """
    Multi-class F-score: TBA
    """
    pass

## Visualization ##
def plot_timeseries(index_to_motion, X_list, y_list, indices, explore=True, X_ms=np.array([])):
    num_series = len(y_list)
    for i in range(num_series):
        plt.plot(X_list[i], y_list[i], c=COLOR_LIST[indices[i]], lw=0.5)
    plt.title('Time series')
    patches = []
    L = len(index_to_motion)
    for k in range(L):
        patches.append(mpatches.Patch(color=COLOR_LIST[k], label=index_to_motion[k]))
    plt.legend(handles=patches)
    if not explore:
        for k in range(L):
            plt.scatter(X_ms[k].reshape(-1), np.zeros(X_ms[k].shape[0]), color=COLOR_LIST[k])
    if explore:
        plt.savefig('out/plot.png')
    else:
        plt.savefig('out/inducing_pts.png')

def plot_motion_types(index_to_motion, X_lists, Y_lists, X_test, means, covars, coord=0):
    L = len(X_lists)
    for k in range(L):
        num_series = len(X_lists[k])
        for i in range(num_series):
            plt.plot(X_lists[k][i], Y_lists[k][i][:, coord], c=COLOR_LIST[k], lw=0.5)
        std = np.sqrt(np.diag(covars[k])).reshape(-1)
        mean = means[k].reshape(-1)
        X_test = X_test.reshape(-1)
        plt.plot(X_test, mean, c=COLOR_LIST[(k+1)%len(X_lists)], lw=0.5)
        plt.fill_between(X_test, mean+2*std, mean-2*std,
            color=COLOR_LIST[(k+2)%len(X_lists)], alpha=0.1)
        #plt.scatter(X_ms[k].reshape(-1), np.zeros(X_ms[k].shape[0]), color=colors[k])
        plt.legend(handles=[mpatches.Patch(color=COLOR_LIST[k], label=index_to_motion[k])])
        plt.savefig('out/multiple/' + index_to_motion[k] + '.png')
        plt.clf()