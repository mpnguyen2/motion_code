import os, pickle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Color list for plotting
COLOR_LIST = ['red', 'blue', 'green', 'yellow', 'black', 'orange', 'brown', 'grey', 'purple', 'hotpink']

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

def find_min_max_from_data_list(l):
    cur_min = np.min(l[0])
    cur_max = np.max(l[0])
    for e in l:
        cur_min = min(cur_min, np.min(e))
        cur_max = max(cur_max, np.max(e))
    return cur_min, cur_max

## Metric ##
def accuracy(pred, gt):
    """
    Return accuracy/precision metric from testing
    
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
def plot_timeseries(X_list, y_list, labels, explore=True, X_ms=np.array([])):
    # Plot timeseries
    num_series = len(y_list)
    for i in range(num_series):
        plt.plot(X_list[i], y_list[i], c=COLOR_LIST[labels[i]], lw=0.5)
    plt.title('Time series')
    # Draw legend
    patches = []
    labels_unique = np.unique(labels)
    L = len(labels_unique)
    for k in range(L):
        patches.append(mpatches.Patch(color=COLOR_LIST[k], label=labels_unique[k]))
    plt.legend(handles=patches)
    plt.savefig('out/plot.png')

def plot_motion_types(X_lists, Y_lists, X_test, means, covars, X_m_ks, coord=0):
    L = len(X_lists)
    min_Y = 1e9
    for k in range(L):
        num_series = len(X_lists[k])
        for i in range(num_series):
            min_Y = min(min_Y, np.min(Y_lists[k][i][:, coord]))
            plt.plot(X_lists[k][i], Y_lists[k][i][:, coord], c=COLOR_LIST[k], lw=0.5)
        std = np.sqrt(np.diag(covars[k])).reshape(-1)
        mean = means[k].reshape(-1)
        X_test = X_test.reshape(-1)
        color = COLOR_LIST[(k+1)%len(X_lists)]
        plt.plot(X_test, mean, c=color, lw=0.5)
        plt.fill_between(X_test, mean+2*std, mean-2*std,
            color=COLOR_LIST[(k+2)%len(X_lists)], alpha=0.1)
        plt.scatter(X_m_ks[k].reshape(-1), min_Y * np.ones(X_m_ks[k].shape[0]), color=color)
        plt.legend(handles=[mpatches.Patch(color=COLOR_LIST[k], label=str(k))])
        plt.savefig('out/multiple/' + str(k) + '.png')
        plt.clf()