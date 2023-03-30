import os, pickle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Color list for plotting
COLOR_LIST = ['red', 'green', 'black', 'blue', 'yellow', 'orange', 'brown', 'grey', 'purple', 'hotpink']

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

## Metric ##
def accuracy(pred, gt):
    """
    Return accuracy/precision metric from testing
    
    Parameters
    ----------
    pred and gt: lists of str
    """
    return np.sum(np.array(pred)==np.array(gt))/len(pred)

def RMSE(pred, gt):
    return np.sqrt(np.mean((pred-gt)**2))

## Visualization ##
def plot_timeseries(X_list, y_list, labels, output_dir='out/plot.png', label_names=[]):
    # Plot timeseries
    num_series = len(y_list)
    for i in range(num_series):
        plt.plot(X_list[i], y_list[i], c=COLOR_LIST[labels[i]], lw=0.5)
    plt.title('Time series')
    # Draw legend
    patches = []
    L = len(np.unique(labels))
    if len(label_names) == 0:
        label_names = [str(i) for i in range(L)]
    for k in range(L):
        patches.append(mpatches.Patch(color=COLOR_LIST[k], label=label_names[k]))
    plt.legend(handles=patches)
    plt.savefig(output_dir)

def plot_motion_types(X_train, Y_train, X_test, labels, means, covars, X_m_ks, label_names):
    L = np.unique(labels).shape[0]
    if len(label_names) == 0:
        label_names = [str(i) for i in range(L)]
    X_lists = [[] for _ in range(L)]
    Y_lists = [[] for _ in range(L)]
    for i in range(X_train.shape[0]):
        X_lists[labels[i]].append(X_train[i])
        Y_lists[labels[i]].append(Y_train[i])
    min_Y = 1e9
    for k in range(L):
        X_lists[k] = np.array(X_lists[k])
        Y_lists[k] = np.array(Y_lists[k])
        num_series = len(X_lists[k])
        for i in range(num_series):
            min_Y = min(min_Y, np.min(Y_lists[k][i]))
            plt.plot(X_lists[k][i], Y_lists[k][i], c=COLOR_LIST[k], lw=0.5, zorder=1)
        std = np.sqrt(np.diag(covars[k])).reshape(-1)
        mean = means[k].reshape(-1)
        min_Y = min(min_Y, np.min(mean))
        X_test = X_test.reshape(-1)
        color = COLOR_LIST[(k+1)%len(X_lists)]
        plt.plot(X_test, mean, c=color, lw=0.5, zorder=1)
        plt.fill_between(X_test, mean+2*std, mean-2*std,
            color=COLOR_LIST[(k+2)%len(X_lists)], alpha=0.1, zorder=1)
        min_Y -= abs(min_Y)
        Y_test = np.interp(X_m_ks[k], X_lists[k][0], np.mean(Y_lists[k], axis=0))
        plt.scatter(X_m_ks[k], Y_test, color=color, s=20, zorder=2)
        plt.legend(handles=[mpatches.Patch(color=COLOR_LIST[k], label=label_names[k])])
        plt.savefig('out/multiple/' + str(k) + '.png')
        plt.clf()