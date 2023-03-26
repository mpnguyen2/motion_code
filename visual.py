from preprocessing import load_UCR_UEA_data
import time, argparse
import numpy as np
import jax.numpy as jnp
from scipy.optimize import minimize
import scipy

#from jax.scipy.optimize import minimize

from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from dtaidistance import dtw

from sparse_gp import *
from utils import *

# Color list for plotting
COLOR_LIST = ['red', 'blue', 'green', 'yellow', 'black', 'orange', 'brown', 'grey', 'purple', 'hotpink']
markers = ["." , "," , "o" , "v" , "^" , "<", ">"]

def visualize_data_by_GP(name, m=10, Q=1):
    _, X_train, Y_train, train_labels = load_UCR_UEA_data(name, mode='train', visualize=False)
    _, X_test, Y_test, test_labels = load_UCR_UEA_data(name, mode='test', visualize=False)
    # X_test = X_test[:min(X_test.shape[0], 20)]
    # Y_test = Y_test[:min(Y_test.shape[0], 20)]
    # test_labels = test_labels[:min(test_labels.shape[0], 20)]
    num_motion = np.unique(train_labels).shape[0]
    print('Loaded dataset ' + name)

    # Initialize parameters
    X_m_start = sigmoid_inv(np.linspace(0.1, 0.9, m))
    Sigma_start = softplus_inv(np.ones(Q))
    W_start = softplus_inv(np.ones(Q))

    X_m_list = [[] for _ in range(2*num_motion)]
    cnt = 0
    dims = (m, Q)
    for i in range(X_train.shape[0] + X_test.shape[0]):
        if i < X_train.shape[0]:
            X = X_train[i]; Y = Y_train[i]
        else:
            X = X_test[i-X_train.shape[0]]; Y = Y_test[i-X_train.shape[0]]
            
        # Optimize X_m, and kernel parameters including Sigma, W
        res = minimize(fun=elbo_fn_single(X, Y, sigma_y=0.1, dims=dims),
                        x0 = pack_params([X_m_start, Sigma_start, W_start]),
                        method='L-BFGS-B', jac=True)
        # print('Inducing pts, motion codes, and kernel params successfully optimized: ', res.success)
        X_m, _, _ = unpack_params_single(res.x, dims=dims)
        if i < X_train.shape[0]:
            X_m_list[train_labels[i]].append(X_m)
        else:
            X_m_list[test_labels[i-X_train.shape[0]] + num_motion].append(X_m)
        if res.success:
            cnt += 1
        print('New iteration...')
    print('Total training samples:', X_train.shape[0])
    print('Total success training samples: ', cnt)

    for i in range(num_motion):
        if len(X_m_list) == 0:
            continue
        X_m_list[i] = np.array(X_m_list[i])
        # U, S, Vt
        U, S, _ = scipy.sparse.linalg.svds(X_m_list[i], k=2)
        reduced_X_m = U @ np.diag(S)
        color = COLOR_LIST[i]; marker = "o"           
        plt.scatter(reduced_X_m[:, 0], reduced_X_m[:, 1], c=color, marker=marker)
    plt.savefig('visual_train.png')

    for i in range(num_motion, 2*num_motion):
        if len(X_m_list[i]) == 0:
            continue
        X_m_list[i] = np.array(X_m_list[i])
        # U, S, Vt
        U, S, _ = scipy.sparse.linalg.svds(X_m_list[i], k=2)
        reduced_X_m = U @ np.diag(S)
        color = COLOR_LIST[i-num_motion]; marker = "x"
        plt.scatter(reduced_X_m[:, 0], reduced_X_m[:, 1], c=color, marker=marker)
    plt.savefig('visual_test.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CLI arguments')
    parser.add_argument('--dataset', type=str, default='ItalyPowerDemand')
    args = parser.parse_args()
    visualize_data_by_GP(name=args.dataset)