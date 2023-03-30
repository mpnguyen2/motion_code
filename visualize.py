import argparse
import numpy as np
from scipy.optimize import minimize
import scipy
import matplotlib.pyplot as plt

from sparse_gp import *
from utils import *
from preprocessing import *
from train_utils import load_model

# Color list for plotting
COLOR_LIST = ['red', 'blue', 'green', 'yellow', 'black', 'orange', 'brown', 'grey', 'purple', 'hotpink']
markers = ["." , "," , "o" , "v" , "^" , "<", ">"]

def visualize_data_by_GP(name, m=10, Q=1):
    Y_train, labels_train = load_data(name, split='train')
    X_train, Y_train, train_labels = add_time_variable(Y_train, labels_train)
    Y_test, labels_test = load_data(name, split='test')
    X_test, Y_test, test_labels = add_time_variable(Y_test, labels_test)
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

    for i in range(num_motion, 2*num_motion):
        if len(X_m_list[i]) == 0:
            continue
        X_m_list[i] = np.array(X_m_list[i])
        # U, S, Vt
        U, S, _ = scipy.sparse.linalg.svds(X_m_list[i], k=2)
        reduced_X_m = U @ np.diag(S)
        color = COLOR_LIST[i-num_motion]; marker = "x"
        plt.scatter(reduced_X_m[:, 0], reduced_X_m[:, 1], c=color, marker=marker)
    plt.savefig('visual.png')

def visualize_forecast_results(model_path, name, label_names, percentage=.8):
    # Load data
    Y_train, labels_train = load_data(name, split='train')
    X, Y, labels = add_time_variable(Y_train, labels_train)
    seq_length = Y.shape[1]
    train_num_steps = int(percentage*seq_length)
    Y_train = Y[:, :train_num_steps]
    # Y_test = Y[:, train_num_steps:]
    X_train = X[:, :train_num_steps]
    X_test = X[:, train_num_steps:][0]
    # Extract optimal trained params
    X_m, Z, Sigma, W, mu_ms, A_ms, K_mm_invs = load_model(model_path)
    num_motion = Z.shape[0]
    
    # Average prediction for each type of motion.
    means = []; covars = []
    X_m_ks = []
    for k in range(num_motion):
        X_m_k = sigmoid(X_m @ Z[k])
        X_m_ks.append(X_m_k)
        mean, covar = q(X_test, X_m_k, (Sigma[k], W[k]), mu_ms[k], A_ms[k], K_mm_invs[k])
        means.append(mean); covars.append(covar)
    plot_motion_types(X_train, Y_train, X_test, labels, means, covars, np.array(X_m_ks), label_names)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CLI arguments')
    parser.add_argument('--type', type=str, default='forecast_result', 
                        help="Type of visualization: plot dataset/forecast result/individual-GP cluster")
    parser.add_argument('--dataset', type=str, default='ItalyPowerDemand')
    args = parser.parse_args()
    if args.dataset == 'Sound':
        label_names = ['absorptivity', 'anything']
    elif args.dataset == 'Synthetic':
        label_names = ['Motion 1', 'Motion 2', 'Motion 3']
    else:
        label_names = []

    if args.type == 'plot_dataset':
        Y_train, labels_train = load_data(args.dataset, split='train')
        X_train, Y_train, labels_train = add_time_variable(Y_train, labels_train)
        plot_timeseries(X_train, Y_train, labels_train, 
                        output_dir='out/plot_train_'+ args.dataset + '.png', label_names=label_names)
        Y_test, labels_test = load_data(args.dataset, split='test')
        X_test, Y_test, labels_test = add_time_variable(Y_test, labels_test)
        plot_timeseries(X_test, Y_test, labels_test, 
                        output_dir='out/plot_test_'+ args.dataset + '.png', label_names=label_names)

    elif args.type == 'forecast_result':
        visualize_forecast_results(model_path='saved_models/'+args.dataset+'_forecast', 
                                   name=args.dataset, label_names=label_names)
    else:
        visualize_data_by_GP(name=args.dataset)