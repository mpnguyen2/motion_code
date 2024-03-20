import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sparse_gp import *

# Color list for plotting
COLOR_LIST = ['red', 'blue', 'green', 'orange', 'purple', 'black', 'brown', 'grey', 'yellow', 'black', 'hotpink']

## Metric utils ##
def accuracy(pred, gt):
    """
    Return accuracy metric
    """
    return np.sum(np.array(pred)==np.array(gt))/len(pred)

def RMSE(pred, gt):
    """
    Return root-mean-squared error
    """
    return np.sqrt(np.mean((pred-gt)**2))

## Forecast helper functions given the model ##
def forecast_means_vars(forecaster, Y_train, labels, test_num_steps, num_motion):
    means = [[] for _ in range(num_motion)]
    stds = [[] for _ in range(num_motion)]
    fh = np.arange(1, test_num_steps + 1)
    num_samples = Y_train.shape[0]
    for i in range(num_samples):
        forecaster.fit(pd.Series(Y_train[i]))
        means[labels[i]].append(forecaster.predict(fh).to_numpy())
        stds[labels[i]].append(np.sqrt(forecaster.predict_var(fh).to_numpy().reshape(-1)))
    
    avg_means = [np.mean(np.array(means[k]), axis=0) for k in range(num_motion)]
    avg_stds = [np.mean(np.array(stds[k]), axis=0) for k in range(num_motion)]
    return avg_means, avg_stds

def forecast_mean_vars_motion_codes(model, test_time_horizon):
    # Average prediction for each type of motion.
    X_m, Z = model.X_m, model.Z
    means = []; stds = []
    X_m_ks = []
    for k in range(model.num_motion):
        X_m_k = sigmoid(X_m @ Z[k])
        X_m_ks.append(X_m_k)
        mean, covar = model.forecast_predict(test_time_horizon, k)
        means.append(mean); stds.append(np.sqrt(np.diag(covar)).reshape(-1))
    return means, stds

## Plotting utils ##
def plot_timeseries(X_list, y_list, labels, label_names=[], output_file='out/plot.png'):
    # Plot timeseries
    if isinstance(y_list, list):
        num_series = len(y_list)
    else:
        num_series = y_list.shape[0]
    L = len(np.unique(labels))
    if len(label_names) == 0:
        label_names = [str(i) for i in range(L)]
    is_legend_drawn = [False for _ in range(L)]
    for i in range(num_series):
        label = labels[i]
        if not is_legend_drawn[label]:
            plt.plot(X_list[i], y_list[i], c=COLOR_LIST[labels[i]], lw=0.5, label=label_names[label])
            is_legend_drawn[label] = True
        else:
            plt.plot(X_list[i], y_list[i], c=COLOR_LIST[labels[i]], lw=0.5)
    plt.legend(fontsize='10')
    plt.savefig(output_file)
    plt.clf()

def plot_motion_codes(X_train, Y_train, test_time_horizon, labels, label_names,
                           model, output_dir='out/multiple/', additional_data=None):
    # Get prediction and inducing points
    num_motion = np.unique(labels).shape[0]
    X_m, Z = model.X_m, model.Z
    X_m_ks = [sigmoid(X_m @ Z[k]) for k in range(num_motion)]

    # Plot individual stochastic process with motion code prediction and inducing pts
    if len(label_names) == 0:
        label_names = [str(i) for i in range(num_motion)]
    
    # Forecast mean and variance if `test_time_horizon` is specified. 
    if test_time_horizon is not None:
        means, stds = forecast_mean_vars_motion_codes(model, test_time_horizon)
    if additional_data is not None:
        X_original = additional_data['X']
        Y_original = additional_data['Y']
    for k in range(num_motion):
        if isinstance(X_train, list):
            indices = list(np.where(labels==k)[0])
            X = [X_train[i] for i in indices]
            Y = [Y_train[i] for i in indices]
            num_series = len(X)
        else:
            X = X_train[labels==k, :]
            Y = Y_train[labels==k, :]
            num_series = X.shape[0]
        plt.plot(X[0], Y[0], c=COLOR_LIST[k], lw=0.5, zorder=1, label=label_names[k])
        color = COLOR_LIST[(k+1)%num_motion]
        for i in range(1, num_series):
            plt.plot(X[i], Y[i], c=COLOR_LIST[k], lw=0.5, zorder=1)
        if test_time_horizon is not None:
            std = stds[k]; mean = means[k]
            plt.plot(test_time_horizon, mean, c=color, lw=2, 
                    zorder=1, label='Mean prediction')
            plt.fill_between(test_time_horizon, mean+2*std, mean-2*std,
                color=COLOR_LIST[(k+1)%num_motion], alpha=0.1, zorder=1)
        # mean, std = model.forecast_predict(test_time_horizon=X_m_ks[k], label=k)
        if additional_data is not None:
            X1 = X_original[0]
            Y1 = Y_original[labels==k, :]
        else:
            X1 = X[0]
            Y1 = Y
        Y_test = np.interp(X_m_ks[k], X1, np.mean(Y1, axis=0))
        plt.scatter(X_m_ks[k], Y_test, color=color, s=20, zorder=2,
                    label='Mean values at the most\ninformative timestamps')
        handle_list, _ = plt.gca().get_legend_handles_labels()
        if test_time_horizon is not None:
            handle_list.append(mpatches.Patch(color=COLOR_LIST[(k+1)%num_motion], 
                                            label='Uncertainty region'))
        plt.legend(handles=handle_list, fontsize='10') #, loc ="lower left"
        # max_Y = np.max(np.abs(Y))
        # plt.ylim(-2*max_Y, 2*max_Y)
        plt.savefig(output_dir + str(k) + '.png')
        plt.clf()

def plot_mean_covars(X_train, Y_train, Y_test, labels, label_names, 
                     test_time_horizon, forecasters, output_dir='out/multiple/'):
    num_motion = np.unique(labels).shape[0]
    if len(label_names) == 0:
        label_names = [str(i) for i in range(num_motion)]
    test_num_steps = test_time_horizon.shape[0]
    all_means = [{} for _ in range(num_motion)]
    all_stds = [{} for _ in range(num_motion)]
    for forecaster, forecaster_name in forecasters:
        if forecaster_name == 'Motion code':
            means, stds = forecast_mean_vars_motion_codes(forecaster, test_time_horizon)
        else:
            means, stds = forecast_means_vars(forecaster, Y_train, labels, test_num_steps, num_motion)
        for k in range(num_motion):
            all_means[k][forecaster_name] = means[k]
            all_stds[k][forecaster_name] = stds[k]
    num_forecaster = len(forecasters)
    for k in range(num_motion):
        X = X_train[labels==k, :]
        Y = Y_train[labels==k, :]
        truth_value = np.mean(Y_test[labels==k, :], axis=0)
        plt.plot(X[0], Y[0], c=COLOR_LIST[num_forecaster], lw=0.5, zorder=1, label=label_names[k])
        for i in range(1, X.shape[0]):
            plt.plot(X[i], Y[i], c=COLOR_LIST[num_forecaster], lw=0.5, zorder=1)
        # True value
        plt.plot(test_time_horizon, truth_value, 
                 c=COLOR_LIST[num_forecaster], lw=2, zorder=1, label='True value')
        cnt = 0
        for _, forecaster_name in forecasters:
            std = all_stds[k][forecaster_name]; mean = all_means[k][forecaster_name]
            color = COLOR_LIST[cnt]
            plt.plot(test_time_horizon, mean, c=color, lw=2, 
                     zorder=1, label='Mean prediction by ' + forecaster_name)
            plt.fill_between(test_time_horizon, mean+2*std, mean-2*std,
                color=color, alpha=0.1, zorder=1)
            cnt += 1

        handle_list, _ = plt.gca().get_legend_handles_labels()
        for i in range(num_forecaster):
            _, name = forecasters[i]
            handle_list.append(mpatches.Patch(color=COLOR_LIST[i], label='Uncertainty region by ' + name))
        plt.legend(handles=handle_list, fontsize='8', loc ="lower left")
        M = 1.1*max(np.max(np.abs(Y_train)), np.max(np.abs(Y_test)))
        plt.ylim(-M, M)
        plt.savefig(output_dir + str(k) + '.png')
        plt.clf()

## Get sparse GP inducing points for individual series
def get_inducing_pts_for_individual_series(m, Q, data, num_motion):
    X_train, Y_train, labels_train, X_test, Y_test, labels_test = data
    # Initialize parameters
    X_m_start = sigmoid_inv(np.linspace(0.1, 0.9, m))
    Sigma_start = softplus_inv(np.ones(Q))
    W_start = softplus_inv(np.ones(Q))

    X_m_list = [[] for _ in range(2*num_motion)]
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
        X_m, _, _ = unpack_params_single(res.x, dims=dims)
        if i < X_train.shape[0]:
            X_m_list[labels_train[i]].append(X_m)
        else:
            X_m_list[labels_test[i-X_train.shape[0]] + num_motion].append(X_m) 
        
    return X_m_list