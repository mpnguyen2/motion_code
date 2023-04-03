import argparse
import numpy as np

import scipy
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.arima import ARIMA
from sktime.forecasting.structural import UnobservedComponents
from sktime.forecasting.tbats import TBATS

from sparse_gp import *
from utils import *
from data_processing import load_data, process_data_for_motion_codes, split_train_test_forecasting
from motion_code import MotionCode

# Color list for plotting
COLOR_LIST = ['red', 'blue', 'green', 'orange', 'purple', 'black', 'brown', 'grey', 'yellow', 'black', 'hotpink']
markers = ["." , "," , "o" , "v" , "^" , "<", ">"]

def visualize_data_by_GP(data, num_motion, m=10, Q=1, label_names=[], plot_path='out/gp_cluster_visual.png'):
    '''
    Show inducing points for individual time series to see if there are some nature clusters.
    '''
    X_m_list = get_inducing_pts_for_individual_series(m, Q, data, num_motion)

    # Plot inducing point for time series in train data.
    for i in range(2*num_motion):
        if len(X_m_list) == 0:
            continue
        # Condense inducing point vector into a 2D vector for a representation.
        X_m_list[i] = np.array(X_m_list[i])
        U, S, _ = scipy.sparse.linalg.svds(X_m_list[i], k=2)
        reduced_X_m = U @ np.diag(S)
        # Plot the condensed 2D points
        color = COLOR_LIST[i] if i < num_motion else COLOR_LIST[i-num_motion]
        marker = "x" if i < num_motion else "o"
        plt.scatter(reduced_X_m[:, 0], reduced_X_m[:, 1], c=color, marker=marker)
    
    handles = []
    for k in range(num_motion):
        handles.append(mlines.Line2D([], [], color=COLOR_LIST[k], marker='x', 
                        linestyle='None', markersize=5, label=label_names[k] + ' train data'))
        handles.append(mlines.Line2D([], [], color=COLOR_LIST[k], marker='o', 
                        linestyle='None', markersize=5, label=label_names[k] + ' test data'))
    plt.legend(handles=handles)
    plt.savefig(plot_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CLI arguments')
    parser.add_argument('--type', type=str, default='forecast_motion_code', 
                        help="Type of visualization: plot_dataset/gp_clusters/forecast_motion_code/forecast_mean_var")
    parser.add_argument('--dataset', type=str, default='ItalyPowerDemand')
    args = parser.parse_args()

    # Load train/test data.
    name = args.dataset
    Y_train, labels_train = load_data(name, split='train')
    X_train, Y_train, labels_train = process_data_for_motion_codes(Y_train, labels_train)
    num_motion = np.unique(labels_train).shape[0]
    if args.type == 'plot_dataset' or args.type == 'gp_clusters':
        Y_test, labels_test = load_data(name, split='test')
        X_test, Y_test, labels_test = process_data_for_motion_codes(Y_test, labels_test)
        data = (X_train, Y_train, labels_train, X_test, Y_test, labels_test)
    else:
        percentage = .8
        Y_train, Y_test, train_num_steps, test_num_steps = split_train_test_forecasting(Y_train, percentage)
        test_time_horizon = X_train[0, train_num_steps:]
        X_train = X_train[:, :train_num_steps]

        # Load motion code model for forecasting.
        model_path='saved_models/'+args.dataset+'_forecast'
        motion_code_model = MotionCode()
        motion_code_model.load(model_path)

    print('Loaded dataset ' + name)

    # Labels for legends
    if args.dataset == 'Sound':
        label_names = ['absorptivity', 'anything']
    elif args.dataset == 'Synthetic':
        label_names = ['Motion 1', 'Motion 2', 'Motion 3']
    elif args.dataset == 'MoteStrain':
        label_names = ['Humidity', 'Temperature']
    elif args.dataset == 'FreezerSmallTrain':
        label_names = ['Kitchen', 'Garage']
    elif args.dataset == 'PowerCons':
        label_names = ['Warm', 'Cold']
    else:
        label_names = []

    if args.type == 'plot_dataset':
        plot_timeseries(X_train, Y_train, labels_train, label_names=label_names,
                        output_file='out/plot_train_'+ args.dataset + '.png')
        plot_timeseries(X_test, Y_test, labels_test, label_names=label_names,
                        output_file='out/plot_test_'+ args.dataset + '.png')

    elif args.type == 'forecast_motion_code':
        plot_motion_codes(X_train, Y_train, test_time_horizon, labels_train, label_names,
                            motion_code_model, output_dir='out/multiple/' + name)
        
    elif args.type == 'forecast_mean_var':
        forecasters = [(motion_code_model, "Motion code"),
                        (NaiveForecaster(strategy="last", sp=12), 'Last seen'),
                        #(ExponentialSmoothing(trend="add", seasonal="additive", sp=12), 'Exponential Smoothing'),
                        (ARIMA(order=(1, 1, 0), seasonal_order=(0, 1, 0, 12), suppress_warnings=True), 'ARIMA'),
                        (TBATS(use_box_cox=False, use_trend=False, 
                               use_damped_trend=False, sp=12, use_arma_errors=False, n_jobs=1), "TBATS"),
                        (UnobservedComponents(level="local linear trend", freq_seasonal=[{"period": 12, "harmonics": 10}]), 'State-space')]
        plot_mean_covars(X_train, Y_train, Y_test, labels_train, label_names, 
                     test_time_horizon, forecasters, output_dir='out/multiple/uncertainty_' + name)
        
    elif args.type == 'gp_clusters':
        visualize_data_by_GP(data, num_motion, label_names=label_names)