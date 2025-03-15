import argparse
import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.arima import ARIMA
from sktime.forecasting.structural import UnobservedComponents
from sktime.forecasting.tbats import TBATS
from data_processing import get_train_test_data_classify, get_train_test_data_forecast
from utils import plot_timeseries, plot_motion_codes, plot_mean_covars
from motion_code import MotionCode


# Color list for plotting
COLOR_LIST = ['red', 'blue', 'green', 'orange', 'purple', 'black', 'brown', 'grey', 'yellow', 'black', 'hotpink']
markers = ["." , "," , "o" , "v" , "^" , "<", ">"]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CLI arguments')
    parser.add_argument('--type', type=str, default='forecast_motion_code', 
                        help="Type of visualization: plot_dataset/classify_motion_code"
                        + "/forecast_motion_code/forecast_mean_var")
    parser.add_argument('--dataset', type=str, default='ItalyPowerDemand')
    parser.add_argument('--load_existing_data', type=bool, default=False)
    args = parser.parse_args()
    viz_type = args.type
    name = args.dataset
    load_existing_data = args.load_existing_data

    # Load train/test data.
    if viz_type == 'forecast_motion_code' or viz_type == 'forecast_mean_var':
        benchmark_data, motion_code_data = (
                get_train_test_data_forecast(name)
            )
        X_train, Y_train, labels, test_time_horizon, Y_test = motion_code_data
    else:
        benchmark_data, motion_code_data = (
                get_train_test_data_classify(name, load_existing_data=False, add_noise=False)
            )
        X_train, Y_train, labels_train, X_test, Y_test, labels_test = motion_code_data
    
    # Train/Load pretrained model
    if viz_type != 'plot_dataset':
        # Initialize motion code model
        if name == 'PD setting 1':
            motion_code_model = MotionCode(m=6, Q=2, latent_dim=2)
        elif name == 'PD setting 2':
            motion_code_model = MotionCode(m=12, Q=2, latent_dim=2)
        else:
            motion_code_model = MotionCode(m=12, Q=1, latent_dim=2)

        if viz_type == 'classify_motion_code':
            model_path='saved_models/visualize/' + name + '_classify'
            motion_code_model.fit(X_train, Y_train, labels_train, model_path)
        else:
            model_path='saved_models/' + name+'_forecast'    
        motion_code_model.load(model_path)

    print(f'Loaded dataset {name}')

    # Labels for legends
    if name == 'Pronunciation Audio':
        label_names = ['absorptivity', 'anything']
    elif name == 'PD setting 1':
        label_names = ['normal', 'light tremor']
    elif name == 'PD setting 2':
        label_names = ['normal', 'light tremor', 'noticeable tremor']
    elif name == 'Synthetic':
        label_names = ['Motion 1', 'Motion 2', 'Motion 3']
    elif name == 'MoteStrain':
        label_names = ['Humidity', 'Temperature']
    elif name == 'FreezerSmallTrain':
        label_names = ['Kitchen', 'Garage']
    elif name == 'PowerCons':
        label_names = ['Warm', 'Cold']
    elif name == 'ItalyPowerDemand':
        label_names = ['October to March', 'April to September']
    elif name == 'SonyAIBORobotSurface2':
        label_names = ['Cement', 'Carpet']
    elif name == 'FreezerSmallTrain':
        label_names = ['Kitchen', 'Garage']
    elif name == 'Chinatown':
        label_names = ['Weekend', 'Weekday']
    elif name == 'InsectEPGRegularTrain':
        label_names = ['Class 1', 'Class 2', 'Class 3']
    else:
        label_names = ['0', '1']

    if args.type == 'plot_dataset':
        plot_timeseries(X_train, Y_train, labels_train, label_names=label_names,
                        output_file='out/plot_train_'+ args.dataset + '.png')
        plot_timeseries(X_test, Y_test, labels_test, label_names=label_names,
                        output_file='out/plot_test_'+ args.dataset + '.png')
    
    elif args.type=='classify_motion_code':
        plot_motion_codes(X_train, Y_train, None, labels_train, label_names,
                            motion_code_model, output_dir='out/multiple/classify_' + name)
        
    elif args.type == 'forecast_motion_code':
        plot_motion_codes(X_train, Y_train, test_time_horizon, labels, label_names,
                        motion_code_model, output_dir='out/multiple/' + name)
        
    elif args.type == 'forecast_mean_var':
        forecasters = [(motion_code_model, "Motion code"),
                        (NaiveForecaster(strategy="last", sp=12), 'Last seen'),
                        #(ExponentialSmoothing(trend="add", seasonal="additive", sp=12), 'Exponential Smoothing'),
                        (ARIMA(order=(1, 1, 0), seasonal_order=(0, 1, 0, 12), suppress_warnings=True), 'ARIMA'),
                        (TBATS(use_box_cox=False, use_trend=False, 
                               use_damped_trend=False, sp=12, use_arma_errors=False, n_jobs=1), "TBATS"),
                        (UnobservedComponents(level="local linear trend", freq_seasonal=[{"period": 12, "harmonics": 10}]), 'State-space')]
        plot_mean_covars(X_train, Y_train, Y_test, labels, label_names, 
                     test_time_horizon, forecasters, output_dir='out/multiple/uncertainty_' + name)
