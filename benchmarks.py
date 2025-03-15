import argparse
from collections import defaultdict
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
from sktime.classification.dictionary_based import IndividualBOSS, BOSSEnsemble
from sktime.classification.interval_based import RandomIntervalSpectralEnsemble, TimeSeriesForestClassifier
from sklearn.gaussian_process.kernels import RBF
from sktime.dists_kernels import AggrDist
from sktime.classification.kernel_based import TimeSeriesSVC, RocketClassifier
from sktime.classification.early_classification import TEASER
from sktime.classification.feature_based import Catch22Classifier
from sktime.classification.shapelet_based import ShapeletTransformClassifier
from sktime.classification.sklearn import RotationForest
from sktime.classification.hybrid import HIVECOTEV2
from sktime.classification.deep_learning import LSTMFCNClassifier
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.arima import ARIMA
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.structural import UnobservedComponents
from sktime.forecasting.tbats import TBATS
from data_processing import get_train_test_data_forecast, get_train_test_data_classify
from motion_code import MotionCode, motion_code_classify, motion_code_forecast
from utils import RMSE


################################ Helper functions ################################
def run_classify_benchmark(clf, Y_train, labels_train, Y_test, labels_test, name=""):
    clf.fit(Y_train, labels_train)
    labels_pred = clf.predict(Y_test)
    if type(labels_pred) is tuple:
        labels_pred, _ = labels_pred
    return np.sum(labels_pred == labels_test)/labels_pred.shape[0]

def run_forecast_benchmark(forecaster, Y_train, Y_test, labels, test_num_steps):
    num_samples = labels.shape[0]
    if not num_samples:
        return -1
    
    # specifying forecasting horizon
    fh = np.arange(1, test_num_steps + 1) 

    # Fitting and store errors.
    num_motion = np.unique(labels).shape[0]
    all_errors = [[] for _ in range(num_motion)]
    for i in range(num_samples):
        forecaster.fit(pd.Series(Y_train[i]))
        y_pred = forecaster.predict(fh).to_numpy()
        all_errors[labels[i]].append(RMSE(y_pred, Y_test[i]))

    # Return mean error.
    errs = np.zeros(num_motion)
    for i in range(num_motion):
        errs[i] = np.mean(np.array(all_errors[i]))    
    return errs

def get_data_dict(datasets, forecast, load_existing_data):
    data_dict = {}
    for name in datasets:
        if forecast:
            benchmark_data, motion_code_data = get_train_test_data_forecast(name)
        else:
            benchmark_data, motion_code_data = (
                get_train_test_data_classify(name, load_existing_data)
            )
        data_dict[name] = benchmark_data
        data_dict[name + '_motion_code'] = motion_code_data
    return data_dict


################################ Main benchmarking function ################################
def main(forecast, dataset_type,
         load_existing_data,
         load_existing_model,
         output_path):
    
    # Load and process data
    datasets = []
    if dataset_type == 'basics':
        datasets = ['ECGFiveDays', 'FreezerSmallTrain', 'HouseTwenty',
                    'InsectEPGRegularTrain', 'ItalyPowerDemand', 'Lightning7',
                    'MoteStrain', 'PowerCons', 'SonyAIBORobotSurface2', 'UWaveGestureLibraryAll']
    if dataset_type == 'pronunciation':
        datasets = ['Pronunciation Audio']
    if dataset_type == 'parkinson_1':
        datasets = ['PD setting 1']
    if dataset_type == 'parkinson_2':
        datasets = ['PD setting 2']
    data_dict = get_data_dict(datasets, forecast, load_existing_data)

    if forecast:
        # Focus on classification for Parkinson's disease sensor data
        if dataset_type == 'parkinson_1' or dataset_type == 'parkinson_2':
            return 0

        # Initialize forecasters
        all_forecasters = [(ExponentialSmoothing(trend="add", seasonal="additive", sp=12), 'Exponential Smoothing'),
                            (ARIMA(order=(1, 1, 0), seasonal_order=(0, 1, 0, 12), suppress_warnings=True), 'ARIMA'),
                            (UnobservedComponents(level="local linear trend", freq_seasonal=[{"period": 12, "harmonics": 10}]), 'State-space'),
                            (NaiveForecaster(strategy="last", sp=12), 'Last seen'),
                            (TBATS(use_box_cox=False, use_trend=False, 
                                use_damped_trend=False, sp=12, use_arma_errors=False, n_jobs=1), "TBATS"),
                            (MotionCode(), 'Motion Code')]
        # Run forecasters
        result = defaultdict(list)
        for forecaster, forecaster_name in all_forecasters:
            print(forecaster_name)
            for name in datasets:
                try:
                    if forecaster_name != 'Motion Code':
                        Y_train, Y_test, labels, test_num_steps = data_dict[name]
                        err = run_forecast_benchmark(forecaster, Y_train, Y_test,
                                                     labels, test_num_steps)
                    else:
                        X_train, Y_train, labels, test_time_horizon, Y_test = (
                            data_dict[name + '_motion_code']
                        )
                        err = motion_code_forecast(forecaster, name, X_train, Y_train, labels,
                                                   test_time_horizon, Y_test, load_existing_model)
                    print(name + ': ' + str(err))
                    result[forecaster_name].append(err)
                except:
                    print(name + ': -1')
                    result[forecaster_name].append(-1)
            print('\n')
        pd.DataFrame(result, index=datasets).to_csv(output_path)

    else:
        # Initialize classifier
        mean_gaussian_tskernel = AggrDist(RBF()) 
        all_clfs = [(KNeighborsTimeSeriesClassifier(distance="dtw"), "DTW"),
                    (TimeSeriesForestClassifier(n_estimators=5), "TSF"),
                    (RandomIntervalSpectralEnsemble(), "RISE"),
                    (IndividualBOSS(), "BOSS"),
                    (BOSSEnsemble(max_ensemble_size=3), "BOSS-E"),
                    (Catch22Classifier(), "catch22"), 
                    (ShapeletTransformClassifier(estimator=RotationForest(n_estimators=3), 
                        n_shapelet_samples=100, max_shapelets=10, batch_size=20), "Shapelet"),
                    (TEASER(), "Teaser"),
                    (TimeSeriesSVC(kernel=mean_gaussian_tskernel), "SVC"),
                    (LSTMFCNClassifier(n_epochs=200, verbose=0), "LSTM-FCN"),
                    (RocketClassifier(num_kernels=500), "Rocket"),
                    (HIVECOTEV2(time_limit_in_minutes=0.2), "Hive-Cote 2")
        ]
        if dataset_type == 'basics' or dataset_type == 'pronunciation':
            all_clfs.append((MotionCode(m=10, Q=1, latent_dim=2), 'Motion Code'))
        if dataset_type == 'parkinson_1':
            all_clfs.append((MotionCode(m=6, Q=2, latent_dim=2), 'Motion Code'))
        if dataset_type == 'parkinson_2':
            all_clfs.append((MotionCode(m=12, Q=2, latent_dim=2), 'Motion Code'))

        # Run classifers.
        result = defaultdict(list)
        for clf, clf_name in all_clfs:
            print(clf_name)
            result[clf_name] = []
            for name in datasets:
                try:
                    if clf_name != 'Motion Code':
                        Y_train, labels_train, Y_test, labels_test = data_dict[name]
                        acc = run_classify_benchmark(clf, Y_train, labels_train, Y_test, labels_test, name)
                    else:
                        X_train, Y_train, labels_train, X_test, Y_test, labels_test = (
                              data_dict[name + '_motion_code']
                        )
                        acc = motion_code_classify(clf, name,
                                                   X_train, Y_train, labels_train,
                                                   X_test, Y_test, labels_test,
                                                   load_existing_model)
                    print(name + ': ' + str(acc))
                    result[clf_name].append(acc)
                except:
                    print(name + ': -1')
                    result[clf_name].append(-1)
            print('\n')
        pd.DataFrame(result, index=datasets).to_csv(output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CLI arguments')
    parser.add_argument('--forecast', type=bool, default=False, help='Type of benchmarks: either classify or forecast')
    parser.add_argument('--dataset_type', type=str, default='basics',
                        help='basics/pronunciation/parkinson_1/parkinson_2')
    parser.add_argument('--load_existing_data', type=bool, default=False, help='Load existing data')
    parser.add_argument('--load_existing_model', type=bool, default=False, help='Load existing Motion Code model')
    parser.add_argument('--output_path', type=str, default='out', help='Output path')
    args = parser.parse_args()
    forecast = args.forecast
    dataset_type = args.dataset_type
    load_existing_data = args.load_existing_data
    load_existing_model = args.load_existing_model
    output_path = args.output_path
    task = 'forecast' if forecast else 'classify'
    print('Command line parameters')
    print(f'Perform {task}')
    print(f'Dataset: {dataset_type}')
    print(f'Output path: {output_path}')
    print('------------------------------------------------------------------\n')

    main(forecast, dataset_type, load_existing_data, load_existing_model, output_path)