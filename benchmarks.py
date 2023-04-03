import argparse
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

from data_processing import load_data, process_data, split_train_test_forecasting
from utils import RMSE
from motion_code import MotionCode, motion_code_classify, motion_code_forecast

def run_classify(clf, clf_name, X_train, y_train, X_test, y_test, name=""):
    if clf_name == 'Motion code':
        return motion_code_classify(clf, name, X_train, y_train, X_test, y_test)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    if type(y_pred) is tuple:
        y_pred, _ = y_pred
    return np.sum(y_pred == y_test)/y_pred.shape[0]

def run_forecast(forecaster, forecaster_name, name, percentage):
    if forecaster_name == 'Motion code':
        return motion_code_forecast(forecaster, name, percentage)
    
    # Get train/test data: collection instead of a single time series
    Y, labels = load_data(name, split='train')
    Y, labels = process_data(Y, labels)
    if Y.shape[0] == 0:
        return -1
    Y_train, Y_test, _, test_num_steps = split_train_test_forecasting(Y, percentage)
    
    # specifying forecasting horizon
    fh = np.arange(1, test_num_steps + 1) 

    # Fitting and store errors.
    num_motion = np.unique(labels).shape[0]
    all_errors = [[] for _ in range(num_motion)]
    num_samples = Y.shape[0]
    for i in range(num_samples):
        forecaster.fit(pd.Series(Y_train[i]))
        y_pred = forecaster.predict(fh).to_numpy()
        all_errors[labels[i]].append(RMSE(y_pred, Y_test[i]))

    # Return mean error for each type of motion.
    errs = np.zeros(num_motion)
    for i in range(num_motion):
        errs[i] = np.mean(np.array(all_errors[i]))
    
    return errs

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CLI arguments')
    parser.add_argument('--forecast', type=bool, default=False, help='Type of benchmarks: either classify or forecast')
    parser.add_argument('--load_existing_data', type=bool, default=False, help='Load saved noisy data')
    args = parser.parse_args()

    datasets = ['Chinatown', 'ECGFiveDays', 'FreezerSmallTrain', 'GunPointOldVersusYoung', 'HouseTwenty', 'InsectEPGRegularTrain', 
            'ItalyPowerDemand', 'Lightning7', 'MoteStrain', 'PowerCons', 'SonyAIBORobotSurface2', 'Sound', 'Synthetic', 
            'UWaveGestureLibraryAll']

    if args.forecast:
        # Work with original versions of data.
        all_forecasters = [(ExponentialSmoothing(trend="add", seasonal="additive", sp=12), 'Exponential Smoothing'),
                            (ARIMA(order=(1, 1, 0), seasonal_order=(0, 1, 0, 12), suppress_warnings=True), 'ARIMA'),
                            (UnobservedComponents(level="local linear trend", freq_seasonal=[{"period": 12, "harmonics": 10}]), 'State-space'),
                            (NaiveForecaster(strategy="last", sp=12), 'naive'),
                            (TBATS(use_box_cox=False, use_trend=False, 
                                use_damped_trend=False, sp=12, use_arma_errors=False, n_jobs=1), "TBATS"),
                            (MotionCode(), 'Motion code')]
        
        for forecaster, forecaster_name in all_forecasters:
            print(forecaster_name)
            for name in datasets:
                try:
                    print(name + ': ' + str(run_forecast(forecaster, forecaster_name, name, percentage=.8)))
                except:
                    print(name + ': -1')
            print('\n')
    else:
        # Load noisy versions of data.
        noisy_data = {}
        for name in datasets:
            data_path = 'data/noisy/' + name
            if args.load_existing_data:
                data = np.load(data_path + '.npy', allow_pickle=True).item()
                X_train, y_train = data.get('X_train'), data.get('y_train')
                X_test, y_test = data.get('X_test'), data.get('y_test')
            else:
                X_train, y_train = load_data(name, split='train', add_noise=True)
                X_test, y_test = load_data(name, split='test', add_noise=True)
                np.save(data_path, {'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test': y_test})
            
            noisy_data[name] = (X_train, y_train, X_test, y_test)
        
        # All classifers for bechmarks
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
                    (HIVECOTEV2(time_limit_in_minutes=0.2), "Hive-Cote 2"),
                    (MotionCode(), "Motion code")]

        # Run classifers.
        result = {}
        for clf, clf_name in all_clfs:
            print(clf_name)
            result[clf_name] = []
            for name in datasets:
                try:
                    X_train, y_train, X_test, y_test = noisy_data[name]
                    acc = run_classify(clf, clf_name, X_train, y_train, X_test, y_test, name)
                    print(name + ': ' + str(acc))
                    result[clf_name].append(acc)
                except:
                    print(name + ': -1')
                    result[clf_name].append(-1)
            print('\n')

        pd.DataFrame(result, index=datasets).to_csv('out/classify_accuracy.csv')