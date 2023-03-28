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
# from sktime.forecasting.ets import AutoETS
from sktime.forecasting.structural import UnobservedComponents

from preprocessing import load_data, add_time_variable
from utils import RMSE
from train import MotionCode, motion_code_classify

def test_classify(clf, X_train, y_train, X_test, y_test, name=""):
    if clf_name == 'Motion code':
        return motion_code_classify(clf, name, X_train, y_train, X_test, y_test)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    if type(y_pred) is tuple:
        y_pred, _ = y_pred
    return np.sum(y_pred == y_test)/y_pred.shape[0]

def test_forecast(forecaster, name, percentage):
    # Get train/test data: collection instead of a single time series
    Y, labels = load_data(name, split='train')
    _, Y, labels = add_time_variable(Y, labels)
    if Y.shape[0] == 0:
        return -1
    num_samples = Y.shape[0]
    num_motion = np.unique(labels).shape[0]
    seq_length = Y.shape[1]
    train_num_steps = int(percentage*seq_length)
    test_num_steps = seq_length - train_num_steps
    all_errors = [[] for _ in range(num_motion)]
    Y_train = Y[:, :train_num_steps]
    Y_test = Y[:, train_num_steps:]
    fh = np.arange(1, test_num_steps + 1) # specifying forecasting horizon

    # Fitting and store errors.
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
    args = parser.parse_args()

    datasets = ['ItalyPowerDemand', 'PowerCons', 'Synthetic', 'Sound', 'MoteStrain', 'ECGFiveDays', 
                'SonyAIBORobotSurface2', 'GunPointOldVersusYoung', 'FreezerSmallTrain']
    '''
    'ItalyPowerDemand', 'PowerCons', 'Synthetic', 'Sound', 'MoteStrain', 'ECGFiveDays',
    'SonyAIBORobotSurface2', 'GunPointOldVersusYoung', 'FreezerSmallTrain', 'UWaveGestureLibraryAll',
    'Earthquakes', 'Lightning7', 'ACSF1', 'HouseTwenty', 'DodgerLoopDay', 'Chinatown', 'InsectEPGRegularTrain'
    '''
    if args.forecast:
        # Work with original versions of data.
        all_forecasters = [(NaiveForecaster(strategy="last", sp=12), 'naive'),
                           (ExponentialSmoothing(trend="add", seasonal="additive", sp=12), 'Exponential Smoothing'),
                           (ARIMA(order=(1, 1, 0), seasonal_order=(0, 1, 0, 12), suppress_warnings=True), 'ARIMA'),
                           (UnobservedComponents(level="local linear trend", freq_seasonal=[{"period": 12, "harmonics": 10}]), 'State-space')]
        '''
        (NaiveForecaster(strategy="last", sp=12), 'naive'),
        (ExponentialSmoothing(trend="add", seasonal="additive", sp=12), 'Exponential Smoothing'),
        (ARIMA(order=(1, 1, 0), seasonal_order=(0, 1, 0, 12), suppress_warnings=True), 'ARIMA'),
        (AutoARIMA(sp=12, suppress_warnings=True), 'Auto-ARIMA'),
        (UnobservedComponents(level="local linear trend", freq_seasonal=[{"period": 12, "harmonics": 10}]), 'State-space'),
        (AutoETS(auto=True, sp=12, n_jobs=-1), 'Auto-ETS'),
        '''
        for forecaster, forecaster_name in all_forecasters:
            print(forecaster_name)
            for name in datasets:
                try:
                    print(name + ': ' + str(test_forecast(forecaster, forecaster_name, name, percentage=.8)))
                except:
                    print(name + ': -1')
            print('\n')
    else:
        # Load noisy versions of data.
        noisy_data = {}
        for name in datasets:
            X_train, y_train = load_data(name, split='train', add_noise=True)
            X_test, y_test = load_data(name, split='test', add_noise=True)
            noisy_data[name] = (X_train, y_train, X_test, y_test)
        
        # All classifers for bechmarks
        mean_gaussian_tskernel = AggrDist(RBF()) 
        all_clfs = [(MotionCode(), "Motion code"), 
                    (KNeighborsTimeSeriesClassifier(distance="dtw"), "DTW"),
                    (TimeSeriesForestClassifier(n_estimators=5), "TSF"),
                    (RandomIntervalSpectralEnsemble(), "RISE"),
                    (IndividualBOSS(), "BOSS"),
                    (HIVECOTEV2(time_limit_in_minutes=0.2), "Hive-Cote 2"),
                    (RocketClassifier(num_kernels=500), "Rocket"),
                    (TEASER(), "Teaser"),
                    (Catch22Classifier(), "catch22"), 
                    (LSTMFCNClassifier(n_epochs=200, verbose=0), "LSTM-FCN"),
                    (TimeSeriesSVC(kernel=mean_gaussian_tskernel), "SVC"),
                    (ShapeletTransformClassifier(estimator=RotationForest(n_estimators=3), 
                        n_shapelet_samples=100, max_shapelets=10, batch_size=20), "Shapelet"),
                    (BOSSEnsemble(max_ensemble_size=3), "BOSS-E")
                    ]
        '''
        (MotionCode(), "Motion code"),
        (KNeighborsTimeSeriesClassifier(distance="dtw"), "DTW"),
        (Catch22Classifier(), "catch22"), 
        (TEASER(), "Teaser"), 
        (IndividualBOSS(), "BOSS"),
        mean_gaussian_tskernel = AggrDist(RBF()) (TimeSeriesSVC(kernel=mean_gaussian_tskernel), "SVC"),
        (RandomIntervalSpectralEnsemble(), "RISE"),
        (ElasticEnsemble(proportion_of_param_options=0.1, proportion_train_for_test=0.1, 
                        distance_measures = ["dtw","ddtw"], majority_vote=True), "EE"),
        (BOSSEnsemble(max_ensemble_size=3), "BOSS-E"),
        (TimeSeriesForestClassifier(n_estimators=5), "TSF"),
        (RocketClassifier(num_kernels=500), "Rocket"),
        (ShapeletTransformClassifier(estimator=RotationForest(n_estimators=3), 
            n_shapelet_samples=100, max_shapelets=10, batch_size=20), "Shapelet"),
        (LSTMFCNClassifier(n_epochs=200, verbose=0), "LSTM-FCN"),
        (HIVECOTEV2(time_limit_in_minutes=0.2), "Hive-Cote 2"),
        '''

        # Run classifers.
        for clf, clf_name in all_clfs:
            print(clf_name)
            for name in datasets:
                X_train, y_train, X_test, y_test = noisy_data[name]
                try:
                    print(name + ': ' + str(test_classify(clf, X_train, y_train, X_test, y_test, name)))
                except: 
                    print(name + ': -1')
            print('\n')