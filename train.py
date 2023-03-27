import time, argparse
import numpy as np
import matplotlib.pyplot as plt
from train_utils import train, test_classify, test_forecast
from preprocessing import load_data, add_time_variable

class MotionCode:
    def __init__(self, m, Q, latent_dim, sigma_y):
        self.m = m # Num inducing pts
        self.Q = Q # Num of kernel components
        self.latent_dim = latent_dim # Dim of motion code
        self.sigma_y = sigma_y # Noise of target

    def fit(self, X_train, Y_train, labels_train, model_path):
        train(X_train, Y_train, labels_train, model_path=model_path, 
              m=self.m, Q=self.Q, latent_dim=self.latent_dim, sigma_y=self.sigma_y)
        self.model_path = model_path

    def classify_predict(self, X_test, Y_test, labels_test):
        return test_classify(self.model_path, X_test, Y_test, labels_test)
    
    def forecast_predict(self, X_test, Y_test_list, labels_test):
        return test_forecast(self.model_path, X_test, Y_test_list, labels_test)

def run_motion_code_on(model, name, forecast=False):
    # Train and test
    if not forecast:
        # Classify mode: Run on data with 0.3-Gaussian-noise.
        Y_train, labels_train = load_data(name, split='train', add_noise=True)
        X_train, Y_train, labels_train = add_time_variable(Y_train, labels_train)
        model_path = 'saved_models/' + name + '_classify'
        model.fit(X_train, Y_train, labels_train, model_path)
        Y_test, labels_test = load_data(name, split='test', add_noise=True)
        X_test, Y_test, labels_test = add_time_variable(Y_test, labels_test)
        err = model.classify_predict(X_test, Y_test, labels_test)
        return err
    else:      
        # Train/test for forecasting on the training set but split on time horizon
        Y_train, labels_train = load_data(name, split='train')
        X, Y, labels = add_time_variable(Y_train, labels_train)
        
        # Split train/test wrt time horizon
        seq_length = Y.shape[1]
        train_num_steps = int(args.percentage*seq_length)
        Y_train = Y[:, :train_num_steps]
        Y_test = Y[:, train_num_steps:]
        X_train = X[:, :train_num_steps]
        X_test = X[:, train_num_steps:]

        # Train/test
        model_path = 'saved_models/' + name + '_forecast'
        model.fit(X_train, Y_train, labels, model_path)
        err = model.forecast_predict(X_test[0], Y_test, labels)

        return err

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CLI arguments')
    parser.add_argument('--num_inducing_pts', type=int, default=10, help='Number of inducing points for the model')
    parser.add_argument('--num_kernel_comps', type=int, default=1, help='Number of components for kernel')
    parser.add_argument('--motion_code_dim', type=int, default=2, help='Dimension of motion code')
    parser.add_argument('--sigma_y', type=float, default=0.1, help='Noise variance for target variable')
    parser.add_argument('--forecast', type=bool, default=False, help='Whether to use model for forecasting or classification')
    parser.add_argument('--percentage', type=float, default=.8, help='Split percentage for train/test forecasting')
    args = parser.parse_args()

    # Build motion code model
    model = MotionCode(m=args.num_inducing_pts, Q=args.num_kernel_comps, 
                       latent_dim=args.motion_code_dim, sigma_y=args.sigma_y)

    # Run model on all datasets and either classify or forecast testsets.
    datasets = ['ItalyPowerDemand', 'PowerCons', 'Synthetic', 'Sound', 'MoteStrain', 'ECGFiveDays',
                'SonyAIBORobotSurface2', 'GunPointOldVersusYoung', 'FreezerSmallTrain', 'UWaveGestureLibraryAll']
    for name in datasets:
        start_time = time.time()
        err = run_motion_code_on(model, name, forecast=args.forecast)
        print(name +': ' + str(err) + '. Take %.2f seconds' % (time.time()-start_time))