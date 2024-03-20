import time
from tqdm import tqdm
import numpy as np

from data_processing import load_data, process_data_for_motion_codes, split_train_test_forecasting
from motion_code_utils import optimize_motion_codes, classify_predict_helper
from sparse_gp import sigmoid, q
from utils import accuracy, RMSE

class MotionCode:
    """
    Class for motion code model

    Attributes
    ----------
    m: int
        Number of inducing points
    Q: int
        Number of kernel components
    latent_dim: int
        Dimension of motion codes
    sigma_y: float
        Noise of the target variable
    model_path: str
        Path to save/load the model
    num_motion: int
        Number of stochastic processes the model considered
    X_m: numpy.ndarray
        The common transformation for all stochastic processes underlying collections of time series data
    Z: numpy.ndarray
        Stacked motion codes
    Sigma: numpy.ndarray
        Stacked kernel parameters for the exponents of all stochastic processes
    W: numpy.ndarray
        Stacked kernel parameters for the scales of all stochastic processes
    kernel_params:
        Stacked pairs of kernel parameters (Sigma, W) for all stochastic processes
    mu_ms: numpy.ndarray
        Stacked mean prediction over the inducing points for all stochastic processes
    A_ms: numpy.ndarray
        Stacked covariance of mu_ms for all stochastic processes
    K_mm_invs: numpy.ndarray
        Stacked inverses of the kernel over inducing points for all stochastic processes
        
    Methods
    -------
    fit(X_train, Y_train, labels_train, model_path)
        Train model on `(X_train, Y_train)` time series with collection labels `labels_train`
        and then save model to `model_path`

    load(model_path='')
        Load model at `model_path`
    classify_predict(X_test, Y_test):
        Predict the label for a single time series `(X_test, Y_test)` in classification problem.

    classify_predict_on_batches(X_test_list, Y_test_list, true_labels)
        Predict the labels for a list of time series `(X_test_list, Y_test_list)`
        and then compare against `true_labels` and return accuracy

    forecast_predict(self, test_time_horizon, label)
        Predict future values on `test_time_horizon` for stochastic process with label `label`
    
    forecast_predict_on_batches(self, test_time_horizon, Y_test_list, labels)
        Predict future (mean) values on `test_time_horizon` for all stochastic processes
        and then use the results to compare against the time series list `Y_test_list` with labels `labels`
        for RMSE errors.

    """
    def __init__(self, m=10, Q=1, latent_dim=2, sigma_y=0.1):
        self.m = m # Num inducing pts
        self.Q = Q # Num of kernel components
        self.latent_dim = latent_dim # Dim of motion code
        self.sigma_y = sigma_y # Noise of target

    def fit(self, X_train, Y_train, labels_train, model_path):
        start_time = time.time()
        self.model_path = model_path
        optimize_motion_codes(X_train, Y_train, labels_train, model_path=model_path, 
              m=self.m, Q=self.Q, latent_dim=self.latent_dim, sigma_y=self.sigma_y)
        self.train_time = time.time() - start_time

    def load(self, model_path=''):
        if len(model_path) == 0 and self.model_path is not None:
            model_path = self.model_path
        params = np.load(model_path + '.npy', allow_pickle=True).item()
        self.X_m, self.Z, self.Sigma, self.W = params.get('X_m'), params.get('Z'), params.get('Sigma'), params.get('W') 
        self.mu_ms, self.A_ms, self.K_mm_invs = params.get('mu_ms'), params.get('A_ms'), params.get('K_mm_invs')
        self.num_motion = self.Z.shape[0]
        self.kernel_params = []
        for k in range(self.num_motion):
            self.kernel_params.append((self.Sigma[k], self.W[k]))

    def classify_predict(self, X_test, Y_test):
        return classify_predict_helper(X_test, Y_test, self.kernel_params, 
                                       self.X_m, self.Z, self.mu_ms, self.A_ms, self.K_mm_invs)
    
    def classify_predict_on_batches(self, X_test_list, Y_test_list, true_labels):
        # Predict each trajectory/timeseries in the test dataset
        num_predicted = 0
        pred = []; gt = []
        if isinstance(Y_test_list, list):
            num_test = len(Y_test_list)
        else:
            num_test = Y_test_list.shape[0]
        pbar = tqdm(zip(X_test_list, Y_test_list), total=num_test, leave=False)
        num_predicted = 0
        for X_test, Y_test in pbar:
            # Get predict and ground truth motions
            pred_label = self.classify_predict(X_test, Y_test)
            gt_label = true_labels[num_predicted]
            pbar.set_description(f'Predict: {pred_label}; gt: {gt_label}')
            # Append results to lists for final evaluation
            pred.append(pred_label); gt.append(gt_label)
            num_predicted += 1

        # Accurary evaluation
        return accuracy(pred, gt)
    
    def forecast_predict(self, test_time_horizon, label):
        k = label
        return q(test_time_horizon, sigmoid(self.X_m @ self.Z[k]), 
                 self.kernel_params[k], self.mu_ms[k], self.A_ms[k], self.K_mm_invs[k])
    
    def forecast_predict_on_batches(self, test_time_horizon, Y_test_list, labels):
        # Average prediction for each type of motion.
        mean_preds = []
        for k in range(self.num_motion):
            mean, _ = self.forecast_predict(test_time_horizon, label=k)
            mean_preds.append(mean)
        
        all_errors = [[] for _ in range(self.num_motion)]
        
        for i in range(len(Y_test_list)):
            label = labels[i]
            all_errors[label].append(RMSE(mean_preds[label], Y_test_list[i]))

        errs = np.zeros(self.num_motion)
        for i in range(self.num_motion):
            errs[i] = np.mean(np.array(all_errors[i]))
        
        return errs

## Convenient functions that combine train, load, test, report errors.
def motion_code_classify(model, name, Y_train, labels_train, Y_test, labels_test, load_existing_model=False):
    # Train
    X_train, Y_train, labels_train = process_data_for_motion_codes(Y_train, labels_train)
    model_path = 'saved_models/' + name + '_classify'
    if not load_existing_model:
        model.fit(X_train, Y_train, labels_train, model_path)
    # Test
    model.load(model_path)
    X_test, Y_test, labels_test = process_data_for_motion_codes(Y_test, labels_test)
    acc = model.classify_predict_on_batches(X_test, Y_test, labels_test)
    return acc

def motion_code_forecast(model, name, percentage, load_existing_model=False):
    # Train/test for forecasting on the training set but split on time horizon
    Y_train, labels_train = load_data(name, split='train')
    X, Y, labels = process_data_for_motion_codes(Y_train, labels_train)
    
    # Split train/test wrt time horizon
    Y_train, Y_test, train_num_steps, _ = split_train_test_forecasting(Y, percentage)
    X_train = X[:, :train_num_steps] # Motion code need addition X variable
    test_time_horizon = X[0, train_num_steps:]

    # Train/test
    model_path = 'saved_models/' + name + '_forecast'
    if not load_existing_model:
        model.fit(X_train, Y_train, labels, model_path)
    model.load(model_path)
    err = model.forecast_predict_on_batches(test_time_horizon, Y_test, labels)

    return err