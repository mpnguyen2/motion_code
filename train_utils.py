import time, argparse
import numpy as np
import jax.numpy as jnp
from scipy.optimize import minimize
#from jax.scipy.optimize import minimize

from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from dtaidistance import dtw

from sparse_gp import *
from utils import *

def opt_callback(x):
    print("Optimizing...")
    pass

def train(X_list, Y_list, labels, model_path, m=10, Q=8, latent_dim=3, sigma_y=0.1):
    num_motion = np.unique(labels).shape[0]
    dims = (num_motion, m, latent_dim, Q)

    # Initialize parameters
    X_m_start = np.repeat(sigmoid_inv(np.linspace(0.1, 0.9, m)).reshape(1, -1), latent_dim, axis=0).swapaxes(0, 1)
    Z_start = np.ones((num_motion, latent_dim))
    Sigma_start = softplus_inv(np.ones((num_motion, Q)))
    W_start = softplus_inv(np.ones((num_motion, Q)))

    # Optimize X_m, Z, and kernel parameters including Sigma, W
    res = minimize(fun=elbo_fn(X_list, Y_list, labels, sigma_y, dims),
        x0 = pack_params([X_m_start, Z_start, Sigma_start, W_start]),
        method='L-BFGS-B', jac=True)
    # print('Inducing pts, motion codes, and kernel params successfully optimized: ', res.success)
    X_m, Z, Sigma, W = unpack_params(res.x, dims=dims)
    Sigma = softplus(Sigma)
    W = softplus(W)

    # We now optimize distribution params for each motion and store means in mu_ms, covariances in A_ms, and for convenient K_mm_invs
    mu_ms = []; A_ms = []; K_mm_invs = []

    # All timeseries of the same motion is put into a list, an element of X_motion_lists and Y_motion_lists
    X_motion_lists = []; Y_motion_lists = []
    for _ in range(num_motion):
        X_motion_lists.append([]); Y_motion_lists.append([])
    for i in range(len(Y_list)):
        X_motion_lists[labels[i]].append(X_list[i])
        Y_motion_lists[labels[i]].append(Y_list[i])

    # For each motion, using trained kernel parameter in "pair" form to obtain optimal distribution params for each motion.
    for k in range(num_motion):
        kernel_params = (Sigma[k], W[k])
        mu_m, A_m, K_mm_inv = phi_opt(sigmoid(X_m@Z[k]), X_motion_lists[k], Y_motion_lists[k], sigma_y, kernel_params) 
        mu_ms.append(mu_m); A_ms.append(A_m); K_mm_invs.append(K_mm_inv)
    
    # Save model to path.
    model = {'X_m': X_m, 'Z': Z, 'Sigma': Sigma, 'W': W, 
             'mu_ms': mu_ms, 'A_ms': A_ms, 'K_mm_invs': K_mm_invs}
    np.save(model_path, model)
    return

def load_model(model_path):
    model = np.load(model_path + '.npy', allow_pickle=True)
    X_m, Z, Sigma, W = model.item().get('X_m'), model.item().get('Z'), model.item().get('Sigma'), model.item().get('W') 
    mu_ms, A_ms, K_mm_invs = model.item().get('mu_ms'), model.item().get('A_ms'), model.item().get('K_mm_invs')
    return X_m, Z, Sigma, W, mu_ms, A_ms, K_mm_invs

# interpolate series (X_test, Y_test) on time values X_m.
def interpolate(X_m, X_test, Y_test):
    m = X_m.shape[0]
    Y = np.zeros(m)
    for i in range(m):
        Y[i] = Y_test[int(X_m[i]*m)]
    return Y

def classify(X_test, Y_test, kernel_params_all_motions, X_m, Z, mu_ms, A_ms, K_mm_invs, mode='dtw'):
    """
    Classify by calculate distance between inducing (mean) values and interpolated test values at inducing pts.
    """
    num_motion = len(kernel_params_all_motions)
    ind = -1; min_ll = 1e9
    for k in range(num_motion):
        X_m_k = sigmoid(X_m @ Z[k])
        if mode == 'simple':
            Y = np.interp(X_m_k, X_test, Y_test)
            ll = ((mu_ms[k]-Y).T)@(mu_ms[k]-Y)
        elif mode == 'variational':
            Sigma, W = kernel_params_all_motions[k]
            K_mm = spectral_kernel(X_m_k, X_m_k, Sigma, W) + jitter(X_m_k.shape[0])
            K_mn = spectral_kernel(X_m_k, X_test, Sigma, W)
            trace_avg_all_comps = jnp.sum(W**2)
            y_n_k = Y_test.reshape(-1, 1) # shape (n, 1)
            ll = elbo_fn_from_kernel(K_mm, K_mn, y_n_k, trace_avg_all_comps, sigma_y=0.1)
        elif mode == 'dtw':
            mean, _ = q(X_test, X_m_k, kernel_params_all_motions[k], mu_ms[k], A_ms[k], K_mm_invs[k])
            # ll = jnp.log(jnp.linalg.det(covar)) + ((Y_test-mean).T)@jnp.linalg.inv(covar)@(Y_test-mean)
            # ll = dtw.distance(np.array(mean), np.array(Y_test))
            ll = ((mean-Y_test).T)@(mean-Y_test) 
        if ind == -1:
            ind = k; min_ll = ll
        elif min_ll > ll: 
            ind = k; min_ll = ll
    
    return ind

def test_classify(model_path, X_test_list, Y_test_list, true_labels, max_predictions=1000):
    # print('Number of motions to predict: ', len(set(true_labels)))

    # Extract optimal trained params
    X_m, Z, Sigma, W, mu_ms, A_ms, K_mm_invs = load_model(model_path)
    num_motion = Z.shape[0]
    kernel_params = []
    for k in range(num_motion):
        kernel_params.append((Sigma[k], W[k]))

    # Predict each trajectory/timeseries in the test dataset
    num_predicted = 0
    pred = []; gt = []
    pbar = tqdm(zip(X_test_list, Y_test_list), total=min(Y_test_list.shape[0], max_predictions), leave=False)
    for X_test, Y_test in pbar:
        # Get predict and ground truth motions
        pred_label = classify(X_test, Y_test, kernel_params, X_m, Z, mu_ms, A_ms, K_mm_invs)
        gt_label = true_labels[num_predicted]
        pbar.set_description(f'Predict: {pred_label}; gt: {gt_label}')

        # Append results to lists for final evaluation
        pred.append(pred_label); gt.append(gt_label)
        num_predicted += 1
        if num_predicted >= min(Y_test_list.shape[0], max_predictions):
            break

    # Accurary evaluation
    return accuracy(pred, gt)

def test_forecast(model_path, X_test, Y_test_list, labels):
    # Extract optimal trained params
    X_m, Z, Sigma, W, mu_ms, A_ms, K_mm_invs = load_model(model_path)
    num_motion = Z.shape[0]
    
    # Average prediction for each type of motion.
    mean_preds = []
    for k in range(num_motion):
        mean, _ = q(X_test, sigmoid(X_m @ Z[k]), (Sigma[k], W[k]), mu_ms[k], A_ms[k], K_mm_invs[k])
        mean_preds.append(mean)
    
    all_errors = [[] for _ in range(num_motion)]
    
    for i in range(len(Y_test_list)):
        label = labels[i]
        all_errors[label].append(RMSE(mean_preds[label], Y_test_list[i]))

    errs = np.zeros(num_motion)
    for i in range(num_motion):
        errs[i] = np.mean(np.array(all_errors[i]))
    
    return errs