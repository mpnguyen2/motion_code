import time, argparse
import numpy as np
import jax.numpy as jnp
from scipy.optimize import minimize
#from jax.scipy.optimize import minimize

from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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
        method='L-BFGS-B', jac=True, callback=opt_callback)
    print('Inducing pts, motion codes, and kernel params successfully optimized: ', res.success)
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

def test(model_path, X_test_list, Y_test_list, true_labels, max_predictions=1000):
    print('Number of motions to predict: ', len(set(true_labels)))

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
        pred_label = simple_predict(X_test, Y_test, kernel_params, X_m, Z, mu_ms, A_ms, K_mm_invs)
        gt_label = true_labels[num_predicted]
        pbar.set_description(f'Predict: {pred_label}; gt: {gt_label}')

        # Append results to lists for final evaluation
        pred.append(pred_label); gt.append(gt_label)
        num_predicted += 1
        if num_predicted >= min(Y_test_list.shape[0], max_predictions):
            break

    # Accurary evaluation
    print('Accuracy is:', accuracy(pred, gt))