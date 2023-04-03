import numpy as np
import jax.numpy as jnp
from scipy.optimize import minimize

from sparse_gp import *
from utils import *

def optimize_motion_codes(X_list, Y_list, labels, model_path, m=10, Q=8, latent_dim=3, sigma_y=0.1):
    '''
    Main algorithm to optimize all variables for the Motion Code model.
    '''
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

def classify_predict_helper(X_test, Y_test, kernel_params_all_motions, X_m, Z, mu_ms, A_ms, K_mm_invs, mode='dt'):
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
        elif mode == 'dt':
            mean, _ = q(X_test, X_m_k, kernel_params_all_motions[k], mu_ms[k], A_ms[k], K_mm_invs[k])
            # ll = jnp.log(jnp.linalg.det(covar)) + ((Y_test-mean).T)@jnp.linalg.inv(covar)@(Y_test-mean)
            ll = ((mean-Y_test).T)@(mean-Y_test) 
        if ind == -1:
            ind = k; min_ll = ll
        elif min_ll > ll: 
            ind = k; min_ll = ll
    
    return ind