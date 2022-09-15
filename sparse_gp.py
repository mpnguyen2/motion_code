import numpy as np
import jax.numpy as jnp
import jax.scipy as jsp

from jax import jit, value_and_grad
from jax.config import config

config.update("jax_enable_x64", True)

# Constants
TWO_PI_SQRT = jnp.sqrt(jnp.pi)

## Helper math fcts ##
def sigmoid(x):
    return 1/(1+jnp.exp(-x))

def sigmoid_inv(y):
    return np.log(y/(1-y))

def softmax(logits):
    exp_logits = jnp.exp(logits)
    return exp_logits/jnp.sum(exp_logits)

def softplus(X):
  return jnp.log(1+jnp.exp(X))

def softplus_inv(X):
  return np.log(np.exp(X)-1)

def jitter(d, value=1e-6):
    return jnp.eye(d)*value

## Methods for finding kernels from data ##
def get_param_matrices_from_core_params(Sigma, Mu, W, Phi, Theta):
    """
    Returns parameters for pairs of components with shape (num_comp, num_comp, Q) for each motion.

    Parameters
    ----------
    Sigma, Mu, W, Phi, Theta: ndarray
        Core kernel parameters with shape (num_motion, num_comp, Q)
    """
    num_motion = Sigma.shape[0]
    num_comp = Sigma.shape[1]
    Q = Sigma.shape[2]
    
    # Calculate Sigma
    Sigma_i = Sigma.reshape(num_motion, num_comp, 1, Q)
    Sigma_j = Sigma.reshape(num_motion, 1, num_comp, Q)
    Sigma_ij = 2 * Sigma_i * 1.0/(Sigma_i + Sigma_j) * Sigma_j

    # Calculate Mu
    Mu_i = Mu.reshape(num_motion, num_comp, 1, Q)
    Mu_j = Mu.reshape(num_motion, 1, num_comp, Q)
    Mu_ij = 1/(Sigma_i+Sigma_j) * (Mu_i*Sigma_j + Sigma_i*Mu_j)

    # Calculate Alpha
    W_i = W.reshape(num_motion, num_comp, 1, Q)
    W_j = W.reshape(num_motion, 1, num_comp, Q)
    W_ij = W_i * W_j * jnp.exp(-0.25 * (Sigma_i-Sigma_j) * 1.0/(Sigma_i+Sigma_j) * (Sigma_i-Sigma_j))
    Alpha_ij = W_ij * TWO_PI_SQRT * jnp.sqrt(Sigma_ij)
    
    # Calculate Theta and Phi
    Theta_ij = Theta.reshape(num_motion, num_comp, 1, Q) - Theta.reshape(num_motion, 1, num_comp, Q)
    Phi_ij = Phi.reshape(num_motion, num_comp, 1, Q) - Phi.reshape(num_motion, 1, num_comp, Q)

    return Sigma_ij, Mu_ij, Alpha_ij, Theta_ij, Phi_ij

def repeat_param_for_kernel(param, num_x1, num_x2):
    """
    Returns repeated parameter over a pair of point sets for vectorization.
    Returned result has shape (num_comp*num_x1, num_comp*num_x2, Q)
    
    Parameters
    ----------
    param: has shape (num_comp, num_comp, Q)
    num_x1, num_x2: number of time variable for X1 and X2

    """
    # Repeat parameter num_x1 * num_x2 times and reshape in the form of (num_comp, num_x1, num_comp, num_x2, Q)
    num_comp = param.shape[0]
    Q = param.shape[2]
    repeated_param = jnp.repeat(param.reshape(num_comp, num_comp, Q, 1), num_x1*num_x2, axis=-1)
    repeated_param = jnp.swapaxes(repeated_param, 2, 3)
    repeated_param = repeated_param.reshape(num_comp, num_comp, num_x1, num_x2, Q)
    repeated_param = jnp.swapaxes(repeated_param, 1, 2)
    
    return repeated_param.reshape(num_comp, num_x1, num_comp*num_x2, Q).reshape(num_comp*num_x1, num_comp*num_x2, Q)

def spectral_kernel(X1, X2, sigma_ij, mu_ij, alpha_ij, phi_ij, theta_ij):
    """
    Returns spectral kernel between two sets of points in X. 
    For simplicity, we assume points in X is time variable with dim 1
    
    Parameters
    ----------
    X1, X2: ndarray
        Two sets of time variables (1D vectors of length n where n is # of data points in timeseries).
    num_comp: int
        Number of components of vector-valued Gaussian process
    sigma_ij, mu_ij, w_ij, phi_ij, theta_ij: ndarray
        Kernel params of dim (num_comp, num_comp, Q). Here Q = # of components in kernel.
    """
    num_x1 = X1.shape[0]
    num_x2 = X2.shape[0]
    num_comp = sigma_ij.shape[0]
    Q = sigma_ij.shape[2]

    # Difference btw X1, X2.
    X12_no_repeat = X1.reshape(num_x1, 1) - X2.reshape(1, num_x2)
    X12 = jnp.repeat(X12_no_repeat, (num_comp**2)*Q).reshape(-1, (num_comp**2)*Q).swapaxes(0, 1)
    X12 = X12.reshape(num_x1, num_x2, num_comp, num_comp, Q)
    X12 = jnp.swapaxes(X12, 1, 2).reshape(num_x1*num_comp, num_x2, num_comp, Q).reshape(num_x1*num_comp, num_x2*num_comp, Q)

    # kernel parameters that are repeated to allow vectorization.
    sigma = repeat_param_for_kernel(sigma_ij, num_x1, num_x2)
    mu = repeat_param_for_kernel(mu_ij, num_x1, num_x2)
    alpha = repeat_param_for_kernel(alpha_ij, num_x1, num_x2)
    phi = repeat_param_for_kernel(phi_ij, num_x1, num_x2)
    theta = repeat_param_for_kernel(theta_ij, num_x1, num_x2)

    return jnp.sum(alpha * jnp.exp(-0.5 *(X12+theta)*sigma*(X12+theta)) * jnp.cos((X12+theta)*mu+phi), axis=-1)

## Pack/unpack parameters ##
def pack_params(params):
    '''
    Returns a single 1D vector
    
    Parameters
    ----------
    params is a list of parameters

    '''
    flatten = []
    for p in params:
        flatten.extend(p.reshape(-1))
    return np.array(flatten)

def unpack_params(params, dims):
    '''
    Returns unpacked X_m, Z, Sigma, Mu, W, Phi, Theta
    X_m is a pack of inducing point with shape (m, latent_dim)
    Z is all motion codes stacking together with shape (num_motion, latent_dim)
    (Sigma, Mu, W, Phi, Theta) are kernel params of all motions, each has shape (num_motion, num_comp, Q)    
    '''
    num_motion, num_comp, m, latent_dim, Q = dims
    cnt = 0
    X_m = params[cnt:cnt+m*latent_dim]; cnt += m*latent_dim
    Z = params[cnt:cnt+num_motion*latent_dim]; cnt += num_motion*latent_dim
    Sigma = params[cnt:cnt+num_motion*num_comp*Q]; cnt += num_motion*num_comp*Q
    Sigma = Sigma.reshape(num_motion, num_comp, Q)
    Mu = params[cnt:cnt+num_motion*num_comp*Q]; cnt += num_motion*num_comp*Q
    Mu = Mu.reshape(num_motion, num_comp, Q)
    W = params[cnt:cnt+num_motion*num_comp*Q]; cnt += num_motion*num_comp*Q
    W = W.reshape(num_motion, num_comp, Q)
    Phi = params[cnt:cnt+num_motion*num_comp*Q]; cnt += num_motion*num_comp*Q
    Phi = Phi.reshape(num_motion, num_comp, Q)
    Theta = params[cnt:cnt+num_motion*num_comp*Q]; cnt += num_motion*num_comp*Q
    Theta = Theta.reshape(num_motion, num_comp, Q)

    return jnp.array(X_m).reshape(m, latent_dim), jnp.array(Z).reshape(num_motion, latent_dim), Sigma, Mu, W, Phi, Theta

## ELBO functions ##
def elbo_fn_from_kernel(K_mm, K_mn, y, trace_avg_all_comps, sigma_y):
    """
    Calculate elbo function from given kernels and y-data
    """
    # n is the number of training samples
    n = y.shape[0]
    L = jnp.linalg.cholesky(K_mm)
    A = jsp.linalg.solve_triangular(L, K_mn, lower=True)/sigma_y
    AAT = A @ A.T
    B = jnp.eye(K_mn.shape[0]) + AAT
    LB = jnp.linalg.cholesky(B)
    c = jsp.linalg.solve_triangular(LB, A.dot(y), lower=True)/sigma_y

    lb = -n/2 * jnp.log(2*jnp.pi)
    lb -= jnp.sum(jnp.log(jnp.diag(LB)))
    lb -= n/2 * jnp.log(sigma_y**2)
    lb -= 0.5/sigma_y**2 * y.T.dot(y)
    lb += 0.5 * c.T.dot(c)
    lb -= 0.5/sigma_y**2 * n * trace_avg_all_comps
    lb += 0.5 * jnp.trace(AAT)

    return -lb[0, 0]

def elbo_fn(X_list, Y_list, indices, sigma_y, dims):
    """
    Returns ELBO function from a list of timeseries with each timeseries is a specific motion.
    
    Parameters
    ----------
    X_list: A list of timeseries's time variable, whose element has shape (n, ).
    Y_list: A list of timeseries's target/output variable, whose element has shape (n, num_comp)
    Here n is the number of data points in a particular timeseries.
    indices: map each timeseries to the motion (number) it represents.
    sigma_y: Target noise.
    dims: tuple of (num_motion, num_comp, m=num_inducing_pts, latent_dim)

    """

    # m is the number of inducing points
    _, num_comp, _, _, _ = dims

    def elbo(params):
        # X_m is a pack of inducing point with shape (m, latent_dim)
        # Z is all motion codes stacking together with shape (num_motion, latent_dim)
        # Currently, each motion has a separate set of params (sigma, mu, w, phi, theta)
        # They are stacked in (Sigma, Mu, W, Phi, Theta), with each has shape (num_motion, num_comp, Q)
        X_m, Z, Sigma, Mu, W, Phi, Theta = unpack_params(params, dims)
        Sigma = softplus(Sigma)
        W = softplus(W)
        
        # Get parameters for pairs of components for all motions. Each stacked param has shape (num_motion, num_comp, num_comp, Q)
        Sigma_ij, Mu_ij, Alpha_ij, Theta_ij, Phi_ij = get_param_matrices_from_core_params(Sigma, Mu, W, Phi, Theta)

        loss = 0
        for i in range(len(X_list)):
            k = indices[i]  # motion index of current timeseries
            X_m_k = sigmoid(X_m @ Z[k])
            K_mm = spectral_kernel(X_m_k, X_m_k, Sigma_ij[k], Mu_ij[k], Alpha_ij[k], Phi_ij[k], Theta_ij[k])
            K_mn = spectral_kernel(X_m_k, X_list[i], Sigma_ij[k], Mu_ij[k], Alpha_ij[k], Phi_ij[k], Theta_ij[k])
            trace_avg_all_comps = TWO_PI_SQRT * jnp.sum(W[k]**2)/num_comp
            y_n_k = jnp.swapaxes(Y_list[i], 0, 1).reshape(-1, 1) # shape (num_comp*n, 1)
            loss += elbo_fn_from_kernel(K_mm, K_mn, y_n_k, trace_avg_all_comps, sigma_y)
        
        return loss/len(X_list)

    elbo_grad = jit(value_and_grad(elbo))

    def elbo_grad_wrapper(params):
        value, grads = elbo_grad(params)
        return np.array(value), np.array(grads)

    return elbo_grad_wrapper

## Predict distribution, mean and covariance methods from trained kernel parameters and inducing pts ##
@jit
def phi_opt(X_m, X_list, Y_list, sigma_y, kernel_params_ij):
    """
    Find optimal mu_m and A_m: approximate distribution params for f_m.
    Note that mu_m and A_m are for a single motion with all timeseries data corresponding to that motion.
    """
    sigma_ij, mu_ij, alpha_ij, theta_ij, phi_ij = kernel_params_ij
    precision = 1.0/(sigma_y**2)
    B = len(X_list)

    # Get K_mm and its inverse
    K_mm = spectral_kernel(X_m, X_m, sigma_ij, mu_ij, alpha_ij, phi_ij, theta_ij)\
        + jitter(X_m.shape[0])
    K_mm_inv = jnp.linalg.inv(K_mm)
    
    # Get list of K_nm and K_mn
    K_nm_list = []
    K_mn_list = []
    for j in range(B):
        K_nm_list.append(spectral_kernel(X_list[j], X_m, sigma_ij, mu_ij, alpha_ij, phi_ij, theta_ij))
        K_mn_list.append(K_nm_list[j].T)

    # Get Sigma in mean and variance formulas
    Lambda = K_mm
    for j in range(B):
        Lambda += precision/B * K_mn_list[j] @ K_nm_list[j]
    Sigma = jnp.linalg.inv(Lambda)
    factor = 1/B*precision*K_mm @ Sigma

    # Calculate variance
    A_m = K_mm @ Sigma @ K_mm

    # Calculate mean
    y_n = jnp.swapaxes(Y_list[0], 0, 1).reshape(-1)
    mu_m = (factor @ K_mn_list[0]).dot(y_n)
    for j in range(1, B):
        y_n = jnp.swapaxes(Y_list[j], 0, 1).reshape(-1)
        mu_m += (factor @ K_mn_list[j]).dot(y_n)

    return mu_m, A_m, K_mm_inv

@jit
def q(X_test, X_m, kernel_params_ij, mu_m, A_m, K_mm_inv):
    """
    Distribution prediction for a new collection of time variables
    """
    sigma_ij, mu_ij, alpha_ij, phi_ij, theta_ij = kernel_params_ij
    K_ss = spectral_kernel(X_test, X_test, sigma_ij, mu_ij, alpha_ij, phi_ij, theta_ij)
    K_sm = spectral_kernel(X_test, X_m, sigma_ij, mu_ij, alpha_ij, phi_ij, theta_ij)
    K_ms = K_sm.T

    f_q = (K_sm @ K_mm_inv).dot(mu_m)
    f_q_cov = K_ss - K_sm @ K_mm_inv @ K_ms + K_sm @ K_mm_inv @ A_m @ K_mm_inv @ K_ms

    return f_q, f_q_cov

def simple_predict(X_test, Y_test, kernel_params_ijs, X_m, Z, mu_ms, A_ms, K_mm_invs):
    """
    Simple predict using argmin of negative log-likelihood over all possible classes
    """
    num_motion = len(kernel_params_ijs)
    ind = -1; min_ll = 0
    y_test = jnp.swapaxes(Y_test, 0, 1).reshape(-1)
    for k in range(num_motion):
        # Calculate likelihood conditioned on motion type k
        mean, covar = q(X_test, kernel_params_ijs[k], (X_m @ Z[k]), mu_ms[k], A_ms[k], K_mm_invs[k])
        ll = jnp.log(jnp.linalg.det(covar)) + ((y_test-mean).T)@jnp.linalg.inv(covar)@(y_test-mean)
        #ll = ((y_test-mean).T)@(y_test-mean)
        if ind == -1:
            ind = k; min_ll = ll[0][0]
        elif min_ll > ll[0][0]: 
            ind = k; min_ll = ll[0][0]
    
    return ind