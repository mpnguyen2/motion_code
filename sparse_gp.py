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
def spectral_kernel(X1, X2, sigma, alpha):
    num_x1 = X1.shape[0]
    num_x2 = X2.shape[0]
    X12 = (X1.reshape(num_x1, 1) - X2.reshape(1, num_x2)).reshape(num_x1, num_x2, 1)
    return jnp.sum(alpha.reshape(1, 1, -1) * jnp.exp(-0.5 * X12 * sigma.reshape(1, 1, -1) * X12), axis=-1)

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

def unpack_params_single(params, dims):
    m, Q = dims
    cnt = 0
    X_m = params[cnt:cnt+m]; cnt += m
    Sigma = params[cnt:cnt+Q]; cnt += Q
    W = params[cnt:cnt+Q]; cnt += Q

    return jnp.array(X_m).reshape(m),  Sigma, W

def unpack_params(params, dims):
    '''
    Returns unpacked X_m, Z, Sigma, W
    X_m is a pack of inducing point with shape (m, latent_dim)
    Z is all motion codes stacking together with shape (num_motion, latent_dim)
    (Sigma, W) are kernel params of all motions, each has shape (num_motion, Q)    
    '''
    num_motion, m, latent_dim, Q = dims
    cnt = 0
    X_m = params[cnt:cnt+m*latent_dim]; cnt += m*latent_dim
    Z = params[cnt:cnt+num_motion*latent_dim]; cnt += num_motion*latent_dim
    Sigma = params[cnt:cnt+num_motion*Q]; cnt += num_motion*Q
    Sigma = Sigma.reshape(num_motion, Q)
    W = params[cnt:cnt+num_motion*Q]; cnt += num_motion*Q
    W = W.reshape(num_motion, Q)
  
    return jnp.array(X_m).reshape(m, latent_dim), Z.reshape(num_motion, latent_dim), Sigma, W

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

def elbo_fn_single(X, Y, sigma_y, dims):
    """
    Returns ELBO function for a single time series.
    
    Parameters
    ----------
    X: Timeseries's time variable
    Y: Timeseries's target/output variable
    dims = (m, Q)
    """

    def elbo(params):
        # X_m is inducing pt (m, ), Sigma, W are kernel parameters of shape Q
        X_m, Sigma, W = unpack_params_single(params, dims)
        Sigma = softplus(Sigma)
        W = softplus(W)
        K_mm = spectral_kernel(X_m, X_m, Sigma, W) + jitter(X_m.shape[0])
        K_mn = spectral_kernel(X_m, X, Sigma, W)
        trace_avg_all_comps = jnp.sum(W**2)
        y_n_k = Y.reshape(-1, 1)
        
        return elbo_fn_from_kernel(K_mm, K_mn, y_n_k, trace_avg_all_comps, sigma_y)

    elbo_grad = jit(value_and_grad(elbo))

    def elbo_grad_wrapper(params):
        value, grads = elbo_grad(params)
        return np.array(value), np.array(grads)

    return elbo_grad_wrapper

def elbo_fn(X_list, Y_list, labels, sigma_y, dims):
    """
    Returns ELBO function from a list of timeseries with each timeseries is a specific motion.
    
    Parameters
    ----------
    X_list: A list of timeseries's time variable, whose element has shape (n, ).
    Y_list: A list of timeseries's target/output variable, whose element has shape (n, )
    Here n is the number of data points in a particular timeseries.
    labels: map each timeseries to the motion (number) it represents.
    sigma_y: Target noise.
    dims: tuple of (num_motion,  m=num_inducing_pts, latent_dim, Q). Recall Q is the number of terms in kernel.
    """

    def elbo(params):
        # X_m is a pack of inducing point with shape (m, latent_dim)
        # Z is all motion codes stacking together with shape (num_motion, latent_dim)
        # Currently, each motion has a separate set of params (sigma, mu, w, phi, theta)
        # They are stacked in (Sigma, W), with each has shape (num_motion, num_comp, Q)
        X_m, Z, Sigma, W = unpack_params(params, dims)
        Sigma = softplus(Sigma)
        W = softplus(W)

        loss = 0
        for i in range(len(X_list)):
            k = labels[i]  # label of the current timeseries
            X_m_k = sigmoid(X_m@Z[k])
            K_mm = spectral_kernel(X_m_k, X_m_k, Sigma[k], W[k]) + jitter(X_m_k.shape[0])
            K_mn = spectral_kernel(X_m_k, X_list[i], Sigma[k], W[k])
            trace_avg_all_comps = jnp.sum(W[k]**2)
            y_n_k = Y_list[i].reshape(-1, 1) # shape (n, 1)
            loss += elbo_fn_from_kernel(K_mm, K_mn, y_n_k, trace_avg_all_comps, sigma_y)
        
        return loss/len(X_list)

    elbo_grad = jit(value_and_grad(elbo))

    def elbo_grad_wrapper(params):
        value, grads = elbo_grad(params)
        return np.array(value), np.array(grads)

    return elbo_grad_wrapper

## Predict distribution, mean and covariance methods from trained kernel parameters and inducing pts ##
@jit
def phi_opt(X_m, X_list, Y_list, sigma_y, kernel_params):
    """
    Find optimal mu_m and A_m: approximate distribution params for f_m.
    Note that mu_m and A_m are for a single motion with all timeseries data corresponding to that motion.
    
    Parameters
    ----------
    X_m: inducing points of one motion
    X_list: A list of timeseries's time variables for this motion
    Y_list: A list of timeseries's target variable for this motion
    kernel_params: kernel parameters for gaussian approx of this motion
    """
    sigma, alpha= kernel_params

    precision = 1.0/(sigma_y**2)
    B = len(X_list)

    # Get K_mm and its inverse
    K_mm = spectral_kernel(X_m, X_m, sigma, alpha)\
        + jitter(X_m.shape[0])
    K_mm_inv = jnp.linalg.inv(K_mm)
    
    # Get list of K_nm and K_mn
    K_nm_list = []
    K_mn_list = []
    for j in range(B):
        K_nm_list.append(spectral_kernel(X_list[j], X_m, sigma, alpha))
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
    y_n = Y_list[0]
    mu_m = (factor @ K_mn_list[0]).dot(y_n)
    for j in range(1, B):
        y_n = Y_list[j]
        mu_m += (factor @ K_mn_list[j]).dot(y_n)

    return mu_m, A_m, K_mm_inv

@jit
def q(X_test, X_m, kernel_params, mu_m, A_m, K_mm_inv):
    """
    Distribution prediction for a new collection of time variables
    """
    sigma, alpha = kernel_params
    K_ss = spectral_kernel(X_test, X_test, sigma, alpha)
    K_sm = spectral_kernel(X_test, X_m, sigma, alpha)
    K_ms = K_sm.T

    f_q = (K_sm @ K_mm_inv).dot(mu_m)
    f_q_cov = K_ss - K_sm @ K_mm_inv @ K_ms + K_sm @ K_mm_inv @ A_m @ K_mm_inv @ K_ms

    return f_q, f_q_cov