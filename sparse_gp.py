import numpy as np
import jax.numpy as jnp
import jax.scipy as jsp

from jax import jit, value_and_grad
from jax.config import config

config.update("jax_enable_x64", True)

# map from R to [0, 1]
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

def gaussian_kernel_diag(d, theta1):
  return jnp.full(shape=d, fill_value=jnp.sum(theta1**2))

def gaussian_kernel(X1, X2, theta):
    sqdist = jnp.sum(X1**2, 1).reshape(-1, 1) + jnp.sum(X2**2, 1) - 2*jnp.dot(X1, X2.T)
    return theta[1]**2 * jnp.exp(-0.5/theta[0]**2 * sqdist)

def multi_gaussian_kernel(X1, X2, theta0, theta1):
    sqdist = jnp.sum(X1**2, 1).reshape(-1, 1) + jnp.sum(X2**2, 1) - 2*jnp.dot(X1, X2.T)
    sqdist = sqdist.reshape(sqdist.shape[0], sqdist.shape[1], 1)
    return jnp.sum(theta1.reshape(1, 1, -1)**2 * jnp.exp(-0.5/theta0.reshape(1, 1, -1)**2 * sqdist), axis=2)

def pack_params(params):
    #params is a list
    flatten = []
    for p in params:
        flatten.extend(p.reshape(-1))
    return np.array(flatten)

def unpack_params(params, mode='simple_train', dims=[1, 1, 1]):
    '''
    theta0, theta1 dim d for code_train
    Theta1 list of theta1 with dim L x d
    Z with dim L x d; z with dim d
    big X_m with dim  m x d
    '''
    m, L, d = dims[0], dims[1], dims[2]
    if mode == 'simple_train':
        return jnp.array(params[:2]), jnp.array(params[2:]).reshape(-1, 1)

    if mode == 'code_train':
        cnt = d; theta0 = params[:cnt]
        Z = params[cnt:(cnt+L*d)].reshape(L, d); cnt += L*d
        Theta1 = params[cnt:(cnt+L*d)].reshape(L, d); cnt += L*d
        X_m = params[cnt:(cnt+m*d)].reshape(m, d)
        return jnp.array(theta0), jnp.array(Z), jnp.array(Theta1), jnp.array(X_m)
    
    if mode == 'code_pred':
        theta1 = params[:d]
        z = params[d:]
        return jnp.array(theta1), jnp.array(z)

def elbo_fn(X_list, y_list, indices, sigma, dims, mode='simple_train', trained_params=[]):
    # L is the number of motion
    # d is the dimension of motion code vector
    # m is the number of inducing points
    m, L = dims[0], dims[1]

    def elbo_fn_from_kernel(theta1, K_mm, K_mn, y):
        # n is the number of training samples
        n = y.shape[0]
        L = jnp.linalg.cholesky(K_mm)
        A = jsp.linalg.solve_triangular(L, K_mn, lower=True)/sigma
        AAT = A @ A.T
        B = jnp.eye(K_mn.shape[0]) + AAT
        LB = jnp.linalg.cholesky(B)
        c = jsp.linalg.solve_triangular(LB, A.dot(y), lower=True)/sigma

        lb = -n/2 * jnp.log(2*jnp.pi)
        lb -= jnp.sum(jnp.log(jnp.diag(LB)))
        lb -= n/2 * jnp.log(sigma**2)
        lb -= 0.5/sigma**2 * y.T.dot(y)
        lb += 0.5 * c.T.dot(c)
        lb -= 0.5/sigma**2 * jnp.sum(gaussian_kernel_diag(n, theta1))
        lb += 0.5 * jnp.trace(AAT)

        return -lb[0, 0]

    def elbo(params):
        if mode == 'simple_train':
            theta, X_m = unpack_params(params, mode, dims)
            theta = softplus(theta)
            X_m = sigmoid(X_m)
            K_mm = gaussian_kernel(X_m, X_m, theta) + jitter(m)
            loss = 0
            for i in range(len(X_list)):
                K_mn = gaussian_kernel(X_m, jnp.array(X_list[i]), theta)
                loss += elbo_fn_from_kernel(theta[1], K_mm, K_mn, jnp.array(y_list[i]))
            return loss/len(X_list)

        if mode == 'simple_pred':
            n = y_list[0].shape[0]
            prob = softmax(params).reshape(1, -1)
            K_mms_opt, K_mns_opt, theta1s_opt = trained_params[0], trained_params[1], trained_params[2]
            K_mm = (prob @ K_mms_opt.reshape(L, -1)).reshape(m, m)
            K_mn = (prob @ K_mns_opt.reshape(L, -1)).reshape(m, n)
            theta1 = (prob @ theta1s_opt.reshape(L, -1)).reshape(-1)
            return elbo_fn_from_kernel(theta1, K_mm, K_mn, y_list[0])

        if mode == 'code_train':
            # We train the common exponent factor for gaussian kernel theta0
            # For each type of motion k from 1 to L, z[k] is the motion code with dim d, 
            # and Theta[k] is the coefficients for multi-Gaussian kernel to be trained
            # Finally, a pack of inducing point X_m with size m x d
            theta0, Z, Theta1, X_m = unpack_params(params, mode, dims)
            theta0, Theta1 = softplus(theta0), softplus(Theta1)
            loss = 0
            for i in range(len(X_list)):
                k = indices[i]
                X_m_k = (X_m @ Z[k].reshape(-1, 1))
                K_mm = multi_gaussian_kernel(X_m_k, X_m_k, theta0, Theta1[k]) + jitter(X_m_k.shape[0])
                K_mn = multi_gaussian_kernel(X_m_k, X_list[i], theta0, Theta1[k])
                loss += elbo_fn_from_kernel(Theta1[k], K_mm, K_mn, y_list[i])
            return loss

        if mode == 'code_pred':
            # We predict the motion code z for the incoming data (y, X). (online version later)
            theta1, z = unpack_params(params, mode, dims)
            X_m_all, theta0 = trained_params[0], trained_params[1]
            X_m = X_m_all @ z.reshape(-1, 1)
            K_mm = multi_gaussian_kernel(X_m, X_m, theta0, theta1) + jitter(X_m.shape[0])
            K_mn = multi_gaussian_kernel(X_m, X_list[0], theta0, theta1)
            return elbo_fn_from_kernel(theta1, K_mm, K_mn, y_list[0])
    
    elbo_grad = jit(value_and_grad(elbo))

    def elbo_grad_wrapper(params):
        value, grads = elbo_grad(params)
        return np.array(value), np.array(grads)

    return elbo_grad_wrapper

# Find optimal mu_m and A_m: approximate distribution params for f_m
@jit
def phi_opt(theta, X_m, X_list, y_list, sigma):
    precision = 1.0/(sigma**2)
    B = len(X_list)
    # Get K_mm and its inverse
    K_mm = gaussian_kernel(X_m, X_m, theta) + jitter(X_m.shape[0])
    K_mm_inv = jnp.linalg.inv(K_mm)
    # Get list of K_nm and K_mn
    K_nm_list = []
    K_mn_list = []
    for j in range(B):
        K_nm_list.append(gaussian_kernel(X_list[j], X_m, theta))
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
    mu_m = (factor @ K_mn_list[0]).dot(y_list[0])
    for j in range(1, B):
        mu_m += (factor @ K_mn_list[j]).dot(y_list[j])

    return mu_m, A_m, K_mm_inv

# Prediction
@jit
def q(X_test, theta, X_m, mu_m, A_m, K_mm_inv):
    K_ss = gaussian_kernel(X_test, X_test, theta)
    K_sm = gaussian_kernel(X_test, X_m, theta)
    K_ms = K_sm.T

    f_q = (K_sm @ K_mm_inv).dot(mu_m)
    f_q_cov = K_ss - K_sm @ K_mm_inv @ K_ms + K_sm @ K_mm_inv @ A_m @ K_mm_inv @ K_ms

    return f_q, f_q_cov