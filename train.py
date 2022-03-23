import time
import pickle
import numpy as np
import jax.numpy as jnp
from scipy.optimize import minimize
#from jax.scipy.optimize import minimize

from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sparse_gp import *

Nfeval = 0

# pred and gt are both lists of str
def precision(pred, gt):
    return np.sum(np.array(pred)==np.array(gt))/len(pred)

# Multi-class F-score: TBA
def F_score(pred, gt, num_class):
    pass

# Visualize helper fcts
def plot_timeseries(index_to_motion, X_list, y_list, indices, colors, explore=True, X_ms=np.array([])):
    num_series = len(y_list)
    for i in range(num_series):
        plt.plot(X_list[i], y_list[i], c=colors[indices[i]], lw=0.5)
    plt.title('Time series')
    patches = []
    L = len(index_to_motion)
    for k in range(L):
        patches.append(mpatches.Patch(color=colors[k], label=index_to_motion[k]))
    plt.legend(handles=patches)
    if not explore:
        for k in range(L):
            plt.scatter(X_ms[k].reshape(-1), np.zeros(X_ms[k].shape[0]), color=colors[k])
    if explore:
        plt.savefig('out/plot.png')
    else:
        plt.savefig('out/inducing_pts.png')

# Calculate prediction
def predict(X_test, X_train_list, y_train_list, sigma, theta, X_m):
    mu_m, A_m, K_mm_inv = phi_opt(theta, X_m, X_train_list, y_train_list, sigma)
    return q(X_test, theta, X_m, mu_m, A_m, K_mm_inv)

def plot_motion_types(index_to_motion, X_lists, y_lists, colors, X_ms, X_m_test, sigma, thetas):
    for k in range(len(X_lists)):
        num_series = len(X_lists[k])
        for i in range(num_series):
            plt.plot(X_lists[k][i], y_lists[k][i], c=colors[k], lw=0.5)
        mean, covariance = predict(X_m_test.reshape(-1, 1), X_lists[k], y_lists[k], sigma, thetas[k], X_ms[k])
        std = np.sqrt(np.diag(covariance)).reshape(-1)
        mean = mean.reshape(-1)
        plt.plot(X_m_test, mean, c=colors[(k+1)%len(X_lists)], lw=0.5)
        plt.fill_between(X_m_test, mean+2*std, mean-2*std,
            color=colors[(k+2)%len(X_lists)], alpha=0.1)
        #plt.scatter(X_ms[k].reshape(-1), np.zeros(X_ms[k].shape[0]), color=colors[k])
        plt.legend(handles=[mpatches.Patch(color=colors[k], label=index_to_motion[k])])
        plt.savefig('out/multiple/' + index_to_motion[k] + '.png')
        plt.clf()
        
# Load data from condensed files (npz and small txt and pickle)
def load_data(prefix):
    # Load look-up dictionaries and array of motion names
    motion_to_index = pickle.load(open(prefix + '_motion_to_index.pickle', 'rb'))
    index_to_motion = np.loadtxt(prefix+'_index_to_motion.txt', dtype=str, delimiter=' ')
    motion_names = np.loadtxt(prefix+'_motion_names.txt', dtype=str, delimiter=' ')
    # Load main timeseries data and divide them into different samples of timeseries
    # Also record the appropriate indices correponding to their motion names for later training
    data = np.load(prefix+'_data.npz')
    X = data['X']; Y = data['Y']
    num_observations = data['num_observations']
    X_list = []; Y_list = []; indices = []
    for i in range(len(num_observations)-1):
        X_list.append(X[num_observations[i]:num_observations[i+1]])
        Y_list.append(Y[num_observations[i]:num_observations[i+1]])
        indices.append(motion_to_index[motion_names[i]])


    return index_to_motion, motion_names, X_list, Y_list, indices

# Currently only support scalar-valued fct. Thus, need to convert vector-valued to scalar
# Currently take first component from vector-valued
def vector_to_scalar_Y(X_list, Y_list, use_weight=False, coord=12, weights=np.array([])):
    max_X = 0; min_X = 1
    for X in X_list:
        max_X = max(max_X, np.max(X))
        min_X = min(min_X, np.min(X))
    y_list = []
    if use_weight:
        num_features = Y_list[0].shape[1]
        if weights.shape[0] == 0:
            weights = np.random.rand(num_features)
        for Y in Y_list:
            y_list.append(np.sum(Y*weights.reshape(1,-1), axis=1).reshape(-1, 1))
    else:
        for Y in Y_list:
            y_list.append(Y[:, coord:(coord+1)])
    return min_X, max_X, y_list

def callbackF(Xi):
    global Nfeval
    Nfeval += 1
    print(f'{Nfeval}th evaluation')

def train(prefix, m, d, mode='code', sigma=0.1, use_weight=False, weights=np.array([]), colors=[]):
    # Load and process data
    index_to_motion, _, X_list, y_list, indices = load_data(prefix)
    min_X, max_X, y_list = vector_to_scalar_Y(X_list, y_list, use_weight=use_weight, weights=weights)
    L = index_to_motion.shape[0]
    dims = [m, L, d]
    # Optimization
    if mode == 'simple':
        # Divide data into L bins corresponding to each type of motion
        X_lists = []; y_lists = []
        for _ in range(L):
            X_lists.append([]); y_lists.append([])
        for i in range(len(y_list)):
            X_lists[indices[i]].append(X_list[i])
            y_lists[indices[i]].append(y_list[i])
        # Optimize thetas, X_ms
        thetas = []; X_ms = []
        theta_start = softplus_inv(np.array([1., 1.]))
        #initial_val = min_X + (max_X-min_X)/3
        X_m_start = sigmoid_inv(np.linspace(0.1, 0.2, m))
        #thetas_start = np.tile(theta_start, (1, L)).swapaxes(0, 1)
        #X_ms_start = np.tile(X_m_start, (1, L)).swapaxes(0, 1).reshape(L, m, 1)
        # jax.scipy.optimize.minimize(fun, x0, args=(), *, method, tol=None, options=None)
        '''
        res = minimize(elbo_fn(X_list, y_list, indices, sigma, dims, mode='simple_train'),
            x0 = pack_params([jnp.array(thetas_start), jnp.array(X_ms_start)]),
            method='BFGS')
        '''
        print('Start inducing pts: ', sigmoid(X_m_start))
        for k in range(L):
            res = minimize(fun=elbo_fn(X_lists[k], y_lists[k], indices, sigma, 
                dims, mode='simple_train'),
                x0 = pack_params([jnp.array(theta_start), jnp.array(X_m_start)]),
                method='L-BFGS-B', jac=True)
            print('Type of motion: ', index_to_motion[k], '; Success: ', res.success, '; Value: ', res.fun)
            theta, X_m = unpack_params(res.x, mode='simple_train', dims=dims)
            X_m = sigmoid(X_m); theta = softplus(theta)# retransform domain
            print('Inducing pts: ', X_m.ravel())
            thetas.append(theta); X_ms.append(X_m)
        # Plot inducing pts
        #plot_timeseries(index_to_motion, X_list, y_list, indices, colors=colors, explore=False, X_ms=sigmoid(np.array(X_ms)))
        plot_motion_types(index_to_motion, X_lists, y_lists, colors, 
                X_ms=X_ms, X_m_test=np.linspace(-0.1, 1.1, 30), 
                sigma=sigma, thetas=thetas)

        return np.array(thetas), np.array(X_ms), index_to_motion

    if mode == 'code':
        # Optimize theta0, Z, Theta1, X_m
        theta0_start = np.ones(d)
        Theta1_start = np.ones((L, d))
        X_m_start = np.tile(np.linspace(min_X, max_X, m), (1, d)).reshape(m, d)
        Z_start = np.zeros((L, d))
        res = minimize(fun=elbo_fn(X_list, y_list, sigma, dims, mode='code_train'),
            x0 = pack_params([theta0_start, Z_start, Theta1_start, X_m_start]),
            method='L-BFGS-B', jac=True)
        theta0, Z, Theta1, X_m = unpack_params(res.x, mode='code_train', dims=dims)
        return theta0, Z, Theta1, X_m
    
def test(prefix, m, d, mode='code', sigma=0.1, params=[], max_predictions=100, use_weight=False, weights=np.array([])):
    # Load and process test data
    _, motion_names, X_list, y_list, indices = load_data(prefix)
    print('Number of motions to predict: ', motion_names.shape[0])
    _, _, y_list = vector_to_scalar_Y(X_list, y_list, use_weight=use_weight, weights=weights)
    thetas, X_ms, index_to_motion = params[0], params[1], params[2]
    L = index_to_motion.shape[0]; dims = [m, L, d]
    # Predict using given trained params stored in params
    if mode == 'simple':
        # Extract optimal trained params
        theta1s_opt = thetas[:, 1]; K_mms_opt = np.zeros((L, m, m))
        print('Getting K_mm kernel matrices...')
        for k in range(L):
            K_mms_opt[k] = gaussian_kernel(X_ms[k], X_ms[k], thetas[k])
        print('Done getting K_mm. Predicting...')
        # Predict
        num_predicted = 0
        pred = []; gt = []
        pbar = tqdm(zip(X_list, y_list), total=min(len(y_list), max_predictions), leave=False)
        for X, y in pbar:
            n = X.shape[0]
            K_mns_opt = np.zeros((L, m, n))
            for k in range(L):
                K_mns_opt[k] = gaussian_kernel(X_ms[k], X, thetas[k])
            res = minimize(fun=elbo_fn([X], [y], indices, sigma, dims, mode='simple_pred', 
                trained_params=[jnp.array(K_mms_opt), jnp.array(K_mns_opt), jnp.array(theta1s_opt)]),
                x0 = pack_params([0.2*jnp.ones(L)]),
                method='L-BFGS-B', jac=True)
            logit = res.x.reshape(-1)
            # Get predict and ground truth motions
            pred_motion = index_to_motion[np.argmax(logit)]
            gt_motion = motion_names[num_predicted]
            pbar.set_description(f'Predict: {pred_motion}; gt: {gt_motion}')
            # Append results to lists for final evaluation
            pred.append(pred_motion); gt.append(gt_motion)
            num_predicted += 1
            if num_predicted >= min(len(y_list), max_predictions):
                break
        # Final evaluation
        print('\nPrecision is:', precision(pred, gt))

    if mode == 'code':
        pass

if __name__ == '__main__':
    mode = 'simple'
    data_dir = 'data/auslan/processed'
    m = 10; d = 4
    max_predictions = 100
    color_list = ['red', 'blue', 'green', 'yellow', 'black', 'orange', 'brown', 'grey', 'purple', 'hotpink']

    if mode == 'explore':
        index_to_motion, motion_names, X_list_data, Y_list_data, indices = load_data(prefix=data_dir+'/train')
        X_list = []; y_list = []
        for X_data in X_list_data:
            X_list.append(X_data.reshape(-1))
        for Y_data in Y_list_data:
            y_list.append(Y_data[:, 12])
        plot_timeseries(index_to_motion, X_list, y_list, indices, colors=color_list)

    # Simple classification mode
    if mode == 'simple':
        start_time = time.time()
        print('\n\nTraining...')
        index_to_motion, thetas, X_ms = train(prefix=data_dir+'/train', m=m, d=d, mode='simple', colors=color_list)
        print('Done training. Take {:.3f} seconds'.format(time.time()-start_time))
        print('First testing...')
        '''
        trained_params = [index_to_motion, thetas, X_ms]
        test(prefix=data_dir+'/test1', m=m, d=d, mode='simple', params=trained_params, max_predictions=max_predictions)
        print('Done first testing. Second testing...')
        test(prefix=data_dir+'/test2', m=m, d=d, mode='simple', params=trained_params, max_predictions=max_predictions)
        print('Done.')
        '''
    # Latent code mode
    if mode =='code':
        theta0, Z, Theta1, X_m = train(data_dir+'/train', m, d, mode=mode)
        trained_params = [theta0, Z, Theta1, X_m]      