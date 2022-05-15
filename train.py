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

# The training currently only support scalar-valued fct. 
# Thus, need to convert vector-valued to scalar-valued fct
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

def train(prefix, m, d, mode='code', sigma=0.01, use_weight=False, weights=np.array([]), colors=[]):
    # Load and process data
    index_to_motion, _, X_list, y_list, indices = load_data(prefix)
    min_X, max_X, y_list = vector_to_scalar_Y(X_list, y_list, coord=0, use_weight=use_weight, weights=weights)
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
        # DEBUG
        '''
        for k in range(L):
            X_lists[k] = X_lists[k][:1]
            y_lists[k] = y_lists[k][:1]
        '''
        # END DEBUG
        # For training, we optimize thetas, X_ms, mu_ms, A_ms, K_mm_invs
        # Then we do a simple test on equally spaced X-data before passing optimized params to next stage
        thetas = []; X_ms = []; mu_ms = []; A_ms = []; K_mm_invs = []
        # Starting theta and X_m for all types of motion (with free domains)
        theta_start = softplus_inv(np.array([1., 1.]))
        X_m_start = sigmoid_inv(np.linspace(0.1, 0.9, m))
        # Optimize theta and X_m for each type of motions by maximizing the upperbound of the ELBO (of likelihood)
        for k in range(L):
            res = minimize(fun=elbo_fn(X_lists[k], y_lists[k], indices, sigma, 
                dims, mode='simple_train'),
                x0 = pack_params([jnp.array(theta_start), jnp.array(X_m_start)]),
                method='L-BFGS-B', jac=True)
            print('Type of motion: ', index_to_motion[k], '; Successfully optimized: ', res.success)
            theta, X_m = unpack_params(res.x, mode='simple_train', dims=dims)
            X_m = sigmoid(X_m); theta = softplus(theta) # retransform domain
            thetas.append(theta); X_ms.append(X_m)
            # After getting inducing point X_m, get its optimal q-distribution with mean mu_m and covariance A_m
            # And also inverse of K_mm for easier calculation later
            mu_m, A_m, K_mm_inv = phi_opt(theta, X_m, X_list, y_list, sigma) 
            mu_ms.append(mu_m); A_ms.append(A_m); K_mm_invs.append(K_mm_inv)
        # Choose simple equally space test data and produce predicted results to sanity check the training
        print('\nSimple sanity check (plot results saved in out/multiple folder)...')
        X_test = np.linspace(-0.1, 1.1, 30).reshape(-1, 1); means = []; covars = []
        for k in range(L):
            mean, covar = q(X_test, thetas[k], X_ms[k], mu_ms[k], A_ms[k], K_mm_invs[k])
            means.append(mean); covars.append(covar)
        # Plot (each type of motion separately) training data and test prediction on X_test
        plot_motion_types(index_to_motion, X_lists, y_lists, X_test, 
                means=means, covars=covars, colors=colors)
        print('Sanity check done.')
        return (thetas, X_ms, mu_ms, A_ms, K_mm_invs), index_to_motion

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
    
def test(prefix, index_to_motion, m, d, trained_params, mode='code', max_predictions=100):
    # Load and process test data
    index_to_motion, motion_names, X_list, y_list, _ = load_data(prefix)
    _, _, y_list = vector_to_scalar_Y(X_list, y_list, coord=0)
    print('Number of motions to predict: ', len(set(motion_names)))
    # Dimension list
    L = index_to_motion.shape[0]; dims = [m, L, d]
    # Predict using given trained params stored in the tuple trained_params
    if mode == 'simple':
        # Extract optimal trained params
        thetas, X_ms, mu_ms, A_ms, K_mm_invs = trained_params
        # Predict each trajectory/timeseries in the test dataset
        num_predicted = 0
        pred = []; gt = []
        pbar = tqdm(zip(X_list, y_list), total=min(len(y_list), max_predictions), leave=False)
        for X_test, y_test in pbar:
            # Get predict and ground truth motions
            pred_index = simple_predict(X_test, y_test, thetas, X_ms, mu_ms, A_ms, K_mm_invs)
            pred_motion = index_to_motion[pred_index]
            gt_motion = motion_names[num_predicted]
            pbar.set_description(f'Predict: {pred_motion}; gt: {gt_motion}')
            #print(f'Predict: {pred_motion}; gt: {gt_motion}')
            # Append results to lists for final evaluation
            pred.append(pred_motion); gt.append(gt_motion)
            num_predicted += 1
            if num_predicted >= min(len(y_list), max_predictions):
                break
        # Final evaluation
        print('Precision is:', precision(pred, gt))

    if mode == 'code':
        pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CLI arguments')
    parser.add_argument('--mode', type=str, default='simple', help='Training mode')
    parser.add_argument('--data_mode', type=str, default='artificial')
    parser.add_argument('--num_inducing_pts', type=int, default=2, help='Number of inducing points for the model')
    parser.add_argument('--motion_code_degree', type=int, default=8, help='Dimension of motion code')
    args = parser.parse_args()

    if args.data_mode == 'artificial':
        prefix = 'data/artificial/artificial'
    else:
        prefix = 'data/auslan/processed/train'

    m = args.num_inducing_pts; d = args.motion_code_degree
    max_predictions = 200
    color_list = ['red', 'blue', 'green', 'yellow', 'black', 'orange', 'brown', 'grey', 'purple', 'hotpink']

    if args.mode == 'explore':
        index_to_motion, motion_names, X_list_data, Y_list_data, indices = load_data(prefix=prefix)
        X_list = []; y_list = []
        for X_data in X_list_data:
            X_list.append(X_data.reshape(-1))
        for Y_data in Y_list_data:
            y_list.append(Y_data[:, 12])
        plot_timeseries(index_to_motion, X_list, y_list, indices, colors=color_list)

    # Simple classification mode
    if args.mode == 'simple':
        start_time = time.time()
        print('\n--------------------------------------------')
        print('Number of inducing points: ', args.num_inducing_pts)
        print('Training...')
        trained_params, index_to_motion = train(prefix=prefix, m=m, d=d, mode='simple', colors=color_list)
        print('Done training. Take {:.3f} seconds'.format(time.time()-start_time))
        print('\n--------------------------------------------')
        print('Testing on the same dataset...')
        test(prefix=prefix, index_to_motion=index_to_motion, m=m, d=d, 
            trained_params=trained_params, mode='simple', max_predictions=max_predictions)

    # Latent code mode
    if args.mode =='code':
        theta0, Z, Theta1, X_m = train(prefix, m, d, mode=args.mode)
        trained_params = [theta0, Z, Theta1, X_m]      