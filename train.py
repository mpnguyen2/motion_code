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

def train(prefix, m=10, Q=8, latent_dim=3, sigma_y=0.1):
    # Load and process data
    index_to_motion, _, X_list, Y_list, indices = load_data(prefix)
    #min_X, max_X = find_min_max_from_data_list(X_list)
    num_motion = index_to_motion.shape[0]
    num_comp = Y_list[0].shape[1]
    dims = (num_motion, num_comp, m, latent_dim, Q)

    # We optimize X_m, Z, and Kernel parameters including Sigma, Mu, W, Phi, Theta  
    # np.tile(np.linspace(min_X, max_X, m), (1, latent_dim)).reshape 
    # X_m_start = np.repeat(sigmoid_inv(np.linspace(0.1, 0.9, m)).reshape(1, -1), num_motion, axis=0) 
    X_m_start = np.repeat(sigmoid_inv(np.linspace(0.1, 0.9, m)).reshape(1, -1), latent_dim, axis=0).swapaxes(0, 1)
    Z_start = np.ones((num_motion, latent_dim))
    Sigma_start = softplus_inv(np.ones((num_motion, Q)))
    #Mu_start = np.zeros((num_motion, num_comp, Q))
    W_start = softplus_inv(np.ones((num_motion, Q)))
    #Phi_start = np.zeros((num_motion, num_comp, Q))
    #Theta_start = np.zeros((num_motion, num_comp, Q))

    res = minimize(fun=elbo_fn(X_list, Y_list, indices, sigma_y, dims),
        x0 = pack_params([X_m_start, Z_start, Sigma_start, W_start]),
        method='L-BFGS-B', jac=True, callback=opt_callback)
    print('Inducing pts, motion codes, and kernel params successfully optimized: ', res.success)
    X_m, Z, Sigma, W = unpack_params(res.x, dims=dims)
    Sigma = softplus(Sigma)
    W = softplus(W)
    print("Latent codes are: ")
    print(np.array(Z))

    # Transform kernel parameter to "pair" form
    #Sigma_ij, Alpha_ij = get_param_matrices_from_core_params(Sigma, W)

    # We now optimize distribution params for each motion and store means in mu_ms, covariances in A_ms, and for convenient K_mm_invs
    mu_ms = []; A_ms = []; K_mm_invs = []

    # All timeseries of the same motion is put into a list, an element of X_motion_lists and Y_motion_lists
    X_motion_lists = []; Y_motion_lists = []
    for _ in range(num_motion):
        X_motion_lists.append([]); Y_motion_lists.append([])
    for i in range(len(Y_list)):
        X_motion_lists[indices[i]].append(X_list[i])
        Y_motion_lists[indices[i]].append(Y_list[i])

    # For each motion, using trained kernel parameter in "pair" form to obtain optimal distribution params for each motion.
    for k in range(num_motion):
        kernel_params_ij = (Sigma[k], W[k])
        mu_m, A_m, K_mm_inv = phi_opt(sigmoid(X_m@Z[k]), X_motion_lists[k], Y_motion_lists[k], sigma_y, kernel_params_ij) 
        mu_ms.append(mu_m); A_ms.append(A_m); K_mm_invs.append(K_mm_inv)

    # Choose simple equally space test data and produce predicted results to sanity check the training
    print('\nSimple sanity check (plot results saved in out/multiple folder)...')
    X_test = np.linspace(-0.1, 1.1, 30).reshape(-1, 1); means = []; covars = []
    for k in range(num_motion):
        kernel_params_ij = (Sigma[k], W[k])
        mean, covar = q(X_test, sigmoid(X_m@Z[k]), kernel_params_ij, mu_ms[k], A_ms[k], K_mm_invs[k])
        means.append(mean); covars.append(covar)
    plot_motion_types(index_to_motion, X_motion_lists, Y_motion_lists, X_test, 
            means=means, covars=covars)
    print('Sanity check done.')

    return (X_m, Z, Sigma, W, mu_ms, A_ms, K_mm_invs), index_to_motion
    
def test(prefix, index_to_motion, trained_params, max_predictions=100):
    # Load and process test data
    index_to_motion, motion_names, X_list, Y_list, _ = load_data(prefix)
    print('Number of motions to predict: ', len(set(motion_names)))

    # Dimension list
    num_motion = index_to_motion.shape[0]

    # Extract optimal trained params
    X_m, Z, Sigma, W, mu_ms, A_ms, K_mm_invs = trained_params
    #Sigma_ij, Alpha_ij = get_param_matrices_from_core_params(Sigma, W)
    kernel_params_ijs = []
    for k in range(num_motion):
        kernel_params_ijs.append((Sigma[k], W[k]))

    # Predict each trajectory/timeseries in the test dataset
    num_predicted = 0
    pred = []; gt = []
    pbar = tqdm(zip(X_list, Y_list), total=min(len(Y_list), max_predictions), leave=False)
    for X_test, Y_test in pbar:
        # Get predict and ground truth motions
        pred_index = simple_predict(X_test, Y_test, kernel_params_ijs, X_m, Z, mu_ms, A_ms, K_mm_invs)
        pred_motion = index_to_motion[pred_index]
        gt_motion = motion_names[num_predicted]
        pbar.set_description(f'Predict: {pred_motion}; gt: {gt_motion}')

        # Append results to lists for final evaluation
        pred.append(pred_motion); gt.append(gt_motion)
        num_predicted += 1
        if num_predicted >= min(len(Y_list), max_predictions):
            break
    # Final evaluation
    print('Precision is:', precision(pred, gt))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CLI arguments')
    parser.add_argument('--train_mode', type=str, default='train', help='Type of training/exploration')
    parser.add_argument('--data_mode', type=str, default='artificial')
    parser.add_argument('--num_inducing_pts', type=int, default=10, help='Number of inducing points for the model')
    parser.add_argument('--num_kernel_comps', type=int, default=1, help='Number of components for kernel')
    parser.add_argument('--motion_code_dim', type=int, default=2, help='Dimension of motion code')
    args = parser.parse_args()

    if args.data_mode == 'artificial':
        prefix = 'data/artificial/'
    elif args.data_mode == 'sound':
        prefix = 'data/sound/processed/'
    else:
        prefix = 'data/auslan/processed/'

    m = args.num_inducing_pts
    latent_dim = args.motion_code_dim
    Q = args.num_kernel_comps
    max_predictions = 200

    if args.train_mode == 'explore':
        print('\n--------------------------------------------')
        print('Exploring data...')
        index_to_motion, motion_names, X_list_data, Y_list_data, indices = load_data(prefix=prefix+'train')
        X_list = []; y_list = []
        for X_data in X_list_data:
            X_list.append(X_data.reshape(-1))
        explore_coord = 0
        for Y_data in Y_list_data:
            y_list.append(Y_data[:, explore_coord])
        plot_timeseries(index_to_motion, X_list, y_list, indices)
        print('Visualization saved.\n')

    # Motion code classification
    if args.train_mode == 'train':
        start_time = time.time()
        print('\n--------------------------------------------')
        print('Number of inducing points: ', m)
        print('Number of kernel component: ', Q)
        print('Motion code dimension: ', latent_dim)
        print('Training...')
        trained_params, index_to_motion = train(prefix=prefix+'train', m=m, Q=Q, latent_dim=latent_dim, sigma_y=0.1)
        print('Done training. Take {:.3f} seconds'.format(time.time()-start_time))
        # Testing
        print('\n--------------------------------------------')
        print('Testing...')
        test(prefix=prefix+'test', index_to_motion=index_to_motion,  
            trained_params=trained_params, max_predictions=max_predictions)