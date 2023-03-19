import time, argparse
import matplotlib.pyplot as plt
from train_utils import train, test
from preprocessing import load_UCR_UEA_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CLI arguments')
    parser.add_argument('--dataset', type=str, default='artificial')
    parser.add_argument('--num_inducing_pts', type=int, default=10, help='Number of inducing points for the model')
    parser.add_argument('--num_kernel_comps', type=int, default=1, help='Number of components for kernel')
    parser.add_argument('--motion_code_dim', type=int, default=2, help='Dimension of motion code')
    args = parser.parse_args()

    model_path='saved_models/' + args.dataset

    # Train
    m = args.num_inducing_pts
    latent_dim = args.motion_code_dim
    Q = args.num_kernel_comps
    num_samples, X_train, Y_train, labels_train = load_UCR_UEA_data(name=args.dataset, mode='train', visualize=False)

    start_time = time.time()
    print('Done loading dataset. Start training...\n')
    train(X_train, Y_train, labels_train, model_path=model_path, m=m, Q=Q, latent_dim=latent_dim, sigma_y=0.1)
    print('Training take %.2f seconds' % (time.time()-start_time))

    # Test
    print('\nStart testing...')
    num_samples, X_test, Y_test, labels_test = load_UCR_UEA_data(name=args.dataset, mode='test', visualize=False)
    test(model_path, X_test, Y_test, labels_test)