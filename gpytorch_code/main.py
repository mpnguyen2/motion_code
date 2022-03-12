import argparse
from gpytorch_code.utils import read_trajectory
from latent_motion import find_trajectories_latent
from cluster import cluster_dtw, cluster_vector

# Pipeline for transforming each trajectory to latent timeseries and then cluster those time series
if __name__ == '__main__':
    # Arguments parser
    parser = argparse.ArgumentParser(description='CLI argument for running encode and cluster pipeline')
    parser.add_argument('--mode', type=str, default='time', help='Mode to encode trajectories')
    parser.add_argument('--latent_dim', type=int, default=1, help='Latent dimension')
    parser.add_argument('--num_epochs', type=int, default=1, help='Number of training epochs for sparse GP')
    parser.add_argument('--num_cluster', type=int, default=2, help='Threshold for vector clustering case')
    parser.add_argument('--threshold_dist', type=float, default=0.3, help='Threshold for time-series clustering (dendrogram) case')
    args = parser.parse_args()
    # Main pipeline
    input_file = 'data/motions.txt'
    trajectories = read_trajectory(input_file)
    print('Encoding...')
    latent_dims = [args.latent_dim]*len(trajectories)
    latents = find_trajectories_latent(trajectories, latent_dims, num_epochs=args.num_epochs, mode=args.mode)
    print('Done encoding. Clustering...')
    if args.mode != 'space':
        cluster_dtw(latents, args.threshold_dist)
    else:
        cluster_vector(latents, args.num_cluster)
    print('\nDone clustering.')