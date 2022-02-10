from utils import read_trajectory
from latent_motion import find_trajectories_latent
from cluster import cluster_dtw

# Pipeline for transforming each trajectory to latent timeseries and then cluster those time series
if __name__ == '__main__':
    input_file = 'data/motions.txt'
    trajectories = read_trajectory(input_file)
    latent_dim = 3
    latent_dims = [latent_dim]*len(trajectories)
    latent_timeseries = find_trajectories_latent(trajectories, latent_dims, num_epochs=1000)
    threshold_dist = 2
    cluster_dtw(latent_timeseries, threshold_dist)