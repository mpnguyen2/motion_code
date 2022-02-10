"""
This file contains all clustering-related fct
"""

import numpy as np
from scipy.cluster.hierarchy import single, fcluster
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Dynamic warping distance btw time series s and t
# s[i] and t[j] are vectors (ndarray)
def DTWDistance(s, t):
    MAXN = 1e9
    n, m = s.shape[0], t.shape[0]
    dtw = MAXN*np.ones((n+1, m+1), dtype=float)
    dtw[0, 0] = 0
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = np.linalg.norm(s[i-1]-t[j-1])
            dtw[i, j] = cost + min(dtw[i-1, j], dtw[i, j-1], dtw[i-1, j-1])
    
    return dtw[n, m]

# Perform hierarchy clustering (single method) using dynamic warping distance
def cluster_dtw(time_series, threshold_dist, plot_file='plot_cluster.png'):
    # Find condense distance matrix
    m = len(time_series) # Number of trajectories
    dist = np.zeros((m*(m-1))//2, dtype=float)
    for j in range(m):
        for i in range(j):
            dist[m * i + j - ((i + 2) * (i + 1)) // 2] = DTWDistance(time_series[i], time_series[j])
    # Z is linkage matrix represent the dendrogram 
    Z = single(dist)
    print('Linkage matrix: ')
    print(Z)
    # From linkage matrix to final cluster label
    labels = fcluster(Z, threshold_dist, criterion='distance')
    print('Labels: ', labels)
    # Plot for visualization
    colors = cm.rainbow(np.linspace(0, 1, np.max(labels)))
    plt.scatter(np.arange(0, len(labels)), np.zeros(len(labels)), c=colors[labels-1])
    plt.savefig(plot_file)