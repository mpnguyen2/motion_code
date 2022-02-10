# lie_cluster
Currently we map each Lie group-valued trajectory to a latent time-series. Then given a set of trajectories, we cluster those trajectories by clustering the corresponding time-series

** File structure: **
* data: Hold trajectory/motion data. To be updated to have data from Blender
* sparse_gp: all sparse gaussian process training with multi-valued fct
* C++ code: currently for handling all Lie group/algebra
* cluster.py: should put all clustering-related functions here
* latent_motion.py: infer latent time-series from original rigid-object trajectory
* main.py: the main pipeline 
