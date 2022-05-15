# lie_cluster
We focus on learning latent code for time-series created by rigid object's motion for detection and recognition tasks

**File structure:**
* data: Hold trajectory/motion data. Currently, 2 folders are for artificial data and auslan (Australian sign language) data set
* sparse_gp: all sparse gaussian process training functions
* gpytorch_code: Old approach that optimize to find signature of each timeseries separately
* C++ code: generate Lie group-valued timeseries
* train.py: training model
* main.py: the main run pipeline: preprocessing data and model training
