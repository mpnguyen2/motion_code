rm -rf out/multiple/* &&
python preprocessing.py --num_points_per_series 5 --num_sin_series 1 --num_cos_series 1 --sigma 0.1 &&
python train.py --num_inducing_pts 2