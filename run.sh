rm -rf out/multiple/* &&
rm -rf data/artificial/* &&
python preprocessing.py --num_pts 1000 --num_train 500 --num_test 100 --sigma 0.1 &&
python train.py --num_inducing_pts 50