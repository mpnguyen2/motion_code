while getopts m: flag
do
    case "${flag}" in
        m) mode=${OPTARG};;
    esac
done
rm -rf out/multiple/* &&
if [ $mode == "A" ]; then
    echo "Using artificial data..."
    rm -rf data/artificial/* &&
    python preprocessing.py --mode artificial --num_pts 500 --num_train 30 --num_test 20 --sigma 0.1 &&
    python train.py --data_mode artificial --train_mode explore &&
    python train.py --num_inducing_pts 50
elif [ $mode == "S" ]; then
    echo "Using sound data"
    rm -rf data/sound/processed/* &&
    python preprocessing.py --mode sound &&
    python train.py --data_mode sound --train_mode explore &&
    python train.py --data_mode sound --num_inducing_pts 10
else
    echo "Using auslan data..."
    rm -rf data/auslan/processed/* &&
    python preprocessing.py --mode auslan &&
    python train.py --data_mode auslan --num_inducing_pts 10
fi