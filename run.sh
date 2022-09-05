while getopts a: flag
do
    case "${flag}" in
        a) artificial=${OPTARG};;
    esac
done
rm -rf out/multiple/* &&
if [ $artificial == "T" ]; then
    echo "Using artificial data..."
    rm -rf data/artificial/* &&
    python preprocessing.py --num_pts 500 --num_train 30 --num_test 20 --sigma 0.1 &&
    python train.py --num_inducing_pts 10
else
    echo "Using auslan data..."
    rm -rf data/auslan/processed/* &&
    python preprocessing.py --mode auslan &&
    python train.py --data_mode auslan --num_inducing_pts 10
fi