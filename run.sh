while getopts m: flag
do
    case "${flag}" in
        m) mode=${OPTARG};;
    esac
done
rm -rf out/multiple/* &&
if [ $mode == "A" ]; then
    echo "Using all UCR data..."
    rm -rf out/multiple/* &&
    python train.py --dataset Yoga --num_inducing_pts 10 
fi