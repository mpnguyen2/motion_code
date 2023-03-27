while getopts m: flag
do
    case "${flag}" in
        m) mode=${OPTARG};;
    esac
done
rm -rf out/multiple/* &&
if [ $mode == "C" ]; then
    echo "Performing classification on motion code model and benchmark models..."
    python benchmarks.py > logs/benchmarks_classify.txt &&
    python train.py > logs/motion_code_classify.txt
else
    echo "Performing forecasting on motion code model and benchmark models..."
    python benchmarks.py --forecast True > logs/benchmarks_forecast.txt &&
    python train.py --forecast True > logs/motion_code_forecast.txt
fi