# Motion Code

## I. Introduction
In this work, we employ variational inference and stochastic process modeling to develop a framework called
Motion Code. Please look at the tutorial notebook `tutorial_notebook.ipynb` to learn how to use Motion Code.

To initialize the model, use the code:
```python
from motion_code import MotionCode
model = MotionCode(m=10, Q=1, latent_dim=2, sigma_y=0.1)
```

For training the Motion Code, use:
```python
model.fit(X_train, Y_train, labels_train, model_path)
```

Motion Code performs both classification and forecasting.
- For the classification task, use:
  ```python
  model.classify_predict(X_test, Y_test)
  ```
- For the forecasting task, use:
  ```python
  mean, covar = model.forecast_predict(test_time_horizon, label=0)
  ```

All package prerequisites are given in `requirements.txt`.

## II. Interpretable Features
To learn how to generate interpretable features, please see `Pronunciation_Audio.ipynb` for further tutorial.
This notebook gets the audio data, trains a Motion Code model on the data, and plots the interpretable features
obtained from Motion Code's most informative timestamps.

## III. Benchmarking
The main benchmark file is `benchmarks.py`.

### III-A. Non Attention-Based Method
#### Running Benchmarks:
1. Classification benchmarking on basic dataset with noise:
   ```sh
   python benchmarks.py --dataset_type="basics" --load_existing_model=True --load_existing_data=True --output_path="out/classify_basics.csv"
   ```
2. Forecasting benchmarking on basic dataset with noise:
   ```sh
   python benchmarks.py --dataset_type="basics" --forecast=True --load_existing_model=True --load_existing_data=True --output_path="out/forecast_basics.csv"
   ```
3. Classification and forecasting benchmarking on (Pronunciation) Audio dataset:
   ```sh
   python benchmarks.py --dataset_type="pronunciation" --load_existing_model=True --load_existing_data=True --output_path="out/classify_pronunciation.csv"
   python benchmarks.py --dataset_type="pronunciation" --forecast=True --load_existing_model=True --load_existing_data=True --output_path="out/forecast_pronunciation.csv"
   ```
4. Benchmarking on Parkinson data for either PD setting 1 or PD setting 2:
   ```sh
   python benchmarks.py --dataset_type="parkinson_1" --load_existing_model=True --output_path="out/classify_parkinson_1.csv"
   python benchmarks.py --dataset_type="parkinson_2" --load_existing_model=True --output_path="out/classify_parkinson_2.csv"
   ```

### III-B. Attention-Based Method
We will use the Time Series Library ([TSLibrary](https://github.com/thuml/Time-Series-Library/)), stored in the `TSLibrary` folder.
To rerun all training, execute the bash script:
```sh
TSLibrary/attention_benchmark.sh
```
For efficiency, it is recommended to use existing models and run `collect_all_benchmarks.ipynb` to retrieve benchmark results.

### III-C. Collecting Benchmarks
To format all classification benchmarks in a highlighted manner, run:
```sh
collect_all_benchmarks.ipynb
```
The output `benchmark_results.html` file contains all classification benchmark results.

### III-D. Hyperparameter Details
#### Classification:
- **DTW:** `distance="dtw"`
- **TSF:** `n_estimators=100`
- **BOSS-E:** `max_ensemble_size=3`
- **Shapelet:** `estimator=RotationForest(n_estimators=3)`, `n_shapelet_samples=100`, `max_shapelets=10`, `batch_size=20`
- **SVC:** `kernel=mean_gaussian_tskernel`
- **LSTM-FCN:** `n_epochs=200`
- **Rocket:** `num_kernels=500`
- **Hive-Cote 2:** `time_limit_in_minutes=0.2`
- **Attention-based parameters**: Refer to `TSLibrary/attention_benchmark.sh`

#### Forecasting:
- **Exponential Smoothing:** `trend="add"`, `seasonal="additive"`, `sp=12`
- **ARIMA:** `order=(1, 1, 0)`, `seasonal_order=(0, 1, 0, 12)`
- **State-space:** `level="local linear trend"`, `freq_seasonal=[{"period": 12, "harmonics": 10}]`
- **TBATS:** `use_box_cox=False`, `use_trend=False`, `use_damped_trend=False`, `sp=12`, `use_arma_errors=False`, `n_jobs=1`

## IV. Visualization
The main visualization file is `visualize.py`.

### IV-A Motion Code Interpretability
To extract interpretable features from Motion Code, run:
```sh
python visualize.py --type="classify_motion_code" --dataset="PD setting 2"
```
Change the dataset argument as needed (e.g., `Pronunciation Audio`, `PD setting 1`, `PD setting 2`).

### IV-B Forecasting Visualization
1. To visualize forecasting with mean and variance:
   ```sh
   python visualize.py --type="forecast_mean_var" --dataset="ItalyPowerDemand"
   ```
2. To visualize forecasting with informative timestamps:
   ```sh
   python visualize.py --type="forecast_motion_code" --dataset="ItalyPowerDemand"
   ```

## V. File Structure
1. **Tutorial Notebooks:** `tutorial_notebook.ipynb`, `Pronunciation_Audio.ipynb`
2. **Data Folder:** Contains three subfolders:
   - `basics`: Basic datasets with noise.
   - `audio`: Pronunciation Audio dataset (not included due to size restrictions).
   - `parkinson`: Parkinson sensor dataset (not included due to size restrictions).
3. **Saved Models Folder:** Pretrained Motion Code models for experiments and benchmarking.
4. **Python Files:**
   - **Data Processing:** `data_processing.py`, `parkinson_data_processing.py`, `utils.py`, `MotionCodeTSC_create.ipynb` (create .ts data for Time Series Library)
   - **Motion Code Model:** `motion_code.py`, `motion_code_utils.py`, `sparse_gp.py`
   - **Benchmarking:** `benchmark.py` (Non-attention), `collect_all_benchmarks.ipynb` (All)
   - **Visualization:** `visualize.py`
5. **Time Series Library:** `TSLibrary` folder, containing `attention_benchmark.sh` for rerunning model training.
