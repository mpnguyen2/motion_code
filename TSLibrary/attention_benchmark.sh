#!/bin/bash

# Clear output folders
: '
echo "Clearing output folders..."

FOLDERS=("checkpoints" "results" "test_results")
for FOLDER in "${FOLDERS[@]}"; do
  if [ -d "$FOLDER" ]; then
    echo "Removing $FOLDER..."
    rm -r "$FOLDER"
  else
    echo "$FOLDER does not exist, skipping."
  fi
done

echo -e "Cleared output folders.\n"
'

# List of dataset names: 
DATASETS=(
  "PDSetting1"
  "PDSetting2"
  "PronunciationAudio"
  "ECGFiveDays"
  "FreezerSmallTrain"
  "HouseTwenty"
  "InsectEPGRegularTrain"
  "ItalyPowerDemand"
  "Lightning7"
  "MoteStrain"
  "PowerCons"
  "SonyAIBORobotSurface2"
  "UWaveGestureLibraryAll"
)

# List of models to train
MODELS=(
  "Informer"
  "Autoformer"
  "FEDformer"
  "ETSformer"
  "LightTS"
  "PatchTST"
  "Crossformer"
  "DLinear"
  "TimesNet"
  "iTransformer"
  "Mamba"
)

# Model-specific parameters
declare -A DMODEL_MAP=( ["iTransformer"]=2048 ["TimesNet"]=64)
DEFAULT_DMODEL=128  # Default for all other models

# Dataset-specific top_k values
declare -A TOP_K_MAP=( ["PDSetting1"]=6 ["PDSetting2"]=12 )
DEFAULT_TOP_K=10  # Default for all other datasets

for DATASET in "${DATASETS[@]}"; do
  # Assign dataset-specific top_k, default to 10 if not specified
  TOP_K=${TOP_K_MAP[$DATASET]:-$DEFAULT_TOP_K}

  for MODEL in "${MODELS[@]}"; do
    DMODEL=${DMODEL_MAP[$MODEL]:-$DEFAULT_DMODEL}  # Assign model-specific d_model if available

    echo -e "\nTraining $MODEL on $DATASET with top_k=$TOP_K\n"

    if [[ "$MODEL" == "iTransformer" ]]; then
      python -u run.py \
        --task_name classification \
        --is_training 1 \
        --root_path "./dataset/MotionCodeTSC/$DATASET/" \
        --model_id "$DATASET" \
        --model "$MODEL" \
        --data UEA \
        --e_layers 3 \
        --batch_size 16 \
        --d_model "$DMODEL" \
        --d_ff 256 \
        --top_k "$TOP_K" \
        --des 'Exp' \
        --itr 1 \
        --learning_rate 0.001 \
        --train_epochs 100 \
        --patience 10 \
        --enc_in 3  # Only for iTransformer

    elif [[ "$MODEL" == "TimesNet" ]]; then
      python -u run.py \
        --task_name classification \
        --is_training 1 \
        --root_path "./dataset/MotionCodeTSC/$DATASET/" \
        --model_id "$DATASET" \
        --model "$MODEL" \
        --data UEA \
        --e_layers 2 \
        --batch_size 16 \
        --d_model "$DMODEL" \
        --d_ff 256 \
        --top_k "$TOP_K" \
        --num_kernels 4 \
        --des 'Exp' \
        --itr 1 \
        --learning_rate 0.001 \
        --train_epochs 30 \
        --patience 10

    elif [[ "$MODEL" == "ETSformer" ]]; then
      python -u run.py \
        --task_name classification \
        --is_training 1 \
        --root_path "./dataset/MotionCodeTSC/$DATASET/" \
        --model_id "$DATASET" \
        --model "$MODEL" \
        --data UEA \
        --e_layers 3 \
        --d_layers 3 \
        --batch_size 16 \
        --d_model "$DMODEL" \
        --d_ff 256 \
        --top_k "$TOP_K" \
        --des 'Exp' \
        --itr 1 \
        --learning_rate 0.001 \
        --train_epochs 100 \
        --patience 10

    else
      python -u run.py \
        --task_name classification \
        --is_training 1 \
        --root_path "./dataset/MotionCodeTSC/$DATASET/" \
        --model_id "$DATASET" \
        --model "$MODEL" \
        --data UEA \
        --e_layers 3 \
        --batch_size 16 \
        --d_model "$DMODEL" \
        --d_ff 256 \
        --top_k "$TOP_K" \
        --des 'Exp' \
        --itr 1 \
        --learning_rate 0.001 \
        --train_epochs 100 \
        --patience 10

    fi
  done
done
