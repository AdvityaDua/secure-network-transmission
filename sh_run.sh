#!/bin/bash

set -e  # Stop script if any command fails

############################################
# CONFIG VARIABLES
############################################
EPOCHS=80
BATCH_SIZE=32
HEIGHT=256
WIDTH=256
BASE_CHANNELS=64
LEARNING_RATE=1e-4
DATASET_ROOT=./dataset/mimic-cxr-dataset

THRESHOLD_MB=2000
CHECK_EVERY=30

archive_results() {
  local run_name="$1"
  (cd ./results && tar -czvf "${run_name}.tar.gz" "${run_name}")
}

############################################
# WAIT FOR MIG SLICE
############################################
echo "Waiting for MIG GPU slice to be free..."

while true; do
  USED=$(nvidia-smi | grep -Eo '[0-9]+MiB / [0-9]+MiB' | head -n 1 | awk '{print $1}' | tr -d 'MiB')

  if [ -z "$USED" ]; then
    echo "Could not read MIG memory — retrying..."
    sleep $CHECK_EVERY
    continue
  fi

  echo "MIG memory used: ${USED} MB"

  if [ "$USED" -lt "$THRESHOLD_MB" ]; then
    echo "GPU slice looks free — starting training sequence"
    break
  fi

  sleep $CHECK_EVERY
done

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

sleep 5

############################################
# RES AE DEEP
############################################
echo "Starting Res AE Deep training..."

uv run python scripts/training/train_res_ae_deep.py \
  --epochs $EPOCHS \
  --learning_rate $LEARNING_RATE \
  --dataset_root $DATASET_ROOT \
  --checkpoint_path "./results/res_ae_deep/checkpoints/best_model.pth" \
  --results ./results/res_ae_deep \
  --positions AP \
  --batch_size $BATCH_SIZE \
  --shuffle True \
  --num_workers 4 \
  --height $HEIGHT \
  --width $WIDTH \
  --base_channels $BASE_CHANNELS

archive_results "res_ae_deep"

echo "Res AE Deep finished. Cooling down..."
sleep 5


############################################
# INCEPTION AE
############################################
echo "Starting Inception AE training..."

uv run python scripts/training/train_inceptionae.py \
  --epochs $EPOCHS \
  --learning_rate $LEARNING_RATE \
  --dataset_root $DATASET_ROOT \
  --checkpoint_path "./results/inception_ae/checkpoints/best_model.pth" \
  --results ./results/inception_ae \
  --positions AP \
  --batch_size $BATCH_SIZE \
  --shuffle True \
  --num_workers 4 \
  --height $HEIGHT \
  --width $WIDTH \
  --base_channels $BASE_CHANNELS

archive_results "inception_ae"

echo "Inception AE finished. Cooling down..."
sleep 5

############################################
# U-NET NO-SKIP
############################################
echo "Starting U-Net (no skip) training..."

uv run python scripts/training/train_unet.py \
  --epochs $EPOCHS \
  --learning_rate $LEARNING_RATE \
  --dataset_root $DATASET_ROOT \
  --checkpoint_path "./results/unet_noskip/checkpoints/best_model.pth" \
  --results ./results/unet_noskip \
  --positions AP \
  --batch_size $BATCH_SIZE \
  --shuffle True \
  --num_workers 4 \
  --height $HEIGHT \
  --width $WIDTH \
  --base_channels $BASE_CHANNELS

archive_results "unet_noskip"

echo "U-Net (no skip) finished. Cooling down..."
sleep 5

############################################
# U-NET ONE
############################################
echo "Starting U-Net One training..."

uv run python scripts/training/train_unet_one.py \
  --epochs $EPOCHS \
  --learning_rate $LEARNING_RATE \
  --dataset_root $DATASET_ROOT \
  --checkpoint_path "./results/unet_one/checkpoints/best_model.pth" \
  --results ./results/unet_one \
  --positions AP \
  --batch_size $BATCH_SIZE \
  --shuffle True \
  --num_workers 4 \
  --height $HEIGHT \
  --width $WIDTH \
  --base_channels $BASE_CHANNELS

archive_results "unet_one"

echo "All training runs completed successfully."
