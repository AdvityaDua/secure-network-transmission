#!/bin/bash
echo "Running unet_noskip sweep..."
uv run python scripts/adversarial_attack.py --train_csv dataset/mimic-cxr/train_subset.csv --test_csv dataset/mimic-cxr/test_subset.csv --model unet_noskip --checkpoint checkpoints/unet_noskip/checkpoints/best_model.pth --dataset_sizes 500 5000

echo "Running unet_one sweep..."
uv run python scripts/adversarial_attack.py --train_csv dataset/mimic-cxr/train_subset.csv --test_csv dataset/mimic-cxr/test_subset.csv --model unet_one --checkpoint checkpoints/unet_one/checkpoints/best_model.pth --dataset_sizes 500 5000

echo "Generating plots and tables..."
uv run python scripts/plot_adversarial.py
