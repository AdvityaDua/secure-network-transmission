import json

notebook = {
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    },
    "accelerator": "GPU"
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "intro"
      },
      "source": [
        "# Adversarial Compute-Cost Sweep\n",
        "This notebook runs the adversarial attack sweeps and automatically plots Table 4 and Figure 2 for your paper.\n",
        "\n",
        "**CRITICAL:** Make sure you are using a GPU! Go to `Runtime > Change runtime type` and select `T4 GPU`."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {
        "id": "mount_drive"
      },
      "outputs": [],
      "source": [
        "from google.colab import drive\n",
        "drive.mount('/content/drive')"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "setup_md"
      },
      "source": [
        "## Setup Codebase\n",
        "Unzip the package we created on your Mac into the Colab environment."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {
        "id": "unzip"
      },
      "outputs": [],
      "source": [
        "# Unzip the codebase from your Google Drive into /content/encrypt\n",
        "!unzip -q \"/content/drive/MyDrive/colab_package.zip\" -d /content/encrypt"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "install_md"
      },
      "source": [
        "## Install Dependencies\n",
        "We use `uv` to manage the environment and install PyTorch and required libraries rapidly."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {
        "id": "install_uv"
      },
      "outputs": [],
      "source": [
        "%cd /content/encrypt\n",
        "!curl -LsSf https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL=\"/usr/local\" sh\n",
        "!uv venv\n",
        "!uv pip install -r pyproject.toml\n",
        "# Ensure correct CUDA version of torch is installed for Colab\n",
        "!uv pip install torch torchvision torchaudio"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "run_sweep_md"
      },
      "source": [
        "## Run Adversarial Sweep\n",
        "This will train the surrogate decoder for `unet_noskip` and `unet_one` on 500 and 2500 images. Since we are on a GPU, this should be very fast!"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {
        "id": "run_sweep"
      },
      "outputs": [],
      "source": [
        "%cd /content/encrypt\n",
        "# Create a results folder directly on your Google Drive so progress is saved instantly!\n",
        "!mkdir -p \"/content/drive/MyDrive/Adversarial_Results\"\n",
        "\n",
        "# Run for Baseline (unet_noskip)\n",
        "!uv run python scripts/adversarial_attack.py --train_csv dataset/mimic-cxr/train_subset.csv --test_csv dataset/mimic-cxr/test_subset.csv --model unet_noskip --checkpoint checkpoints/unet_noskip/checkpoints/best_model.pth --dataset_sizes 500 2500 --results_dir \"/content/drive/MyDrive/Adversarial_Results\"\n",
        "\n",
        "# Run for Compression Model (unet_one)\n",
        "!uv run python scripts/adversarial_attack.py --train_csv dataset/mimic-cxr/train_subset.csv --test_csv dataset/mimic-cxr/test_subset.csv --model unet_one --checkpoint checkpoints/unet_one/checkpoints/best_model.pth --dataset_sizes 500 2500 --results_dir \"/content/drive/MyDrive/Adversarial_Results\"\n",
        "\n",
        "# Run for Proposed Defense Architecture (unet_res_xl)\n",
        "!uv run python scripts/adversarial_attack.py --train_csv dataset/mimic-cxr/train_subset.csv --test_csv dataset/mimic-cxr/test_subset.csv --model unet_res_xl --checkpoint checkpoints/unet_res_xl/checkpoints/best_model.pth --dataset_sizes 500 2500 --results_dir \"/content/drive/MyDrive/Adversarial_Results\"\n",
        "\n",
        "print(\"✅ Sweep complete and results safely saved in your Google Drive folder: Adversarial_Results!\")"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "plot_md"
      },
      "source": [
        "## Generate Table 4 and Figure 2\n",
        "We run the plotting script to generate the final LaTeX table and Matplotlib charts."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {
        "id": "plot_sweep"
      },
      "outputs": [],
      "source": [
        "%cd /content/encrypt\n",
        "!uv run python scripts/plot_adversarial.py\n",
        "\n",
        "from IPython.display import Image, display\n",
        "print(\"\\n--- Figure 2 Preview ---\")\n",
        "display(Image(filename=\"results/figure2.png\"))\n",
        "\n",
        "print(\"\\n✅ Don't forget to download the results/figure2.png and results/table_4.txt files!\")"
      ]
    }
  ]
}

with open("/Volumes/Data/Research Paper/Medical Data Transfer Network/encrypt/Adversarial_Sweep_Colab.ipynb", "w") as f:
    json.dump(notebook, f, indent=2)
