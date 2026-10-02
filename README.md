<p align="center">
  <strong>🔐 Secure Medical Image Transfer via Neural Latent Compression</strong>
</p>

<p align="center">
  <em>Privacy-preserving transfer of medical images using deep autoencoder latent representations and AES-256-GCM encryption over TCP.</em>
</p>

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#models">Models</a> •
  <a href="#setup">Setup</a> •
  <a href="#usage">Usage</a> •
  <a href="#security">Security</a> •
  <a href="#project-structure">Structure</a> •
  <a href="#citation">Citation</a>
</p>

---

## Overview

This repository implements a **secure medical image transfer pipeline** designed for transmitting chest X-ray images (e.g., from the MIMIC-CXR dataset) over untrusted networks **without exposing raw patient data**.

Instead of transmitting full-resolution medical images — which carry significant privacy and compliance risks (HIPAA, GDPR) — this system:

1. **Encodes** the image into a compact latent representation using a trained convolutional autoencoder
2. **Compresses** the latent tensor (FP16 quantization + zlib)
3. **Encrypts** the compressed payload with **AES-256-GCM** (authenticated encryption)
4. **Transmits** the encrypted data over TCP with length-prefixed framing
5. **Decrypts** and **decodes** on the receiving end to reconstruct the image

The key insight is that the **raw image never leaves the sender** — only an encrypted, compressed neural representation is transmitted, providing both **bandwidth efficiency** and **data privacy**.

### Key Features

- **8 autoencoder architectures** ranging from lightweight baselines to attention-augmented models with compressed skip connections
- **AES-256-GCM authenticated encryption** with tamper-proof headers (JSON metadata bound via AAD)
- **Skip-connection compression** — U-Net skip tensors are squeezed to 1 channel before transmission, then expanded back on the decoder side
- **End-to-end TCP client/server** with timing instrumentation at every stage (encode, encrypt, transmit, decrypt, decode)
- **Comprehensive evaluation scripts** — adversarial robustness sweeps, JPEG baselines, compute-cost profiling, and paper-ready result generation

---

## Architecture

```
┌─────────────────── SENDER (Client) ───────────────────┐     ┌─────────────────── RECEIVER (Server) ──────────────────┐
│                                                        │     │                                                         │
│  X-ray Image                                           │     │                                          Reconstructed   │
│  [1, 1, 256, 256]                                      │     │                                          X-ray Image     │
│       │                                                │     │                                               ▲          │
│       ▼                                                │     │                                               │          │
│  ┌──────────┐   ┌──────────────┐   ┌────────────────┐  │     │  ┌────────────────┐   ┌──────────────┐   ┌────┴─────┐   │
│  │ Encoder  │──▶│ FP16 + zlib  │──▶│  AES-256-GCM   │──╋─TCP─╋─▶│  AES-256-GCM   │──▶│ Decompress   │──▶│ Decoder  │   │
│  │ (model)  │   │ Compress     │   │  Encrypt       │  │     │  │  Decrypt       │   │ + Deserialize│   │ (model)  │   │
│  └──────────┘   └──────────────┘   └────────────────┘  │     │  └────────────────┘   └──────────────┘   └──────────┘   │
│                                                        │     │                                                         │
└────────────────────────────────────────────────────────┘     └─────────────────────────────────────────────────────────┘
```

### Wire Protocol

```
TCP Frame:  [ 8 B : msg_len (uint64 BE) ][ msg_len bytes : wire message ]

Wire Message:
  [ 4 B : header_len (uint32 BE) ]
  [ header_len B : UTF-8 JSON header ]  ← AAD (model name, shapes, nonce) — plaintext but tamper-proof
  [ remaining B : AES-GCM ciphertext ]  ← encrypted latent bytes + 16-byte GCM auth tag
```

---

## Models

All models are single-channel (grayscale) convolutional encoder–decoders operating on `[B, 1, H, W]` tensors with values in `[0, 1]`.

| Model | File | Key Features | Skip Connections | Params (64ch) |
|-------|------|-------------|-----------------|---------------|
| **ResAE** | `src/models/res_ae.py` | Plain residual blocks | ✗ | ~28.6M |
| **ResAE Deep** | `src/models/res_ae_deep.py` | 2× residual blocks per stage | ✗ | ~57M |
| **CBAM-XL** | `src/models/res_cbam_xl.py` | Channel + spatial attention (CBAM) | ✗ | ~110M |
| **SE-XL** | `src/models/res_se_xl.py` | SE channel + spatial attention | ✗ | ~110M |
| **InceptionResAE** | `src/models/inception_ae.py` | Multi-branch inception blocks | ✗ | ~45M |
| **UNet (no-skip)** | `src/models/unet_no_skip.py` | DoubleConv encoder–decoder | ✗ | ~27.9M |
| **UNet One** | `src/models/unet_one.py` | Compressed 1-ch skip connections | ✓ | ~31.4M |
| **UNet Res-XL** | `src/models/res_se_xl_skips.py` | SE + attention + compressed skips | ✓ | ~120M |

### Skip Compression Mechanism (UNet One / UNet Res-XL)

For models with skip connections, the full-width skip tensors at each encoder stage are **compressed to 1 channel** via `SkipConvDown` (1×1 conv) before encryption and transmission. On the decoder side, `SkipConvUp` expands them back to full width before concatenation. This drastically reduces the data volume that must be encrypted and transmitted while preserving multi-resolution spatial detail.

---

## Setup

### Prerequisites

- **Python** ≥ 3.12
- **[uv](https://docs.astral.sh/uv/)** (recommended package manager)
- **CUDA 12.1** (for GPU training on Linux; macOS uses CPU PyTorch)

### Installation

```bash
# Clone the repository
git clone https://github.com/AdvityaDua/secure-network-transmission.git
cd secure-network-transmission

# Install dependencies via uv
uv sync
```

### Generate an AES-256 Key

> ⚠️ **The key file is `.gitignore`-d and must be generated locally. Never commit encryption keys.**

```bash
uv run python -m src.network.crypto --key aes_key.bin
```

Both the client (sender) and server (receiver) must share the same key file through a secure out-of-band channel.

---

## Usage

### Training a Model

Each architecture has a dedicated training script under `scripts/training/`. All scripts support checkpoint resumption, TensorBoard logging, and per-epoch metrics CSV.

```bash
# Example: train the CBAM-XL autoencoder
uv run python scripts/training/train_res_ae_cbam.py \
  --epochs 80 \
  --learning_rate 1e-4 \
  --dataset_root ./dataset/mimic-cxr-dataset \
  --checkpoint_path ./results/res_ae_cbam/checkpoints/best_model.pth \
  --results ./results/res_ae_cbam \
  --positions AP \
  --batch_size 32 \
  --height 256 --width 256 \
  --base_channels 64
```

Monitor training:

```bash
tensorboard --logdir ./results
```

### Dataset Preparation

Generate CSVs from a MIMIC-CXR-style directory:

```bash
uv run python -m src.data.datamaker \
  --dataset_root ./dataset/mimic-cxr-dataset \
  --images_root ./dataset/mimic-cxr-dataset/official_data_iccv_final/files \
  --meta_root ./dataset/mimic-cxr-dataset/metadata.csv \
  --position AP
```

### Secure Image Transfer

**Start the server (receiver):**

```bash
uv run python -m src.network.server \
  --host 0.0.0.0 --port 9999 \
  --model unet_one \
  --checkpoint results/unet_one/checkpoints/best_model.pth \
  --key aes_key.bin \
  --output_dir results/received
```

**Send an image from the client:**

```bash
uv run python -m src.network.client \
  --host 127.0.0.1 --port 9999 \
  --image dataset/mimic-cxr-dataset/some_image.jpg \
  --model unet_one \
  --checkpoint results/unet_one/checkpoints/best_model.pth \
  --key aes_key.bin
```

### Loading a Trained Model (Inference)

```python
import torch
from src.models.unet_one import UNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(in_channels=1, base_channels=64).to(device)

ckpt = torch.load("results/unet_one/checkpoints/best_model.pth", map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# Encode → latent + skip tensors
x = torch.rand(1, 1, 256, 256, device=device)
with torch.no_grad():
    z, skips = model.encode(x)
    recon = model.decode(z, skips)
```

### Running Tests

```bash
uv run pytest tests/ -v
```

---

## Security

### Encryption Details

| Property | Value |
|----------|-------|
| Algorithm | AES-256-GCM (authenticated encryption with associated data) |
| Key Size | 256 bits (32 bytes) |
| Nonce | 96 bits (12 bytes), randomly generated per message |
| Auth Tag | 128 bits (appended to ciphertext by GCM) |
| AAD | JSON header (model name, tensor shapes, nonce) — plaintext but integrity-protected |

### Threat Model

- **Confidentiality**: The raw image and its latent representation are never transmitted in cleartext. An attacker intercepting the TCP stream sees only AES-256-GCM ciphertext.
- **Integrity**: GCM's authentication tag ensures any modification to the ciphertext or the header is detected and rejected (`InvalidTag` exception).
- **Replay Protection**: Each message uses a freshly generated random nonce; replaying a captured message to a stateful system would not produce meaningful results without the correct model weights.

### Key Management

- Keys are generated via `os.urandom()` (cryptographically secure PRNG)
- Key files (`*.bin`) are excluded from version control via `.gitignore`
- Keys must be distributed out-of-band (e.g., secure file transfer, HSM) before running the pipeline

---

## Project Structure

```
.
├── src/
│   ├── data/
│   │   ├── dataloader.py         # MimicDataset — loads images from CSV
│   │   └── datamaker.py          # Generates train/test CSVs from MIMIC-CXR layout
│   ├── models/
│   │   ├── res_ae.py             # ResAE — plain residual autoencoder
│   │   ├── res_ae_deep.py        # ResAE Deep — 2× blocks per stage
│   │   ├── res_cbam_xl.py        # CBAM-XL — channel + spatial attention
│   │   ├── res_se_xl.py          # SE-XL — squeeze-excitation + spatial attention
│   │   ├── inception_ae.py       # InceptionResAE — multi-branch blocks
│   │   ├── unet_no_skip.py       # UNet (no skips) — plain encoder-decoder
│   │   ├── unet_one.py           # UNet One — compressed skip connections
│   │   ├── unet_model.py         # Standard UNet reference
│   │   └── res_se_xl_skips.py    # UNet Res-XL — SE + attention + compressed skips
│   ├── network/
│   │   ├── crypto.py             # AES-256-GCM encryption + TCP wire protocol
│   │   ├── client.py             # TCP sender — encode + encrypt + transmit
│   │   └── server.py             # TCP receiver — receive + decrypt + decode
│   └── utils/
│       └── loss.py               # Sobel gradient loss for edge-aware training
├── scripts/
│   ├── training/                 # Training entrypoints for each architecture
│   │   ├── train_res_ae.py
│   │   ├── train_res_ae_deep.py
│   │   ├── train_res_ae_cbam.py
│   │   ├── train_resae_mp.py     # Mixed-precision (AMP) training
│   │   ├── train_resae_sobel.py  # Sobel gradient loss variant
│   │   ├── train_inceptionae.py
│   │   ├── train_unet.py
│   │   ├── train_unet_one.py
│   │   └── train_unet_res_xl.py
│   ├── evaluate_pipeline.py      # End-to-end pipeline evaluation
│   ├── adversarial_attack.py     # Adversarial robustness testing
│   ├── jpeg_baseline.py          # JPEG compression baseline comparison
│   └── ...                       # Additional utility & visualization scripts
├── tests/
│   └── test_crypto.py            # Unit tests for encryption pipeline
├── pyproject.toml                # Project config + dependencies (uv/pip)
├── uv.lock                       # Locked dependency versions
├── sh_run.sh                     # Batch training runner (GPU scheduling)
└── .gitignore
```

---

## Evaluation & Benchmarks

The repository includes extensive evaluation tooling:

- **`scripts/evaluate_pipeline.py`** — end-to-end pipeline test with PSNR/SSIM metrics and timing breakdown
- **`scripts/adversarial_attack.py`** — adversarial robustness sweeps (FGSM/PGD on latent space)
- **`scripts/jpeg_baseline.py`** — JPEG compression quality vs. neural compression comparison
- **`scripts/generate_paper_results.py`** — generates publication-ready metrics tables
- **`scripts/plot_combined_compute_cost.py`** — compute cost visualization across architectures

---

## License

This project is part of an ongoing research effort. Please contact the authors before using this code in derivative works.

---

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{dua2026securetransfer,
  title   = {Secure Medical Image Transfer via Neural Latent Compression},
  author  = {Dua, Advitya},
  year    = {2026},
  url     = {https://github.com/AdvityaDua/secure-network-transmission}
}
```
