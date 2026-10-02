"""
client.py — TCP sender (encoder side) for the medical image transfer pipeline.

Loads a chest X-ray image, runs model.encode(),
encrypts the latent tensor with AES-256-GCM, and sends it over TCP.

Usage:
    python client.py \
        --host 127.0.0.1 \
        --port 9999 \
        --image dataset/mimic-cxr-dataset/some_image.jpg \
        --model unet_one \
        --checkpoint unet/checkpoints/best_model.pth \
        --key aes_key.bin \
        --height 256 \
        --width 256 \
        --base_channels 64
"""

import os
import sys
import time
import json
import socket
import argparse
from pathlib import Path

from PIL import Image
import torch
import torchvision.transforms as T

from src.network.crypto import LatentCrypto, tcp_send

# ──────────────────────────────────────────────────────────────────────────────
# Model registry (same as server)
# ──────────────────────────────────────────────────────────────────────────────

def load_model(model_name: str, base_channels: int, checkpoint_path: str, device: torch.device):
    model_name = model_name.lower()

    if model_name == "res_ae":
        from src.models.res_ae import ResAE
        model = ResAE(in_channels=1, base_channels=base_channels)
    elif model_name == "res_ae_deep":
        from src.models.res_ae_deep import ResAEDeep
        model = ResAEDeep(in_channels=1, base_channels=base_channels)
    elif model_name == "res_cbam_xl":
        from src.models.res_cbam_xl import ResAutoencoderXLCBAM
        model = ResAutoencoderXLCBAM(in_channels=1, base_channels=base_channels)
    elif model_name == "res_se_xl":
        from src.models.res_se_xl import ResAutoencoderXLSE
        model = ResAutoencoderXLSE(in_channels=1, base_channels=base_channels)
    elif model_name == "inception_ae":
        from src.models.inception_ae import InceptionResAE
        model = InceptionResAE(in_channels=1, base_channels=base_channels)
    elif model_name == "unet_noskip":
        from src.models.unet_no_skip import UNet
        model = UNet(in_channels=1, base_channels=base_channels)
    elif model_name == "unet_one":
        from src.models.unet_one import UNet
        model = UNet(in_channels=1, base_channels=base_channels)
    elif model_name == "unet_res_xl":
        from src.models.res_se_xl_skips import ResAutoencoderXLSE
        model = ResAutoencoderXLSE(in_channels=1, base_channels=base_channels)
    else:
        raise ValueError(f"Unknown model '{model_name}'")

    if checkpoint_path is not None:
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        epoch = ckpt.get("epoch", "?")
        val_loss = ckpt.get("best_val_loss", "?")
        print(f"[client] Loaded {model_name} checkpoint (epoch {epoch}, val_loss {val_loss})")
    else:
        print(f"[client] Initialized random {model_name} model (no checkpoint)")
        
    model.to(device)
    model.eval()
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

def load_image(path: str, height: int, width: int, device: torch.device) -> torch.Tensor:
    """Load image exactly as dataloader.py does."""
    img = Image.open(path).convert("L")
    transform = T.Compose([
        T.Resize((height, width)),
        T.ToTensor(),
    ])
    tensor = transform(img).unsqueeze(0)  # add batch dimension: [1, 1, H, W]
    return tensor.to(device)


# ──────────────────────────────────────────────────────────────────────────────
# Client
# ──────────────────────────────────────────────────────────────────────────────

def run_client(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[client] Device: {device}")

    if not os.path.exists(args.key):
        print(f"[client] Key file not found: {args.key}. Run 'python crypto.py' to generate.")
        sys.exit(1)
    key = LatentCrypto.load_key(args.key)

    model = load_model(args.model, args.base_channels, args.checkpoint, device)

    # ── Load Image ──────────────────────────────────────────────────────
    t0 = time.perf_counter()
    img_tensor = load_image(args.image, args.height, args.width, device)
    print(f"[client] Loaded {args.image}  shape={tuple(img_tensor.shape)}")

    timings = {}

    # ── Encode ──────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    with torch.no_grad():
        latent = model.encode(img_tensor)
    timings["encode_ms"] = (time.perf_counter() - t0) * 1000
    print(f"[client] Encoded  ({timings['encode_ms']:.1f} ms)")

    # ── Encrypt + Pack ──────────────────────────────────────────────────
    t0 = time.perf_counter()
    wire_bytes = LatentCrypto.pack(latent, key, model_name=args.model)
    timings["pack_encrypt_ms"] = (time.perf_counter() - t0) * 1000
    print(f"[client] Encrypted & Packed ({timings['pack_encrypt_ms']:.1f} ms) "
          f"→ {len(wire_bytes):,} bytes to send")

    # ── Transmit ────────────────────────────────────────────────────────
    print(f"[client] Connecting to {args.host}:{args.port} …")
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((args.host, args.port))

    t0 = time.perf_counter()
    tcp_send(s, wire_bytes)
    timings["tcp_send_ms"] = (time.perf_counter() - t0) * 1000
    print(f"[client] Transmitted ({timings['tcp_send_ms']:.1f} ms)")

    s.close()

    # ── Save timings ────────────────────────────────────────────────────
    timings_path = "results/client_timings.json"
    os.makedirs("results", exist_ok=True)
    with open(timings_path, "w") as f:
        json.dump(timings, f, indent=2)
    print(f"[client] Timings → {timings_path}")
    for k, v in timings.items():
        print(f"         {k:20s}: {v:.2f} ms")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(description="TCP sender — encode + encrypt X-ray image")
    p.add_argument("--host",         default="127.0.0.1",                          help="Server address")
    p.add_argument("--port",         default=9999, type=int,                         help="TCP port")
    p.add_argument("--image",        required=True,                                  help="Path to input .jpg image")
    p.add_argument("--model",        default="unet_one",                             help="Model architecture name")
    p.add_argument("--checkpoint",   default="unet/checkpoints/best_model.pth",      help="Path to .pth checkpoint")
    p.add_argument("--key",          default="aes_key.bin",                          help="AES key file")
    p.add_argument("--height",       default=256, type=int,                          help="Image height")
    p.add_argument("--width",        default=256, type=int,                          help="Image width")
    p.add_argument("--base_channels",default=64, type=int,                           help="base_channels used during training")
    return p.parse_args()


if __name__ == "__main__":
    run_client(get_args())
