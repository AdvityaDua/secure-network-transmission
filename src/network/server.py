"""
server.py — TCP receiver (decoder side) for the medical image transfer pipeline.

Receives an AES-256-GCM encrypted latent tensor over TCP,
decrypts it, and runs model.decode() to reconstruct the X-ray image.

Usage:
    python server.py \
        --host 0.0.0.0 \
        --port 9999 \
        --model unet_one \
        --checkpoint unet/checkpoints/best_model.pth \
        --key aes_key.bin \
        --base_channels 64 \
        --output_dir results/received

Supported --model values:
    res_ae | res_ae_deep | res_cbam_xl | res_se_xl
    inception_ae | unet_noskip | unet_one | unet_res_xl
"""

import os
import sys
import time
import json
import socket
import argparse
from pathlib import Path

import torch
import torchvision.utils as vutils

from src.network.crypto import LatentCrypto, tcp_recv


# ──────────────────────────────────────────────────────────────────────────────
# Model registry
# ──────────────────────────────────────────────────────────────────────────────

def load_model(model_name: str, base_channels: int, checkpoint_path: str, device: torch.device):
    """
    Instantiate the correct model class, load checkpoint weights,
    and return the model in eval mode.
    """
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
        raise ValueError(
            f"Unknown model '{model_name}'. "
            "Choose from: res_ae, res_ae_deep, res_cbam_xl, res_se_xl, "
            "inception_ae, unet_noskip, unet_one, unet_res_xl"
        )

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    epoch = ckpt.get("epoch", "?")
    val_loss = ckpt.get("best_val_loss", "?")
    print(f"[server] Loaded {model_name} checkpoint (epoch {epoch}, val_loss {val_loss})")
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Decode helper
# ──────────────────────────────────────────────────────────────────────────────

def decode_latent(model, latent, device: torch.device) -> torch.Tensor:
    """
    Run model.decode() on the latent.
    Handles both bottleneck-only and (bottleneck, skips) formats.
    """
    with torch.no_grad():
        if isinstance(latent, torch.Tensor):
            z = latent.to(device)
            return model.decode(z)
        else:
            bottleneck, skips = latent
            z     = bottleneck.to(device)
            skips = tuple(s.to(device) for s in skips)
            return model.decode(z, skips)


# ──────────────────────────────────────────────────────────────────────────────
# Server
# ──────────────────────────────────────────────────────────────────────────────

def run_server(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[server] Device: {device}")

    key = LatentCrypto.load_key(args.key)
    model = load_model(args.model, args.base_channels, args.checkpoint, device)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.host, args.port))
    srv.listen(1)
    print(f"[server] Listening on {args.host}:{args.port} …")

    conn, addr = srv.accept()
    print(f"[server] Connection from {addr}")

    timings = {}

    try:
        # ── Receive ──────────────────────────────────────────────────────
        t0 = time.perf_counter()
        wire_bytes = tcp_recv(conn)
        timings["tcp_recv_ms"] = (time.perf_counter() - t0) * 1000
        print(f"[server] Received {len(wire_bytes):,} bytes in {timings['tcp_recv_ms']:.1f} ms")

        # ── Decrypt ──────────────────────────────────────────────────────
        t0 = time.perf_counter()
        latent, header = LatentCrypto.unpack(wire_bytes, key)
        timings["decrypt_ms"] = (time.perf_counter() - t0) * 1000
        print(f"[server] Decrypted OK  ({timings['decrypt_ms']:.1f} ms)")
        print(f"[server] Header: model={header['model_name']}, "
              f"payload_bytes={header['payload_bytes']:,}")

        # ── Decode ───────────────────────────────────────────────────────
        t0 = time.perf_counter()
        recon = decode_latent(model, latent, device)
        timings["decode_ms"] = (time.perf_counter() - t0) * 1000
        print(f"[server] Decoded  ({timings['decode_ms']:.1f} ms)  "
              f"output shape: {tuple(recon.shape)}")

        # ── Save image ───────────────────────────────────────────────────
        out_path = os.path.join(args.output_dir, "received_recon.png")
        vutils.save_image(recon, out_path)
        print(f"[server] Saved reconstruction → {out_path}")

        # ── Save timings ─────────────────────────────────────────────────
        timings_path = os.path.join(args.output_dir, "server_timings.json")
        with open(timings_path, "w") as f:
            json.dump(timings, f, indent=2)
        print(f"[server] Timings → {timings_path}")
        for k, v in timings.items():
            print(f"         {k:20s}: {v:.2f} ms")

    finally:
        conn.close()
        srv.close()


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(description="TCP receiver — decrypt + decode X-ray latent")
    p.add_argument("--host",         default="0.0.0.0",                             help="Bind address")
    p.add_argument("--port",         default=9999, type=int,                         help="TCP port")
    p.add_argument("--model",        default="unet_one",                             help="Model architecture name")
    p.add_argument("--checkpoint",   default="unet/checkpoints/best_model.pth",      help="Path to .pth checkpoint")
    p.add_argument("--key",          default="aes_key.bin",                          help="AES key file")
    p.add_argument("--base_channels",default=64, type=int,                           help="base_channels used during training")
    p.add_argument("--output_dir",   default="results/received",                     help="Directory to save output image")
    return p.parse_args()


if __name__ == "__main__":
    run_server(get_args())
