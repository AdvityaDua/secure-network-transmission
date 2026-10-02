"""
pipeline_test.py — End-to-end test harness for the encrypted transfer pipeline.

Runs the complete encoder → serialize → encrypt → TCP (localhost) → decrypt → deserialize → decoder
pipeline in a single process. Measures timings and verifies zero additional loss vs standalone model.

Usage:
    python pipeline_test.py \
        --image dataset/mimic-cxr-dataset/some_image.jpg \
        --model unet_one \
        --checkpoint unet/checkpoints/best_model.pth \
        --height 256 \
        --width 256 \
        --base_channels 64
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
import json
import socket
import threading
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.utils as vutils
from PIL import Image
import torchvision.transforms as T

from src.network.crypto import LatentCrypto, tcp_send, tcp_recv

# Re-use loaders from client/server
from src.network.client import load_model, load_image
from src.network.server import decode_latent

# ──────────────────────────────────────────────────────────────────────────────
# In-process TCP Server Thread
# ──────────────────────────────────────────────────────────────────────────────

class LocalReceiverThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind(("127.0.0.1", 0))  # OS assigned port
        self.sock.listen(1)
        self.port = self.sock.getsockname()[1]
        self.wire_bytes = None
        self.tcp_recv_ms = 0.0

    def run(self):
        conn, _ = self.sock.accept()
        t0 = time.perf_counter()
        self.wire_bytes = tcp_recv(conn)
        self.tcp_recv_ms = (time.perf_counter() - t0) * 1000
        conn.close()
        self.sock.close()


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline Test
# ──────────────────────────────────────────────────────────────────────────────

def run_test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # 1. Setup crypto key
    key_path = os.path.join(args.output_dir, "test_key.bin")
    key = LatentCrypto.generate_key(key_path)

    # 2. Load model & image
    model = load_model(args.model, args.base_channels, args.checkpoint, device)
    img_tensor = load_image(args.image, args.height, args.width, device)

    timings = {}

    print("\n--- STANDALONE BASELINE ---")
    with torch.no_grad():
        t0 = time.perf_counter()
        standalone_out = model(img_tensor)
        timings["standalone_forward_ms"] = (time.perf_counter() - t0) * 1000
    standalone_mse = F.mse_loss(standalone_out, img_tensor).item()
    print(f"Standalone forward: {timings['standalone_forward_ms']:.1f} ms")
    print(f"Standalone MSE:     {standalone_mse:.8f}")

    def execute_pipeline_path(quantize_fp16, label):
        print(f"\n--- PIPELINE EXECUTION ({label}) ---")
        receiver = LocalReceiverThread()
        receiver.start()
        
        t_pipeline_start = time.perf_counter()
        
        # [ENCODER]
        with torch.no_grad():
            latent = model.encode(img_tensor)
            
        # [PACK / ENCRYPT]
        meta, raw_bytes = LatentCrypto.serialize(latent, quantize_fp16=quantize_fp16)
        wire_bytes = LatentCrypto.pack(latent, key, model_name=args.model, quantize_fp16=quantize_fp16)
        payload_size = len(wire_bytes)
        
        # [TRANSMIT]
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(("127.0.0.1", receiver.port))
        tcp_send(s, wire_bytes)
        s.close()
        
        receiver.join()
        received_bytes = receiver.wire_bytes
        
        # [UNPACK / DECRYPT]
        received_latent, header = LatentCrypto.unpack(received_bytes, key)
        
        # [DECODER]
        pipeline_out = decode_latent(model, received_latent, device)
        
        total_time = (time.perf_counter() - t_pipeline_start) * 1000
        
        print(f"Total payload sent: {payload_size:,} bytes")
        print(f"End-to-end time:    {total_time:.1f} ms")
        
        pipeline_mse = F.mse_loss(pipeline_out, img_tensor).item()
        delta_mse = abs(pipeline_mse - standalone_mse)
        return pipeline_out, delta_mse, payload_size

    # Run Lossless (Crypto/TCP baseline)
    out_lossless, delta_lossless, size_lossless = execute_pipeline_path(quantize_fp16=False, label="LOSSLESS CRYPTO+TCP")
    
    # Run Quantized (Production Bandwidth Optimization)
    out_quantized, delta_quantized, size_quantized = execute_pipeline_path(quantize_fp16=True, label="FP16 QUANTIZED CRYPTO+TCP")

    print("\n--- VERIFICATION & BANDWIDTH TRADE-OFF ---")
    print(f"{'Metric':<25} | {'Lossless (FP32)':<20} | {'Quantized (FP16)':<20}")
    print("-" * 70)
    print(f"{'Payload Size (bytes)':<25} | {size_lossless:<20,} | {size_quantized:<20,}")
    print(f"{'Delta MSE (vs Standalone)':<25} | {delta_lossless:<20.8e} | {delta_quantized:<20.8e}")
    print("-" * 70)
    
    if delta_lossless < 1e-6:
        print("✅ SUCCESS: Lossless pipeline perfectly matches standalone model (Crypto/TCP layers add zero error).")
    else:
        print("❌ FAILED: Lossless pipeline adds unexpected error!")
        
    print(f"ℹ️  INFO: FP16 Quantization adds {delta_quantized:.8e} MSE overhead while saving {size_lossless - size_quantized:,} bytes.")

    print("\n--- OUTPUTS ---")
    
    # Save visual comparison
    comp_tensor = torch.cat([img_tensor, standalone_out, out_lossless], dim=0)
    comp_path = os.path.join(args.output_dir, "comparison.png")
    # Title isn't easily added to save_image, but layout is Input | Standalone | Pipeline
    vutils.save_image(comp_tensor, comp_path, nrow=3, normalize=False)
    print(f"Saved visual comparison (Input | Standalone | Pipeline) → {comp_path}")
    
    # Save timings
    timings_path = os.path.join(args.output_dir, "timing_report.json")
    existing_timings = []
    if os.path.exists(timings_path):
        try:
            with open(timings_path, "r") as f:
                existing_timings = json.load(f)
                if isinstance(existing_timings, dict):
                    existing_timings = [existing_timings]
        except Exception:
            pass
            
    timings["model"] = args.model
    existing_timings.append(timings)
    
    with open(timings_path, "w") as f:
        json.dump(existing_timings, f, indent=2)
    print(f"Saved metrics/timings → {timings_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="End-to-end test harness for encrypted pipeline")
    p.add_argument("--image",        required=True,                                  help="Path to input .jpg image")
    p.add_argument("--model",        default="unet_one",                             help="Model architecture name")
    p.add_argument("--checkpoint",   default="unet/checkpoints/best_model.pth",      help="Path to .pth checkpoint")
    p.add_argument("--height",       default=256, type=int,                          help="Image height")
    p.add_argument("--width",        default=256, type=int,                          help="Image width")
    p.add_argument("--base_channels",default=64, type=int,                           help="base_channels used during training")
    p.add_argument("--output_dir",   default="results/pipeline_test",                help="Output directory for reports")
    args = p.parse_args()
    
    run_test(args)
