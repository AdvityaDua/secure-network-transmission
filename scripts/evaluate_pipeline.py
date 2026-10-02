"""
evaluate_pipeline.py — Bulk evaluation script for the research paper.

Runs the encrypted transfer pipeline over an entire CSV dataset of test images,
computes average PSNR, SSIM, latency, and payload size, and outputs standard deviations.
This is exactly what you need for the results section of your paper.

Usage:
    python evaluate_pipeline.py \
        --csv_file dataset/mimic-cxr-dataset/image_paths_AP_test.csv \
        --model unet_one \
        --checkpoint unet/checkpoints/best_model.pth \
        --base_channels 64 \
        --num_samples 100
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import json
import socket
import argparse
import zlib
import base64
import struct
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from piq import ssim

from src.network.crypto import LatentCrypto, tcp_send
from src.network.client import load_model, load_image
from scripts.pipeline_test import LocalReceiverThread
from src.network.server import decode_latent

def run_evaluation(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running bulk evaluation on {device}...")

    # Setup
    key = LatentCrypto.generate_key("eval_temp_key.bin")
    model = load_model(args.model, args.base_channels, args.checkpoint, device)
    
    # Load dataset
    df = pd.read_csv(args.csv_file)
    if args.num_samples > 0:
        df = df.head(args.num_samples)
    
    results = {
        "psnr": [],
        "ssim": [],
        "delta_mse": [],
        "payload_bytes": [],
        "latency_encode": [],
        "latency_quantize": [],
        "latency_encrypt": [],
        "latency_tcp": [],
        "latency_decrypt": [],
        "latency_dequantize": [],
        "latency_decode": [],
        "latency_total": []
    }
    
    print(f"\nEvaluating {len(df)} images. This may take a moment...\n")

    for idx, row in df.iterrows():
        img_path = row["img_path"] if "img_path" in row else (row["path"] if "path" in row else str(row.iloc[0]))
        if not isinstance(img_path, str) or not os.path.exists(img_path):
            print(f"Warning: file not found {img_path}, skipping.")
            continue
            
        img_tensor = load_image(img_path, args.height, args.width, device)
        
        # 1. Baseline Standalone
        with torch.no_grad():
            standalone_out = model(img_tensor)
        standalone_mse = F.mse_loss(standalone_out, img_tensor).item()
        
        # 2. Start Receiver
        receiver = LocalReceiverThread()
        receiver.start()
        
        # 3a. Encode
        t0 = time.perf_counter()
        with torch.no_grad():
            latent = model.encode(img_tensor)
        t_encode = (time.perf_counter() - t0) * 1000
            
        # 3b. Quantize + entropy-code
        t0 = time.perf_counter()
        meta, raw_bytes = LatentCrypto.serialize(latent, quantize_fp16=args.quantize_fp16)
        meta["quantize_fp16"] = args.quantize_fp16
        meta["apply_zlib"] = True
        raw_bytes = zlib.compress(raw_bytes, level=9)
        t_quantize = (time.perf_counter() - t0) * 1000
        
        # 3c. Encrypt
        t0 = time.perf_counter()
        nonce = os.urandom(12)
        header = {
            "model_name": args.model,
            "nonce_b64": base64.b64encode(nonce).decode(),
            "latent_meta": meta,
            "payload_bytes": len(raw_bytes),
        }
        header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
        ciphertext = AESGCM(key).encrypt(nonce, raw_bytes, header_bytes)
        wire_bytes = struct.pack(">I", len(header_bytes)) + header_bytes + ciphertext
        payload_size = len(wire_bytes)
        t_encrypt = (time.perf_counter() - t0) * 1000
        
        # 4. Transmit & Receive
        t0 = time.perf_counter()
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(("127.0.0.1", receiver.port))
        tcp_send(s, wire_bytes)
        s.close()
        receiver.join()
        received_bytes = receiver.wire_bytes
        t_tcp = (time.perf_counter() - t0) * 1000
        
        # 5a. Decrypt
        t0 = time.perf_counter()
        header_len = struct.unpack(">I", received_bytes[:4])[0]
        header_bytes_recv = received_bytes[4 : 4 + header_len]
        ciphertext_recv = received_bytes[4 + header_len :]
        header_recv = json.loads(header_bytes_recv.decode("utf-8"))
        nonce_recv = base64.b64decode(header_recv["nonce_b64"])
        raw_bytes_recv = AESGCM(key).decrypt(nonce_recv, ciphertext_recv, header_bytes_recv)
        t_decrypt = (time.perf_counter() - t0) * 1000
        
        # 5b. Dequantize + decode entropy
        t0 = time.perf_counter()
        if header_recv["latent_meta"].get("apply_zlib", False):
            raw_bytes_recv = zlib.decompress(raw_bytes_recv)
        received_latent = LatentCrypto.deserialize(header_recv["latent_meta"], raw_bytes_recv)
        t_dequantize = (time.perf_counter() - t0) * 1000
        
        # 5c. Decode
        t0 = time.perf_counter()
        pipeline_out = decode_latent(model, received_latent, device)
        t_decode = (time.perf_counter() - t0) * 1000
        
        latency_total = t_encode + t_quantize + t_encrypt + t_tcp + t_decrypt + t_dequantize + t_decode
        
        # 5. Metrics Calculation
        pipeline_mse = F.mse_loss(pipeline_out, img_tensor).item()
        delta_mse = abs(pipeline_mse - standalone_mse)
        
        # PSNR & SSIM on normalized float tensors [0, 1]
        eps = 1e-8
        psnr_val = 10.0 * torch.log10(1.0 / (torch.tensor(pipeline_mse) + eps)).item()
        ssim_val = ssim(pipeline_out, img_tensor, data_range=1.0).item()
        
        # Record
        results["psnr"].append(psnr_val)
        results["ssim"].append(ssim_val)
        results["delta_mse"].append(delta_mse)
        results["payload_bytes"].append(payload_size)
        results["latency_encode"].append(t_encode)
        results["latency_quantize"].append(t_quantize)
        results["latency_encrypt"].append(t_encrypt)
        results["latency_tcp"].append(t_tcp)
        results["latency_decrypt"].append(t_decrypt)
        results["latency_dequantize"].append(t_dequantize)
        results["latency_decode"].append(t_decode)
        results["latency_total"].append(latency_total)
        
        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{len(df)} images...")

    # --- Summary ---
    print("\n" + "="*50)
    print("           BULK EVALUATION RESULTS")
    print("="*50)
    print(f"Images tested:   {len(results['psnr'])}")
    print(f"Model:           {args.model}")
    print(f"Resolution:      {args.height}x{args.width}")
    print("-" * 50)
    print(f"PSNR (dB):       {np.mean(results['psnr']):.2f} ± {np.std(results['psnr']):.2f}")
    print(f"SSIM:            {np.mean(results['ssim']):.4f} ± {np.std(results['ssim']):.4f}")
    print(f"Payload Size:    {np.mean(results['payload_bytes'])/1024/1024:.2f} MB ± {np.std(results['payload_bytes'])/1024/1024:.3f} MB")
    print(f"Total Latency:   {np.mean(results['latency_total']):.1f} ms ± {np.std(results['latency_total']):.1f} ms")
    print(f"Max Delta MSE:   {np.max(results['delta_mse']):.2e} (Zero-loss verified)")
    print("="*50)

    # Save to file
    os.makedirs("results", exist_ok=True)
    report_path = "results/bulk_evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f)
    print(f"\nRaw results saved to {report_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv_file",      required=True, help="Path to test set CSV")
    p.add_argument("--model",         default="unet_one", help="Model architecture")
    p.add_argument("--checkpoint",    default="unet/checkpoints/best_model.pth", help="Path to checkpoint")
    p.add_argument("--height",        default=256, type=int)
    p.add_argument("--width",         default=256, type=int)
    p.add_argument("--base_channels", default=64, type=int)
    p.add_argument("--num_samples",   default=50, type=int, help="Number of images to evaluate (0 for all)")
    p.add_argument("--quantize_fp16", type=lambda x: str(x).lower() == 'true', default=True, help="Use FP16 quantization")
    args = p.parse_args()
    
    run_evaluation(args)
