import os
import argparse
import numpy as np
import pandas as pd
import torch
from PIL import Image
import io
import time
from piq import ssim as compute_ssim, psnr as compute_psnr

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data.dataloader import MimicDataset
from src.network.crypto import LatentCrypto

def run_jpeg_baseline(args):
    dataset = MimicDataset(args.csv_file, h=256, w=256)
    key = LatentCrypto.generate_key("temp_jpeg_key.bin")
    
    qualities = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]
    results = []

    print(f"Running JPEG baseline on {len(dataset)} images...")
    
    for q in qualities:
        total_psnr = 0
        total_ssim = 0
        total_size = 0
        
        for i in range(len(dataset)):
            img_tensor = dataset[i]
            # Convert [-1, 1] tensor to [0, 255] PIL Image
            img_np = ((img_tensor.squeeze().numpy() + 1.0) / 2.0 * 255).clip(0, 255).astype(np.uint8)
            img_pil = Image.fromarray(img_np, mode='L')
            
            # Compress to JPEG in memory
            buffer = io.BytesIO()
            img_pil.save(buffer, format="JPEG", quality=q)
            jpeg_bytes = buffer.getvalue()
            
            # Encrypt the JPEG bytes using AES-GCM (standard file encryption baseline)
            nonce, ciphertext = LatentCrypto.encrypt(key, jpeg_bytes, aad=b"jpeg_baseline")
            encrypted_size = len(nonce) + len(ciphertext)
            total_size += encrypted_size
            
            # Decrypt and decode
            decrypted_bytes = LatentCrypto.decrypt(key, nonce, ciphertext, aad=b"jpeg_baseline")
            reconstructed_pil = Image.open(io.BytesIO(decrypted_bytes))
            reconstructed_np = np.array(reconstructed_pil).astype(np.float32)
            
            # Convert back to [-1, 1] tensor for fair metric comparison
            recon_tensor = torch.from_numpy(reconstructed_np / 255.0 * 2.0 - 1.0).unsqueeze(0).unsqueeze(0)
            orig_tensor = img_tensor.unsqueeze(0)
            
            # Clamp to [0, 1] range for piq metrics as expected
            orig_01 = (orig_tensor + 1.0) / 2.0
            recon_01 = (recon_tensor + 1.0) / 2.0
            
            total_psnr += compute_psnr(recon_01, orig_01, data_range=1.0).item()
            total_ssim += compute_ssim(recon_01, orig_01, data_range=1.0).item()
            
            if i >= args.num_samples - 1:
                break
                
        num_evaluated = min(args.num_samples, len(dataset))
        avg_psnr = total_psnr / num_evaluated
        avg_ssim = total_ssim / num_evaluated
        avg_size = total_size / num_evaluated
        
        results.append({
            "jpeg_quality": q,
            "avg_payload_bytes": avg_size,
            "avg_psnr": avg_psnr,
            "avg_ssim": avg_ssim
        })
        print(f"Quality {q:3d} | Size: {avg_size/1024:7.2f} KB | PSNR: {avg_psnr:5.2f} | SSIM: {avg_ssim:5.4f}")
        
    df = pd.DataFrame(results)
    os.makedirs(args.results_dir, exist_ok=True)
    out_path = os.path.join(args.results_dir, "jpeg_baseline_metrics.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved baseline results to {out_path}")
    
    if os.path.exists("temp_jpeg_key.bin"):
        os.remove("temp_jpeg_key.bin")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file", type=str, required=True, help="Path to test set CSV")
    parser.add_argument("--results_dir", type=str, default="./results")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of test images to evaluate")
    args = parser.parse_args()
    
    run_jpeg_baseline(args)
