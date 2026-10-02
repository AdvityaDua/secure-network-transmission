import os
import json
import subprocess
import numpy as np

def main():
    models = ["res_ae"]
    
    # Generate a temporary CSV with 50 valid images for timing
    csv_file = "timing_dataset.csv"
    with open(csv_file, "w") as f:
        f.write("img_path\n")
        for _ in range(50):
            f.write("results/sample_images.png\n")
            
    num_samples = 50
    
    print("\n--- TABLE 5 PREVIEW ---")
    print(f"| Architecture | Encode | Quant.+Entropy | Encrypt | Transmit | Decrypt | Decode |")
    print(f"|:-------------|-------:|---------------:|--------:|---------:|--------:|-------:|")
    
    for model in models:
        ckpt = f"checkpoints/{model}/checkpoints/best_model.pth"
        base_ch = "32" if model in ["unet_res_xl", "res_cbam_xl"] else "64"
        
        # Run evaluation just for timing, with FP16 quantized enabled
        cmd = [
            "uv", "run", "python", "scripts/evaluate_pipeline.py",
            "--csv_file", csv_file,
            "--model", model,
            "--checkpoint", ckpt,
            "--num_samples", str(num_samples),
            "--quantize_fp16", "True",
            "--base_channels", base_ch
        ]
        
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        with open("results/bulk_evaluation_report.json", "r") as f:
            data = json.load(f)
            
        t_enc = np.mean(data["latency_encode"])
        t_quant = np.mean(data["latency_quantize"])
        t_crypt = np.mean(data["latency_encrypt"])
        t_tx = np.mean(data["latency_tcp"])
        t_decrypt = np.mean(data["latency_decrypt"])
        
        # In evaluate_pipeline, decoding latency is split into dequantize + decode
        t_dec = np.mean(data["latency_dequantize"]) + np.mean(data["latency_decode"])
        
        display_name = "res_se_xl_skips" if model == "unet_res_xl" else model
        print(f"| {display_name:<15} | {t_enc:>6.1f} | {t_quant:>14.1f} | {t_crypt:>7.1f} | {t_tx:>8.1f} | {t_decrypt:>7.1f} | {t_dec:>6.1f} |")

if __name__ == "__main__":
    main()
