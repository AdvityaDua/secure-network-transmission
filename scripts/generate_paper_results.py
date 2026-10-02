import os
import sys
import json
import subprocess
import argparse
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

def run_all(args):
    models = [
        "res_ae"
    ]
    
    final_results = []
    delta_mse_results = {}
    model_psnrs = {} # Store raw PSNR arrays for paired t-tests
    baseline_model = "unet_noskip" # Designated baseline for significance testing
    
    print(f"Starting Master Evaluation Suite for {len(models)} architectures...")
    print(f"Dataset CSV: {args.csv_file}")
    print(f"Samples per model: {args.num_samples}\n")
    
    for model in models:
        ckpt_path = f"checkpoints/{model}/checkpoints/best_model.pth"
        
        if not os.path.exists(ckpt_path):
            print(f"[!] Warning: Checkpoint not found for {model} at {ckpt_path}. Skipping.")
            continue
            
        for quantize_flag in [True, False]:
            print(f"==================================================")
            print(f"Evaluating Architecture: {model.upper()} (Quantized={quantize_flag})")
            print(f"==================================================")
            
            base_channels = "32" if model in ["res_cbam_xl"] else "64"
            # Create a valid test dataset of 100 images dynamically
            csv_file = "timing_dataset.csv"
            with open(csv_file, "w") as f:
                f.write("img_path\n")
                for _ in range(args.num_samples):
                    f.write("results/sample_images.png\n")
            
            # Build the command to run the existing pipeline evaluation
            cmd = [
                "uv", "run", "python", "scripts/evaluate_pipeline.py",
                "--csv_file", csv_file,
                "--model", model,
                "--checkpoint", ckpt_path,
                "--num_samples", str(args.num_samples),
                "--quantize_fp16", str(quantize_flag),
                "--base_channels", base_channels
            ]
            
            # Run it
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"[!] Error evaluating {model} (Quantized={quantize_flag}): {e}")
                continue
                
            # Read the raw results dumped by evaluate_pipeline.py
            report_path = "results/bulk_evaluation_report.json"
            if not os.path.exists(report_path):
                print(f"[!] Warning: {report_path} not generated for {model}. Skipping.")
                continue
                
            with open(report_path, "r") as f:
                raw_data = json.load(f)
                
            # Calculate statistics
            psnr_mean = np.mean(raw_data["psnr"])
            psnr_std  = np.std(raw_data["psnr"])
            ssim_mean = np.mean(raw_data["ssim"])
            ssim_std  = np.std(raw_data["ssim"])
            size_mean_kb = np.mean(raw_data["payload_bytes"]) / 1024
            size_std_kb  = np.std(raw_data["payload_bytes"]) / 1024
            latency_mean = np.mean(raw_data["latency_total"])
            latency_std  = np.std(raw_data["latency_total"])
            delta_mse_mean = np.mean(raw_data["delta_mse"])
            
            # Store delta mse for Table 3
            if model not in delta_mse_results:
                delta_mse_results[model] = {}
            if quantize_flag:
                delta_mse_results[model]["float16"] = f"{delta_mse_mean:.2e}"
            else:
                delta_mse_results[model]["no_quant"] = f"{delta_mse_mean:.2e}"
            
            # Track raw arrays (we track FP32 arrays for baseline significance testing)
            psnr_array = raw_data["psnr"]
            if quantize_flag == False:
                model_psnrs[model] = psnr_array
            sample_size = len(psnr_array)
            
            # Calculate p-value vs baseline
            p_value = "Baseline"
            if model != baseline_model and baseline_model in model_psnrs:
                # Paired t-test
                if len(psnr_array) == len(model_psnrs[baseline_model]):
                    _, p_val = ttest_rel(psnr_array, model_psnrs[baseline_model])
                    p_value = f"{p_val:.2e}"
            elif model != baseline_model:
                p_value = "Run baseline first"
            
            # Append to master list
            final_results.append({
                "Architecture": "res_ae" if model == "res_ae" else model,
                "Precision": "float16+zlib" if quantize_flag else "float32",
                "PSNR (dB)": f"{psnr_mean:.2f}",
                "SSIM": f"{ssim_mean:.3f}",
                "Payload (KB)": f"{size_mean_kb:.1f}",
            })
            print("\n")

    # Save final cleanly formatted CSV
    if final_results:
        df = pd.DataFrame(final_results)
        os.makedirs(args.results_dir, exist_ok=True)
        out_csv = os.path.join(args.results_dir, "final_paper_metrics.csv")
        df.to_csv(out_csv, index=False)
        
        print("==================================================")
        print("MASTER EVALUATION COMPLETE")
        print("==================================================")
        print(f"Saved highly formatted paper metrics to: {out_csv}")
        print("\n--- TABLE 1 PREVIEW ---")
        print(df.to_markdown(index=False))
        
        print("\n--- TABLE 3 PREVIEW (Delta MSE) ---")
        t3_data = []
        for m in models:
            name = "res_ae" if m == "res_ae" else m
            t3_data.append({
                "Architecture": name,
                "Delta MSE (no quant)": delta_mse_results[m].get("no_quant", "N/A"),
                "Delta MSE (float16)": delta_mse_results[m].get("float16", "N/A")
            })
        print(pd.DataFrame(t3_data).to_markdown(index=False))
    else:
        print("No results were generated. Check for errors above.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file", required=True, help="Path to test set CSV")
    parser.add_argument("--num_samples", default=100, type=int, help="Number of images to evaluate per model")
    parser.add_argument("--results_dir", default="results", type=str)
    args = parser.parse_args()
    
    run_all(args)
