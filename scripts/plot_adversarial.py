import os
import argparse
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def generate_table_4(df, arch_name):
    # Thresholds
    thresholds = [20.0, 25.0, 30.0]
    
    # Dataset sizes
    dataset_sizes = df["dataset_size"].unique()
    
    results = []
    
    for d_size in dataset_sizes:
        subset = df[df["dataset_size"] == d_size]
        row_data = {"Dataset size": d_size}
        
        for thresh in thresholds:
            # Find the first epoch where test_psnr >= thresh
            crossed = subset[subset["test_psnr"] >= thresh]
            if len(crossed) > 0:
                first = crossed.iloc[0]
                imgs = int(first["cumulative_images"])
                mins = first["cumulative_train_time_sec"] / 60.0
                row_data[f"{int(thresh)} dB"] = f"{imgs} imgs / {mins:.1f} min"
            else:
                row_data[f"{int(thresh)} dB"] = "Did not reach"
                
        results.append(row_data)
        
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="./results")
    args = parser.parse_args()
    
    # We will look for all four models
    models = ["unet_noskip", "unet_one", "res_ae", "res_cbam_xl"]
    
    all_data = {}
    for model in models:
        csv_path = os.path.join(args.results_dir, f"adversarial_compute_cost_{model}.csv")
        if os.path.exists(csv_path):
            all_data[model] = pd.read_csv(csv_path)
            
    if not all_data:
        print("No adversarial sweep data found. Run adversarial_attack.py first.")
        return
        
    print("\n==================================================")
    print("               TABLE 4 PREVIEW")
    print("==================================================")
    
    for model, df in all_data.items():
        print(f"\nArchitecture: {model}")
        print("-" * 60)
        table = generate_table_4(df, model)
        
        # Print markdown table
        print(f"| Dataset size | 20 dB | 25 dB | 30 dB |")
        print(f"|:-------------|:------|:------|:------|")
        for row in table:
            print(f"| {row['Dataset size']:<12} | {row.get('20 dB', '-'):<18} | {row.get('25 dB', '-'):<18} | {row.get('30 dB', '-'):<18} |")
    
    # Plot Figure 2
    plt.figure(figsize=(12, 8))
    
    colors = {
        "unet_noskip": "blue", 
        "unet_one": "orange",
        "res_ae": "green",
        "res_cbam_xl": "red"
    }
    linestyles = {500: "--", 2500: "-", 3517: "-", 5000: "-"}
    
    for model, df in all_data.items():
        for d_size in df["dataset_size"].unique():
            subset = df[df["dataset_size"] == d_size]
            
            # Use display name for the paper
            display_name = "res_ae" if model == "res_ae" else model
            
            plt.plot(
                subset["cumulative_train_time_sec"] / 60.0, 
                subset["test_psnr"], 
                label=f"{display_name} ({d_size} imgs)",
                color=colors.get(model, "black"),
                linestyle=linestyles.get(int(d_size), "-"),
                linewidth=2
            )
            
    # Draw reference thresholds
    plt.axhline(y=20.0, color='r', linestyle=':', alpha=0.5, label='20 dB threshold')
    plt.axhline(y=25.0, color='g', linestyle=':', alpha=0.5, label='25 dB threshold')
    plt.axhline(y=30.0, color='b', linestyle=':', alpha=0.5, label='30 dB threshold')
    
    plt.xlabel("Cumulative Training Time (minutes)", fontsize=12)
    plt.ylabel("Surrogate Decoder PSNR (dB)", fontsize=12)
    plt.title("Adversarial Reconstruction Fidelity vs. Computation Time", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot_path = os.path.join(args.results_dir, "adversarial_figure_2.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved Figure 2 plot to: {plot_path}")

if __name__ == "__main__":
    main()
