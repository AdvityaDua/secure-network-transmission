import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    results_dir = "./results"
    
    csv_files = [
        "compute_cost_res_ae.csv",
        "compute_cost_res_cbam_xl.csv",
        "compute_cost_unet_noskip.csv",
        "compute_cost_unet_one.csv"
    ]
    
    titles = [
        "Res AE (res_ae)",
        "Res CBAM XL (res_cbam_xl)",
        "U-Net No Skip (unet_noskip)",
        "U-Net One (unet_one)"
    ]
    
    for i, (csv_name, title) in enumerate(zip(csv_files, titles)):
        csv_path = os.path.join(results_dir, csv_name)
        
        if not os.path.exists(csv_path):
            print(f"File not found: {csv_path}")
            continue
            
        try:
            df = pd.read_csv(csv_path)
            if 'dataset_size' not in df.columns:
                df = pd.read_csv(csv_path, header=None, names=["dataset_size", "epoch", "cumulative_images", "cumulative_train_time_sec", "cumulative_eval_time_sec", "test_psnr", "test_ssim", "stop_reason"])
        except Exception as e:
            print(f"Error reading {csv_path}: {e}")
            continue
        
        # Slightly taller than before so curves aren't squashed flat,
        # and we leave room on the right for an external legend instead
        # of shrinking it into a corner where it overlaps the lines
        fig, ax = plt.subplots(figsize=(9, 4.2))
        
        dataset_sizes = df['dataset_size'].unique()
        
        for d_size in sorted(dataset_sizes):
            subset = df[df['dataset_size'] == d_size]
            ax.plot(
                subset['epoch'], 
                subset['test_psnr'], 
                label=f"{d_size} imgs", 
                linewidth=2
            )
            
        ax.set_title(title, fontsize=13, fontweight='bold', pad=14)
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Test PSNR (dB)", fontsize=11)
        ax.tick_params(labelsize=9)
        ax.grid(True, linestyle='--', alpha=0.7)
        # Legend placed outside the axes, in its own column, so it
        # never sits on top of the curves regardless of plot height
        ax.legend(
            title="Dataset Size", fontsize=9, title_fontsize=9,
            loc='center left', bbox_to_anchor=(1.01, 0.5), framealpha=0.9
        )
        
        plt.tight_layout(pad=0.5)
        # Push the axes down a touch so the bold title has breathing
        # room above it and isn't flush against the top edge of the image
        fig.subplots_adjust(top=0.88)
        output_name = csv_name.replace('compute_cost_', '').replace('.csv', '.png')
        output_path = os.path.join(results_dir, output_name)
        plt.savefig(output_path, dpi=200, bbox_inches='tight', pad_inches=0.08)
        plt.close()
        
        print(f"Graph saved to {output_path}")

if __name__ == "__main__":
    main()