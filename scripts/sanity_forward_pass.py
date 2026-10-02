import torch
from src.data.dataloader import MimicDataset
from torch.utils.data import DataLoader
from src.network.client import load_model
import glob

def main():
    print("Loading test dataset (5 samples)...")
    dataset = MimicDataset("dataset/mimic-cxr/test_subset.csv")
    loader = DataLoader(dataset, batch_size=5, shuffle=False)
    batch = next(iter(loader))
    print(f"Batch shape: {batch.shape}")
    
    models_to_test = ["unet_noskip", "res_cbam_xl", "unet_one", "res_se_xl_skips"]
    
    device = torch.device("cpu")
    
    for model_name in models_to_test:
        print(f"\nTesting {model_name}...")
        try:
            # Look for checkpoint
            ckpt_pattern = f"checkpoints/{model_name}/checkpoints/best_model.pth"
            ckpts = glob.glob(ckpt_pattern)
            if not ckpts:
                print(f"  No checkpoint found for {model_name}!")
                continue
            
            ckpt_path = ckpts[0]
            print(f"  Loading {ckpt_path}...")
            
            # Using base_channels=64 (default in pipeline_test.py)
            model = load_model(model_name, 64, ckpt_path, device)
            model.eval()
            
            with torch.no_grad():
                latent = model.encode(batch)
                print(f"  Encoded latent shape: {latent.shape}")
                recon = model.decode(latent)
                print(f"  Decoded recon shape: {recon.shape}")
                
            print(f"  {model_name} sanity check PASSED.")
        except Exception as e:
            print(f"  {model_name} sanity check FAILED: {e}")

if __name__ == "__main__":
    main()
