import os
import sys
import torch
import torchvision.utils as vutils
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data.dataloader import MimicDataset

def sanity_check():
    dataset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset", "mimic-cxr-subset"))
    train_csv = os.path.join(dataset_dir, "train.csv")
    
    if not os.path.exists(train_csv):
        print(f"❌ Error: {train_csv} not found. Run download_mimic_subset.py first.")
        return
        
    print(f"Loading dataset from: {train_csv}")
    
    try:
        dataset = MimicDataset(train_csv, h=256, w=256)
        print(f"✅ Successfully initialized dataset with {len(dataset)} images.")
    except Exception as e:
        print(f"❌ Failed to initialize dataset: {e}")
        return
        
    dataloader = DataLoader(dataset, batch_size=5, shuffle=True)
    batch = next(iter(dataloader))
    
    print(f"✅ Loaded 1 batch.")
    print(f"Batch shape: {batch.shape} (Expected: 5, 1, 256, 256)")
    print(f"Data type:   {batch.dtype}")
    print(f"Min value:   {batch.min().item():.3f} (Expected: ~0.0)")
    print(f"Max value:   {batch.max().item():.3f} (Expected: ~1.0)")
    
    # Save a quick grid
    sample_path = os.path.join(dataset_dir, "sanity_check_grid.png")
    vutils.save_image(batch, sample_path, nrow=5, normalize=True)
    print(f"✅ Saved a 5-image grid to {sample_path}. Check it visually!")

if __name__ == "__main__":
    sanity_check()
