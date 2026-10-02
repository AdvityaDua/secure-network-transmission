import os
import glob
import pandas as pd
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from src.data.dataloader import MimicDataset

def main():
    images = glob.glob("dataset/mimic-cxr/images/*.jpg")[:5]
    if len(images) < 5:
        print("Not enough images downloaded to test.")
        return
        
    # create a dummy dataframe matching the expected format
    data = []
    for img in images:
        data.append({
            'img_path': img,
            'folder_path': os.path.dirname(img),
            'dicom_id': os.path.basename(img).replace('.jpg', ''),
            'view_position': 'PA'
        })
        
    df = pd.DataFrame(data)
    df.to_csv("dataset/mimic-cxr/sample_subset.csv", index=False)
    
    dataset = MimicDataset("dataset/mimic-cxr/sample_subset.csv")
    print(f"Dataset length: {len(dataset)}")
    
    loader = DataLoader(dataset, batch_size=5, shuffle=True, num_workers=0)
    
    try:
        batch = next(iter(loader))
        print(f"Batch shape: {batch.shape}")
        
        # Save a sample image grid to confirm they load and are valid
        os.makedirs("dataset/sample", exist_ok=True)
        vutils.save_image(batch, "dataset/sample/sample_images.png", nrow=5, normalize=True)
        print("Successfully loaded 5 images and saved to dataset/sample/sample_images.png")
    except Exception as e:
        print(f"Error loading images: {e}")

if __name__ == "__main__":
    main()
