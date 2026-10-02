import pandas as pd
import os

def clean_csv(csv_path):
    if not os.path.exists(csv_path): return
    df = pd.read_csv(csv_path)
    # Check if img_path exists locally
    exists_mask = df['img_path'].apply(os.path.exists)
    cleaned_df = df[exists_mask]
    cleaned_df.to_csv(csv_path, index=False)
    print(f"{os.path.basename(csv_path)}: kept {len(cleaned_df)}/{len(df)} images.")

def main():
    clean_csv("dataset/mimic-cxr/val_subset.csv")
    clean_csv("dataset/mimic-cxr/train_subset.csv")
    clean_csv("dataset/mimic-cxr/test_subset.csv")
    
    # Also delete the 17.7GB zip file
    zip_path = os.path.expanduser("~/Downloads/archive.zip")
    if os.path.exists(zip_path):
        os.remove(zip_path)
        print(f"Deleted {zip_path} successfully!")

if __name__ == "__main__":
    main()
