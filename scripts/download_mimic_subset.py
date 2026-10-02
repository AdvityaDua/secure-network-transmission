import os
import sys
import requests
import pandas as pd
from getpass import getpass
import gzip
import io

def download_subset():
    print("=== MIMIC-CXR-JPG Subset Downloader ===")
    print("This will download a small subset (6,000 images) using the official split.")
    
    username = input("PhysioNet Username: ")
    password = getpass("PhysioNet Password: ")
    
    base_url = "https://physionet.org/files/mimic-cxr-jpg/2.0.0"
    auth = (username, password)
    
    # 1. Download Split and Metadata files
    print("\n[1/4] Downloading split and metadata files...")
    
    def fetch_csv_gz(filename):
        url = f"{base_url}/{filename}"
        r = requests.get(url, auth=auth)
        if r.status_code != 200:
            print(f"Failed to authenticate or download {filename}. Status: {r.status_code}")
            sys.exit(1)
        with gzip.open(io.BytesIO(r.content), 'rt') as f:
            return pd.read_csv(f)

    splits = fetch_csv_gz("mimic-cxr-2.0.0-split.csv.gz")
    metadata = fetch_csv_gz("mimic-cxr-2.0.0-metadata.csv.gz")
    
    # 2. Merge and Filter
    print("[2/4] Filtering for frontal views (AP/PA)...")
    df = pd.merge(splits, metadata, on=["dicom_id", "subject_id", "study_id"])
    frontal_df = df[df["ViewPosition"].isin(["AP", "PA"])]
    
    # 3. Sample Subsets
    train_pool = frontal_df[frontal_df["split"] == "train"]
    val_pool = frontal_df[frontal_df["split"] == "validate"]
    test_pool = frontal_df[frontal_df["split"] == "test"]
    
    # Take a small subset due to 4.2 GiB disk space constraint
    train_subset = train_pool.sample(n=5000, random_state=42)
    val_subset = val_pool.sample(n=500, random_state=42)
    test_subset = test_pool.sample(n=500, random_state=42)
    
    # 4. Verify No Overlap
    print("[3/4] Verifying no overlaps between splits...")
    train_ids = set(train_subset["dicom_id"])
    val_ids = set(val_subset["dicom_id"])
    test_ids = set(test_subset["dicom_id"])
    
    assert len(train_ids.intersection(val_ids)) == 0, "Overlap between train and val!"
    assert len(train_ids.intersection(test_ids)) == 0, "Overlap between train and test!"
    assert len(val_ids.intersection(test_ids)) == 0, "Overlap between val and test!"
    print("✅ Verified: Zero overlap across train, val, and test subsets.")
    
    # 5. Format CSVs for Dataloader
    dataset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset", "mimic-cxr-subset"))
    os.makedirs(dataset_dir, exist_ok=True)
    
    def prepare_and_save_csv(subset_df, split_name):
        # Format expected by dataloader: img_path, folder_path, dicom_id, view_position
        formatted = []
        for _, row in subset_df.iterrows():
            subject = f"p{str(row['subject_id'])[:2]}"
            rel_path = f"files/{subject}/p{row['subject_id']}/s{row['study_id']}/{row['dicom_id']}.jpg"
            img_path = os.path.join(dataset_dir, rel_path)
            folder_path = os.path.dirname(img_path)
            
            formatted.append({
                "img_path": img_path,
                "folder_path": folder_path,
                "dicom_id": row["dicom_id"],
                "view_position": row["ViewPosition"],
                "remote_url": f"{base_url}/{rel_path}"
            })
            
        out_df = pd.DataFrame(formatted)
        csv_path = os.path.join(dataset_dir, f"{split_name}.csv")
        out_df[["img_path", "folder_path", "dicom_id", "view_position"]].to_csv(csv_path, index=False)
        return out_df

    train_out = prepare_and_save_csv(train_subset, "train")
    val_out = prepare_and_save_csv(val_subset, "val")
    test_out = prepare_and_save_csv(test_subset, "test")
    
    print(f"✅ Saved formatted CSVs to {dataset_dir}")
    
    # 6. Download Images
    print("\n[4/4] Downloading image files (~6,000 images). This will take a while...")
    all_files = pd.concat([train_out, val_out, test_out])
    
    for idx, row in all_files.iterrows():
        if os.path.exists(row["img_path"]):
            continue
            
        os.makedirs(row["folder_path"], exist_ok=True)
        r = requests.get(row["remote_url"], auth=auth)
        if r.status_code == 200:
            with open(row["img_path"], "wb") as f:
                f.write(r.content)
        else:
            print(f"Failed to download {row['dicom_id']}. Status: {r.status_code}")
            
        if (idx + 1) % 100 == 0:
            print(f"Downloaded {idx + 1} / {len(all_files)} images...")
            
    print("\nDownload complete! You can now run the sanity check.")

if __name__ == "__main__":
    download_subset()
