import os
import time
from kaggle.api.kaggle_api_extended import KaggleApi
import sys

def main():
    api = KaggleApi()
    api.authenticate()
    
    with open("dataset/mimic-cxr/images_to_download.txt", "r") as f:
        paths = [line.strip() for line in f if line.strip()]
        
    target_dir = "dataset/mimic-cxr/images"
    os.makedirs(target_dir, exist_ok=True)
    
    print(f"Starting sequential download of {len(paths)} images to {target_dir}...")
    
    success_count = 0
    fail_count = 0
    
    for i, path in enumerate(paths):
        full_kaggle_name = path
        if not full_kaggle_name.startswith('official_data_iccv_final/'):
            full_kaggle_name = 'official_data_iccv_final/' + path
            
        filename = os.path.basename(path)
        local_target = os.path.join(target_dir, filename)
        
        if os.path.exists(local_target):
            success_count += 1
            continue
            
        retries = 3
        while retries > 0:
            try:
                api.dataset_download_file("simhadrisadaram/mimic-cxr-dataset", full_kaggle_name, path=target_dir)
                success_count += 1
                break
            except Exception as e:
                # Likely 429 Too Many Requests
                retries -= 1
                time.sleep(2) # Backoff
                if retries == 0:
                    fail_count += 1
                    print(f"Failed to download {filename}: {e}")
                    
        # Sleep to avoid rate limiting
        time.sleep(0.5)
        
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(paths)} (Success: {success_count}, Failed: {fail_count})")
            sys.stdout.flush()

    print(f"Finished downloading. Success: {success_count}, Failed: {fail_count}.")

if __name__ == "__main__":
    main()
