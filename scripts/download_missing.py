"""Download only the missing images from Kaggle using the KGAT_ access token."""
import os
import sys
import time

def main():
    # Read the access token
    token_path = os.path.expanduser("~/.kaggle/access_token")
    with open(token_path) as f:
        token = f.read().strip()

    # Set env vars for Kaggle API — KGAT tokens use KAGGLE_KEY
    os.environ["KAGGLE_KEY"] = token
    os.environ["KAGGLE_USERNAME"] = "_"  # placeholder, KGAT tokens don't need it

    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()

    # Read only the missing image paths
    missing_file = "dataset/mimic-cxr/missing_images.txt"
    with open(missing_file) as f:
        paths = [line.strip() for line in f if line.strip()]

    target_dir = "dataset/mimic-cxr/images"
    os.makedirs(target_dir, exist_ok=True)

    print(f"Downloading {len(paths)} missing images to {target_dir}...")

    success = 0
    fail = 0

    for i, path in enumerate(paths):
        full_kaggle_name = path
        if not full_kaggle_name.startswith("official_data_iccv_final/"):
            full_kaggle_name = "official_data_iccv_final/" + path

        filename = os.path.basename(path)
        local_target = os.path.join(target_dir, filename)

        if os.path.exists(local_target):
            success += 1
            continue

        retries = 3
        while retries > 0:
            try:
                api.dataset_download_file(
                    "simhadrisadaram/mimic-cxr-dataset",
                    full_kaggle_name,
                    path=target_dir,
                )
                success += 1
                break
            except Exception as e:
                retries -= 1
                time.sleep(2)
                if retries == 0:
                    fail += 1
                    print(f"  FAIL: {filename}: {e}")

        time.sleep(0.5)

        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{len(paths)} (OK: {success}, Fail: {fail})")
            sys.stdout.flush()

    print(f"\nDone. Success: {success}, Failed: {fail}")

if __name__ == "__main__":
    main()
