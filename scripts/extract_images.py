import os
import zipfile
import shutil

def main():
    target_dir = "dataset/mimic-cxr/images"
    os.makedirs(target_dir, exist_ok=True)
    
    zip_path = os.path.expanduser("~/Downloads/archive.zip")
    
    if not os.path.exists(zip_path):
        print(f"Error: Zip archive not found at {zip_path}")
        print("Please place the manually downloaded zip file there and run this script again.")
        return
        
    print("Reading paths to extract...")
    if not os.path.exists("dataset/mimic-cxr/images_to_download.txt"):
        print("Error: images_to_download.txt not found. Did you run prepare_subset.py?")
        return
        
    with open("dataset/mimic-cxr/images_to_download.txt", "r") as f:
        paths = [line.strip() for line in f if line.strip()]
        
    print(f"Extracting {len(paths)} files from zip...")
    success_count = 0
    with zipfile.ZipFile(zip_path, 'r') as z:
        zip_files = z.namelist()
        zip_files_set = set(zip_files)
        
        for i, path in enumerate(paths):
            # Try to find the correct path in the zip
            target_in_zip = path
            if path not in zip_files_set and 'official_data_iccv_final/' + path in zip_files_set:
                target_in_zip = 'official_data_iccv_final/' + path
            elif path.startswith('official_data_iccv_final/') and path[25:] in zip_files_set:
                target_in_zip = path[25:]
                
            if target_in_zip in zip_files_set:
                filename = os.path.basename(target_in_zip)
                local_dest = os.path.join(target_dir, filename)
                
                if not os.path.exists(local_dest):
                    with z.open(target_in_zip) as source, open(local_dest, "wb") as target:
                        shutil.copyfileobj(source, target)
                success_count += 1
            
            if (i + 1) % 1000 == 0:
                print(f"Processed {i + 1}/{len(paths)}...")
                
    print(f"Extracted {success_count}/{len(paths)} files successfully.")
    
    # Optional cleanup prompt
    print("\nExtraction complete!")
    print(f"You can now safely delete {zip_path} to free up 18GB of disk space.")

if __name__ == "__main__":
    main()
