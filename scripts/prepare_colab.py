import os
import shutil
import pandas as pd
import zipfile

def prepare_colab_package():
    out_dir = "colab_package"
    
    # 1. Clean previous package if exists
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    
    print("Creating colab_package directory structure...")
    # 2. Copy dataset files
    os.makedirs(f"{out_dir}/dataset/mimic-cxr/images", exist_ok=True)
    
    for csv_name in ["train_subset.csv", "test_subset.csv"]:
        csv_path = f"dataset/mimic-cxr/{csv_name}"
        if not os.path.exists(csv_path):
            continue
            
        shutil.copy(csv_path, f"{out_dir}/dataset/mimic-cxr/{csv_name}")
        df = pd.read_csv(csv_path)
        
        count = 0
        for _, row in df.iterrows():
            img_path = row["img_path"]
            if os.path.exists(img_path):
                dest = os.path.join(out_dir, img_path)
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                shutil.copy(img_path, dest)
                count += 1
        print(f"Copied {count} images from {csv_name}")

    # 3. Copy scripts and src
    print("Copying scripts and src...")
    shutil.copytree("scripts", f"{out_dir}/scripts", ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copytree("src", f"{out_dir}/src", ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    
    # 4. Copy checkpoints (only needed models)
    print("Copying checkpoints...")
    for model in ["unet_noskip", "unet_one", "unet_res_xl"]:
        ckpt_path = f"checkpoints/{model}/checkpoints/best_model.pth"
        if os.path.exists(ckpt_path):
            dest = f"{out_dir}/{ckpt_path}"
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copy(ckpt_path, dest)
            
    # 5. Copy results and config
    print("Copying results and config files...")
    if os.path.exists("results"):
        shutil.copytree("results", f"{out_dir}/results", ignore=shutil.ignore_patterns('__pycache__'))
    
    for f in ["pyproject.toml", "uv.lock"]:
        if os.path.exists(f):
            shutil.copy(f, f"{out_dir}/{f}")

        
    # 6. Zip the package
    print("Zipping package...")
    zip_filename = "colab_package.zip"
    if os.path.exists(zip_filename):
        os.remove(zip_filename)
        
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(out_dir):
            for file in files:
                file_path = os.path.join(root, file)
                # Store it in the zip relative to the out_dir (so the zip extracts cleanly)
                arcname = os.path.relpath(file_path, out_dir)
                zipf.write(file_path, arcname)
                
    print(f"Success! Created {zip_filename}. You can now upload this to Google Drive.")

if __name__ == "__main__":
    prepare_colab_package()
