import pandas as pd
import ast
import os
import random

random.seed(42)

def extract_image(row):
    # Prefer PA over AP
    pa_list = []
    ap_list = []
    
    if pd.notna(row['PA']):
        try:
            pa_list = ast.literal_eval(row['PA'])
        except:
            pass
            
    if pd.notna(row['AP']):
        try:
            ap_list = ast.literal_eval(row['AP'])
        except:
            pass
            
    if len(pa_list) > 0:
        return pa_list[0], 'PA'
    elif len(ap_list) > 0:
        return ap_list[0], 'AP'
    return None, None

def process_df(df, subset_size=None):
    processed = []
    
    # Process rows
    for idx, row in df.iterrows():
        img_path, view = extract_image(row)
        if img_path:
            dicom_id = os.path.basename(img_path).replace('.jpg', '')
            folder_path = os.path.dirname(img_path)
            full_img_path = f"dataset/mimic-cxr/official_data_iccv_final/{img_path}"
            
            # CRITICAL FIX: The 17GB Kaggle dataset is a small subset of the full MIMIC-CXR.
            # Many images in the CSV will not exist in the zip. We MUST check if they exist!
            if os.path.exists(full_img_path):
                processed.append({
                    'subject_id': row['subject_id'],
                    'kaggle_path': img_path, 
                    'img_path': full_img_path,
                    'folder_path': folder_path,
                    'dicom_id': dicom_id,
                    'view_position': view
                })
            
    res_df = pd.DataFrame(processed)
    # Ensure one image per subject
    res_df = res_df.drop_duplicates(subset=['subject_id'])
    
    # Shuffle
    res_df = res_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    if subset_size and len(res_df) > subset_size:
        res_df = res_df.head(subset_size)
        
    return res_df

def main():
    print("Loading Validate CSV...")
    val_df = pd.read_csv("dataset/mimic-cxr/mimic_cxr_aug_validate.csv")
    val_processed = process_df(val_df)
    print(f"Total usable validation images: {len(val_processed)}")

    # FIX: Source the adversarial TEST set from the validation split, NOT
    # from the training split. The victim models were trained on
    # mimic_cxr_aug_train.csv, so using images from the validation split
    # ensures the victim has never seen the test images. This prevents
    # data leakage that inflates adversarial PSNR/SSIM.
    #
    # Split validation pool: first half → test_subset, second half → val_subset
    midpoint = len(val_processed) // 2
    test_processed = val_processed.iloc[:midpoint].reset_index(drop=True)
    val_subset = val_processed.iloc[midpoint:].reset_index(drop=True)

    print(f"Test subset size (from validate split): {len(test_processed)}")
    print(f"Val subset size (from validate split): {len(val_subset)}")

    test_processed.to_csv("dataset/mimic-cxr/test_subset.csv", index=False)
    val_subset.to_csv("dataset/mimic-cxr/val_subset.csv", index=False)
    
    print("Loading Train CSV...")
    train_df = pd.read_csv("dataset/mimic-cxr/mimic_cxr_aug_train.csv")
    train_processed = process_df(train_df, subset_size=5000)
    
    print(f"Train subset size: {len(train_processed)}")
    
    train_processed.to_csv("dataset/mimic-cxr/train_subset.csv", index=False)
    
    # Combine all paths to be downloaded (using the original kaggle path)
    all_paths = pd.concat([val_subset['kaggle_path'], test_processed['kaggle_path'], train_processed['kaggle_path']])
    all_paths.to_csv("dataset/mimic-cxr/images_to_download.txt", index=False, header=False)
    print(f"Total images to download: {len(all_paths)}")


if __name__ == "__main__":
    main()
