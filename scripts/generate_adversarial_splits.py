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

def main():
    print("Loading full dataset CSV...")
    csv_path = "dataset/mimic-cxr/mimic_cxr_aug_train.csv"
    if not os.path.exists(csv_path):
        csv_path = "mimic_cxr_aug_train.csv"
    
    df = pd.read_csv(csv_path)
    
    print("Filtering for images that actually exist on disk...")
    valid_records = []
    for idx, row in df.iterrows():
        img_path, view = extract_image(row)
        if img_path:
            full_path = f"dataset/mimic-cxr/official_data_iccv_final/{img_path}"
            if os.path.exists(full_path):
                dicom_id = os.path.basename(img_path).replace('.jpg', '')
                folder_path = os.path.dirname(img_path)
                valid_records.append({
                    'subject_id': row['subject_id'],
                    'kaggle_path': img_path,
                    'img_path': full_path,
                    'folder_path': folder_path,
                    'dicom_id': dicom_id,
                    'view_position': view
                })
                
    valid_df = pd.DataFrame(valid_records)
    print(f"Total valid images on disk: {len(valid_df)}")
    
    # Ensure one image per subject to guarantee strict subject isolation
    # and exactly map N subjects to N images.
    valid_df = valid_df.drop_duplicates(subset=['subject_id']).reset_index(drop=True)
    print(f"Total unique subjects available: {len(valid_df)}")
    
    if len(valid_df) < 5000:
        print("ERROR: Not enough unique subjects to create a 2500 train set and 2500 test set.")
        return

    # 1. Randomly sample 2500 subjects for the strict TEST set
    test_df = valid_df.sample(n=2500, random_state=42).reset_index(drop=True)
    
    # Remove the test subjects from the available pool
    remaining_df = valid_df[~valid_df['subject_id'].isin(test_df['subject_id'])].reset_index(drop=True)
    
    # 2. Randomly sample 2500 subjects for the maximum TRAIN set
    train_2500_df = remaining_df.sample(n=2500, random_state=42).reset_index(drop=True)
    
    # 3. Create nested train subsets
    # By taking the head of the shuffled 2500 set, the subsets are perfectly nested.
    # (e.g., the 100 images are a subset of the 250, which are a subset of the 500, etc.)
    train_1000_df = train_2500_df.head(1000)
    train_500_df = train_2500_df.head(500)
    train_250_df = train_2500_df.head(250)
    train_100_df = train_2500_df.head(100)
    
    # Save the splits
    os.makedirs("dataset/mimic-cxr", exist_ok=True)
    
    test_df.to_csv("dataset/mimic-cxr/adversary_test_2500.csv", index=False)
    train_2500_df.to_csv("dataset/mimic-cxr/adversary_train_2500.csv", index=False)
    train_1000_df.to_csv("dataset/mimic-cxr/adversary_train_1000.csv", index=False)
    train_500_df.to_csv("dataset/mimic-cxr/adversary_train_500.csv", index=False)
    train_250_df.to_csv("dataset/mimic-cxr/adversary_train_250.csv", index=False)
    train_100_df.to_csv("dataset/mimic-cxr/adversary_train_100.csv", index=False)
    
    print("\nSuccessfully generated strict adversarial subsets:")
    print(f"- Test (2500): dataset/mimic-cxr/adversary_test_2500.csv")
    print(f"- Train (2500): dataset/mimic-cxr/adversary_train_2500.csv")
    print(f"- Train (1000): dataset/mimic-cxr/adversary_train_1000.csv")
    print(f"- Train (500): dataset/mimic-cxr/adversary_train_500.csv")
    print(f"- Train (250): dataset/mimic-cxr/adversary_train_250.csv")
    print(f"- Train (100): dataset/mimic-cxr/adversary_train_100.csv")
    print("\nOverlap check: Total subjects shared between Train (2500) and Test (2500) =", 
          len(set(train_2500_df['subject_id']).intersection(set(test_df['subject_id']))))

if __name__ == "__main__":
    main()
