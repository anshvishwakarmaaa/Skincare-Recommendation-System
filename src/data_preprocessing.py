import os
import pandas as pd
import numpy as np
from PIL import Image
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

# Skin conditions matched from the dataset
CONDITIONS = ['acne', 'blackheads', 'dark spots', 'dryness', 'normal', 'oily', 'pores', 'wrinkles']

def validate_dataset(csv_path="data/skin_dataset.csv", output_path="data/skin_dataset_cleaned.csv"):
    """
    Checks every image path actually exists on disk and can be opened by PIL without corruption.
    Removes bad rows, prints a summary of removed images, and saves the cleaned CSV.
    """
    print(f"🔍 Validating dataset from {csv_path}...")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset CSV not found at {csv_path}")
        
    df = pd.read_csv(csv_path)
    original_len = len(df)
    
    valid_indices = []
    bad_rows_count = 0
    
    for idx, row in df.iterrows():
        img_path = row['image_path']
        
        # Check if path exists
        if not os.path.exists(img_path):
            bad_rows_count += 1
            print(f"❌ Missing file: {img_path}")
            continue
            
        # Check if PIL can open and verify it
        try:
            with Image.open(img_path) as img:
                img.verify()  # Verify that it is an image
            valid_indices.append(idx)
        except Exception as e:
            bad_rows_count += 1
            print(f"❌ Corrupt file ({e}): {img_path}")
            
    cleaned_df = df.loc[valid_indices].reset_index(drop=True)
    
    print(f"\n📊 Validation Summary:")
    print(f"   Original rows: {original_len}")
    print(f"   Removed rows : {bad_rows_count}")
    print(f"   Cleaned rows : {len(cleaned_df)}")
    
    # Save the cleaned CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cleaned_df.to_csv(output_path, index=False)
    print(f"💾 Cleaned CSV saved to {output_path}")
    
    return cleaned_df

def print_dataset_stats(df):
    """
    Prints a rich summary table showing per-class count, percentage, 
    and min/max/mean image file size in KB.
    """
    print("\n" + "═" * 70)
    print("📈 DATASET STATISTICS SUMMARY")
    print("═" * 70)
    
    total_samples = len(df)
    
    # Calculate file sizes in KB
    file_sizes_kb = []
    for path in df['image_path']:
        if os.path.exists(path):
            file_sizes_kb.append(os.path.getsize(path) / 1024.0)
            
    if file_sizes_kb:
        min_size = min(file_sizes_kb)
        max_size = max(file_sizes_kb)
        mean_size = sum(file_sizes_kb) / len(file_sizes_kb)
        print(f"Total valid images analyzed for size: {len(file_sizes_kb)}")
        print(f"File Size (KB) -> Min: {min_size:.1f} | Max: {max_size:.1f} | Mean: {mean_size:.1f}")
    
    print("-" * 70)
    print(f"{'Condition':<15} | {'Count':<8} | {'Percentage'}")
    print("-" * 70)
    for condition in CONDITIONS:
        count = df[condition].sum()
        pct = (count / total_samples) * 100 if total_samples > 0 else 0
        print(f"{condition:<15} | {count:<8} | {pct:.1f}%")
    print("═" * 70 + "\n")

def create_balanced_dataset(df, samples_per_class=200):
    """
    Stratified random sampling for subset balancing up to `samples_per_class`.
    """
    balanced_dfs = []
    for condition in CONDITIONS:
        if condition in df.columns:
            cond_df = df[df[condition] == 1]
            count = len(cond_df)
            n_samples = min(samples_per_class, count)
            if n_samples > 0:
                sampled_df = cond_df.sample(n=n_samples, random_state=42)
                balanced_dfs.append(sampled_df)
                
    if not balanced_dfs:
        return df
        
    balanced_df = pd.concat(balanced_dfs, ignore_index=True)
    
    # Drop duplicates since images can have multiple labels
    balanced_df = balanced_df.drop_duplicates(subset=['image_path']).reset_index(drop=True)
    print(f"⚖️ Balanced dataset size (after dropping overlap): {len(balanced_df)} images")
    
    return balanced_df

def split_dataset(df):
    """
    Stratified multi-label split using iterative-stratification library.
    Generates Train, Validation, and Test groupings.
    """
    X = np.arange(len(df))  # Dummy array of indices
    y = df[CONDITIONS].values.astype(int)  # Ensure targets are typed appropriately
    
    # First split: Separate Test Data (15%)
    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    for train_idx, test_idx in msss.split(X, y):
        X_train_val, X_test = X[train_idx], X[test_idx]
        y_train_val, y_test = y[train_idx], y[test_idx]

    # Second split: Train vs Validation (15% from the intermediate training set)
    msss_val = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    for train_idx, val_idx in msss_val.split(X_train_val, y_train_val):
        X_train, X_val = X_train_val[train_idx], X_train_val[val_idx]
        # y_train, y_val = y_train_val[train_idx], y_train_val[val_idx]

    # Map indices back to original DataFrame
    train_df = df.iloc[X_train].reset_index(drop=True)
    val_df = df.iloc[X_val].reset_index(drop=True)
    test_df = df.iloc[X_test].reset_index(drop=True)

    print("📊 Stratified Split Results:")
    print(f"   Train: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"   Val  : {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
    print(f"   Test : {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
    
    return train_df, val_df, test_df

if __name__ == "__main__":
    print("🚀 Standard Setup Example Run")
    cleaned = validate_dataset("data/skin_dataset.csv", "data/skin_dataset_cleaned.csv")
    print_dataset_stats(cleaned)
    balanced = create_balanced_dataset(cleaned, samples_per_class=200)
    train_df, val_df, test_df = split_dataset(balanced)
