import argparse
import time
import os
import torch
from tqdm import tqdm

from src.data_preprocessing import validate_dataset, create_balanced_dataset, split_dataset
from src.pytorch_train import train_fixed_model, find_optimal_thresholds, evaluate_on_test
from config import CONDITIONS

def main():
    parser = argparse.ArgumentParser(description="AI Skincare Model Training Pipeline")
    parser.add_argument('--epochs', type=int, default=50, help="Total training epochs")
    parser.add_argument('--batch_size', type=int, default=16, help="Training batch size")
    parser.add_argument('--samples_per_class', type=int, default=200, help="Class balancing cap")
    parser.add_argument('--data_path', type=str, default='data', help="Dataset directory")
    parser.add_argument('--skip_phase1', action='store_true', help="Skip head-only training phase")
    args = parser.parse_args()

    start_time = time.time()
    
    print("==================================================")
    print("🚀 STEP 1: Validating & Cleaning Dataset")
    print("==================================================")
    csv_path = os.path.join(args.data_path, "skin_dataset.csv")
    out_path = os.path.join(args.data_path, "skin_dataset_cleaned.csv")
    cleaned_df = validate_dataset(csv_path, out_path)
    
    print("\n==================================================")
    print("🚀 STEP 2: Preprocessing & Creating Balanced CSV")
    print("==================================================")
    balanced_df = create_balanced_dataset(cleaned_df, samples_per_class=args.samples_per_class)
    train_df, val_df, test_df = split_dataset(balanced_df)
    
    print("\n==================================================")
    print(f"🚀 STEP 3: Training Model ({args.epochs} epochs)")
    print("==================================================")
    model, val_loader, test_loader, device = train_fixed_model(
        train_df, val_df, test_df,
        epochs=args.epochs, 
        batch_size=args.batch_size, 
        skip_phase1=args.skip_phase1
    )
    
    print("\n==================================================")
    print("🚀 STEP 4: Finding Optimal Thresholds")
    print("==================================================")
    optimal_thresholds = find_optimal_thresholds(model, val_loader, device, CONDITIONS)
    
    print("\n==================================================")
    print("🚀 STEP 5: Final Evaluation on Test Set")
    print("==================================================")
    evaluate_on_test(model, test_loader, device, CONDITIONS, optimal_thresholds)
    
    elapsed = time.time() - start_time
    hours, rem = divmod(elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"\n⏱️ Total Training Time: {int(hours):02d}h {int(minutes):02d}m {int(seconds):02d}s")
    print("\n✅ Training complete! Run python app.py to start the server.")

if __name__ == '__main__':
    main()
