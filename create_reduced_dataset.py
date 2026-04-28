import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def create_reduced_dataset(target_total=2000):
    """Create a balanced reduced dataset"""
    
    dataset_path = "data/raw_images/dataset"
    conditions = ['acne', 'blackheads', 'dark spots', 'dryness', 
                  'normal', 'oily', 'pores', 'wrinkles']
    
    # Calculate images per condition for balanced dataset
    images_per_condition = target_total // len(conditions)
    print(f"🎯 Target: {target_total} total images")
    print(f"📊 Images per condition: {images_per_condition}")
    
    data = []
    
    print("\n=== Creating Reduced Balanced Dataset ===")
    
    for condition in conditions:
        condition_path = os.path.join(dataset_path, condition)
        
        if os.path.exists(condition_path):
            images = [f for f in os.listdir(condition_path) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            # Take only the required number of images
            if len(images) > images_per_condition:
                selected_images = np.random.choice(images, images_per_condition, replace=False)
            else:
                selected_images = images  # Use all if less than target
                
            print(f"✅ {condition}: {len(selected_images)}/{len(images)} images")
            
            for img_name in selected_images:
                img_path = os.path.join(condition_path, img_name)
                
                # Create multi-label entry
                entry = {'image_path': img_path, 'filename': img_name}
                for cond in conditions:
                    entry[cond] = 1 if cond == condition else 0
                
                data.append(entry)
        else:
            print(f"❌ Folder not found: {condition_path}")
    
    if data:
        df = pd.DataFrame(data)
        df.to_csv('data/skin_dataset_reduced.csv', index=False)
        
        print(f"\n🎉 SUCCESS: Created reduced dataset with {len(df)} images!")
        
        # Show summary
        print("\n📊 REDUCED DATASET SUMMARY:")
        print("-" * 40)
        total_images = 0
        for condition in conditions:
            count = df[condition].sum()
            total_images += count
            percentage = (count / len(df)) * 100
            print(f"   {condition:12}: {count:4} images ({percentage:5.1f}%)")
        
        print("-" * 40)
        print(f"   {'TOTAL':12}: {total_images:4} images")
        
        return df
    else:
        print("❌ ERROR: No images found!")
        return None

def main():
    """Main function with size options"""
    print("=" * 50)
    print("📦 DATASET SIZE REDUCTION TOOL")
    print("=" * 50)
    print("\nChoose your dataset size:")
    print("1. Small (1,600 images) - For low-end laptops")
    print("2. Medium (3,200 images) - For average laptops") 
    print("3. Large (4,800 images) - For good laptops")
    print("4. Custom size")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == '1':
        target_total = 1600
    elif choice == '2':
        target_total = 3200
    elif choice == '3':
        target_total = 4800
    elif choice == '4':
        try:
            target_total = int(input("Enter custom size (e.g., 2000): "))
        except:
            target_total = 2000
    else:
        target_total = 2000  # Default
    
    print(f"\nCreating dataset with {target_total} images...")
    create_reduced_dataset(target_total)

if __name__ == "__main__":
    main()