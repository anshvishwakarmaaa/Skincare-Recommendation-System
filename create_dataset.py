import os
import pandas as pd

def create_skin_dataset():
    """Create dataset CSV from all 8 condition folders"""
    
    dataset_path = "data/raw_images/dataset"
    
    # All 8 skin conditions
    conditions = ['acne', 'blackheads', 'dark spots', 'dryness', 
                  'normal', 'oily', 'pores', 'wrinkles']
    
    data = []
    total_images = 0
    
    print("=== Creating Complete Dataset CSV ===")
    print("Processing all 8 skin conditions...\n")
    
    for condition in conditions:
        condition_path = os.path.join(dataset_path, condition)
        
        if os.path.exists(condition_path):
            images = [f for f in os.listdir(condition_path) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            print(f"✅ {condition}: {len(images)} images")
            
            for img_name in images:
                img_path = os.path.join(condition_path, img_name)
                
                # Create multi-label entry
                entry = {'image_path': img_path, 'filename': img_name}
                for cond in conditions:
                    entry[cond] = 1 if cond == condition else 0
                
                data.append(entry)
            
            total_images += len(images)
        else:
            print(f"❌ Folder not found: {condition_path}")
    
    if data:
        df = pd.DataFrame(data)
        df.to_csv('data/skin_dataset.csv', index=False)
        
        print(f"\n🎉 SUCCESS: Created complete dataset with {len(df)} images!")
        
        # Show detailed summary
        print("\n📊 COMPLETE DATASET SUMMARY:")
        print("-" * 40)
        for condition in conditions:
            count = df[condition].sum()
            percentage = (count / total_images) * 100
            print(f"   {condition:12}: {count:4} images ({percentage:5.1f}%)")
        
        print("-" * 40)
        print(f"   {'TOTAL':12}: {total_images:4} images (100.0%)")
        
        # Save dataset info
        dataset_info = {
            'total_images': total_images,
            'conditions': conditions,
            'class_distribution': {cond: df[cond].sum() for cond in conditions}
        }
        
        print(f"\n💾 Dataset saved as: data/skin_dataset.csv")
        
        return df
    else:
        print("❌ ERROR: No images found!")
        return None

if __name__ == "__main__":
    create_skin_dataset()