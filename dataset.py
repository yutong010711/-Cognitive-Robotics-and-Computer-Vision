import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# Configuration - UPDATED FOR YOUR LINUX SYSTEM
RAW_DATA_PATH = "/home/mscrobotics2425laptop12/Desktop/vision/rgbd-dataset/rgbd-dataset"
PROCESSED_ROOT = "/home/mscrobotics2425laptop12/Desktop/vision/rgbd-dataset/processed"

def verify_dataset_structure():
    """Check if the dataset has the expected structure"""
    print(f"Verifying dataset structure at: {RAW_DATA_PATH}")
    
    # Check if main path exists
    if not os.path.exists(RAW_DATA_PATH):
        print(f"Error: Dataset not found at specified path: {RAW_DATA_PATH}")
        print("Please verify:")
        print("1. The dataset has been downloaded and extracted")
        print("2. The path is correct (case-sensitive in Linux)")
        print("3. You have read permissions for the directory")
        return False
    
    # Check for expected subfolders (sample categories)
    test_categories = ['apple', 'ball', 'banana', 'bowl']  # Add more if needed
    missing = []
    
    for category in test_categories:
        cat_path = os.path.join(RAW_DATA_PATH, category)
        if not os.path.exists(cat_path):
            missing.append(category)
    
    if missing:
        print(f"Warning: Missing expected categories: {missing}")
        print("The dataset might be incomplete or structured differently")
        print("Continue anyway? [y/n]")
        if input().lower() != 'y':
            return False
    
    print("Dataset structure verification passed")
    return True

def create_processing_folders():
    """Create folder structure for processed data with proper permissions"""
    dirs = {
        'cnn': os.path.join(PROCESSED_ROOT, 'cnn'),
        'traditional': os.path.join(PROCESSED_ROOT, 'traditional'),
        'stats': os.path.join(PROCESSED_ROOT, 'stats'),
        'temp': os.path.join(PROCESSED_ROOT, 'temp')  # For intermediate processing
    }
    
    try:
        for dir_path in dirs.values():
            os.makedirs(dir_path, exist_ok=True)
            # Ensure proper permissions (rwx for user, rx for group/others)
            os.chmod(dir_path, 0o755)
        
        print(f"Created processing directories at: {PROCESSED_ROOT}")
        return dirs
    
    except PermissionError:
        print(f"Error: Cannot create directories at {PROCESSED_ROOT}")
        print("You may need to:")
        print("1. Run the script with sudo (not recommended)")
        print("2. Change ownership of the parent directory")
        print(f"Try: sudo chown -R $USER:$USER {os.path.dirname(PROCESSED_ROOT)}")
        return None

def process_category(category, output_dirs):
    """Process all images in a single category"""
    category_path = os.path.join(RAW_DATA_PATH, category)
    processed_count = 0
    
    for instance in os.listdir(category_path):
        instance_path = os.path.join(category_path, instance)
        
        if not os.path.isdir(instance_path):
            continue
            
        for file in os.listdir(instance_path):
            if file.endswith('_crop.png'):  # Process only cropped RGB images
                img_path = os.path.join(instance_path, file)
                
                try:
                    # Read image (ensure proper permissions)
                    img = cv2.imread(img_path)
                    if img is None:
                        print(f"Warning: Could not read {img_path} - may be corrupt")
                        continue
                    
                    # CNN processing
                    img_cnn = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img_cnn = cv2.resize(img_cnn, (224, 224))
                    img_cnn = img_cnn.astype(np.float32) / 255.0
                    
                    # Traditional CV processing
                    img_trad = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Keep as RGB
                    
                    # Save processed images
                    base_name = f"{instance}_{os.path.splitext(file)[0]}"
                    
                    np.save(
                        os.path.join(output_dirs['cnn'], category, f"{base_name}.npy"),
                        img_cnn
                    )
                    
                    cv2.imwrite(
                        os.path.join(output_dirs['traditional'], category, f"{base_name}.jpg"),
                        cv2.cvtColor(img_trad, cv2.COLOR_RGB2BGR)  # Save as BGR for OpenCV
                    )
                    
                    processed_count += 1
                    
                except Exception as e:
                    print(f"Error processing {img_path}: {str(e)}")
                    continue
    
    return processed_count

def preprocess_dataset():
    """Main preprocessing workflow"""
    print("\n" + "="*50)
    print("RGB-D Dataset Preprocessing")
    print(f"Source: {RAW_DATA_PATH}")
    print(f"Destination: {PROCESSED_ROOT}")
    print("="*50 + "\n")
    
    if not verify_dataset_structure():
        return
    
    output_dirs = create_processing_folders()
    if not output_dirs:
        return
    
    # Create category subfolders
    categories = [d for d in os.listdir(RAW_DATA_PATH) 
                 if os.path.isdir(os.path.join(RAW_DATA_PATH, d))]
    
    for category in categories:
        os.makedirs(os.path.join(output_dirs['cnn'], category), exist_ok=True)
        os.makedirs(os.path.join(output_dirs['traditional'], category), exist_ok=True)
    
    # Process all categories
    total_processed = 0
    stats = {}
    
    for category in categories:
        print(f"\nProcessing category: {category}")
        count = process_category(category, output_dirs)
        stats[category] = count
        total_processed += count
        print(f"Processed {count} images in {category}")
    
    # Save statistics
    stats_path = os.path.join(output_dirs['stats'], 'processing_stats.txt')
    with open(stats_path, 'w') as f:
        f.write("RGB-D Dataset Processing Report\n")
        f.write(f"Date: {datetime.datetime.now()}\n")
        f.write(f"Source: {RAW_DATA_PATH}\n")
        f.write(f"Total categories processed: {len(categories)}\n")
        f.write(f"Total images processed: {total_processed}\n\n")
        
        f.write("Category-wise counts:\n")
        for category, count in stats.items():
            f.write(f"{category}: {count}\n")
    
    print("\n" + "="*50)
    print("Preprocessing completed successfully!")
    print(f"Total images processed: {total_processed}")
    print(f"CNN data saved to: {output_dirs['cnn']}")
    print(f"Traditional CV data saved to: {output_dirs['traditional']}")
    print(f"Statistics saved to: {stats_path}")
    print("="*50)

if __name__ == "__main__":
    import datetime  # For timestamping the stats file
    
    try:
        preprocess_dataset()
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
    except Exception as e:
        print(f"\nFatal error: {str(e)}")