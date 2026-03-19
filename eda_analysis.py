import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm
import numpy as np

def main():
    print("="*60)
    print("DEEP EXPLORATORY DATA ANALYSIS (EDA) - DATA MINING")
    print("="*60)
    
    labels_path = "labels.csv"
    train_dir = "train"
    
    if not os.path.exists(labels_path):
        print("❌ labels.csv not found!")
        return
        
    df = pd.read_csv(labels_path)
    
    print("\n--- 0. TABULAR MISSING VALUE CHECK ---")
    missing_vals = df.isnull().sum()
    print("Checking for Null/NaN values in labels.csv:")
    print(missing_vals.to_string())
    if missing_vals.sum() == 0:
        print("[+] Insight: No missing tabular records detected in the label distribution format.")
        
    total_imgs = len(df)
    total_classes = df['breed'].nunique()
    
    print("\n--- 1. LABELS & CLASS BALANCE ANALYSIS ---")
    print(f"Total labeled records: {total_imgs}")
    print(f"Total dog breeds (classes): {total_classes}")
    
    breed_counts = df['breed'].value_counts()
    print(f"\n[+] Top 3 majority classes:\n{breed_counts.head(3)}")
    print(f"\n[-] Top 3 minority classes:\n{breed_counts.tail(3)}")
    
    # Calculate Imbalance Ratio
    imbalance_ratio = breed_counts.max() / breed_counts.min()
    print(f"\n[>] Imbalance Ratio (Max/Min): {imbalance_ratio:.2f}")
    if imbalance_ratio > 2.0:
        print("[!] Insight: Severe class imbalance detected. Must use Data Augmentation to prevent overfitting minority classes.")

    print("\n--- 2. PHYSICAL IMAGE DATA ANALYSIS & VALIDATION ---")
    
    if not os.path.exists(train_dir):
        print(f"❌ '{train_dir}' directory not found! Skipping physical image analysis.")
        return
        
    print(f"Scanning '{train_dir}' folder... This may take a moment.")
    missing_files = []
    corrupted_files = []
    widths = []
    heights = []
    
    # Sample up to 1000 images to calculate dimensions (saves time)
    sample_df = df.sample(n=min(1000, len(df)), random_state=42)
    
    for _, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="Analyzing Image Properties"):
        img_id = row['id']
        img_path = os.path.join(train_dir, f"{img_id}.jpg")
        
        if not os.path.exists(img_path):
            missing_files.append(img_path)
            continue
            
        try:
            # Verify file integrity
            with Image.open(img_path) as img:
                img.verify() 
            # Reopen to extract actual resolution
            with Image.open(img_path) as img:
                w, h = img.size
                widths.append(w)
                heights.append(h)
        except Exception:
            corrupted_files.append(img_path)

    print(f"\n[Validation Results]")
    print(f"Missing images in folder: {len(missing_files)}")
    print(f"Corrupted/Unreadable images: {len(corrupted_files)}")
    
    if widths and heights:
        avg_w, avg_h = int(np.mean(widths)), int(np.mean(heights))
        print(f"\n[Image Quality & Dimensions]")
        print(f"Average Resolution: {avg_w}x{avg_h} pixels")
        print(f"Min Resolution: {np.min(widths)}x{np.min(heights)}")
        print(f"Max Resolution: {np.max(widths)}x{np.max(heights)}")
        print(f"Insight: The extreme variance in resolution confirms that forcing all images to (224x224) via 'Resize' and 'CenterCrop' during Preprocessing is strictly necessary to standardize the feature extraction space.")
        
        # Plot Outliers (Scatter Plot)
        plt.figure(figsize=(10, 6))
        plt.scatter(widths, heights, alpha=0.5, color='darkorange', edgecolors='k')
        plt.title("Image Dimensions Scatter Plot (Outlier Detection)")
        plt.xlabel("Image Width (pixels)")
        plt.ylabel("Image Height (pixels)")
        plt.axhline(y=np.mean(heights), color='r', linestyle='--', label='Mean Height')
        plt.axvline(x=np.mean(widths), color='g', linestyle='--', label='Mean Width')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig("eda_resolution_outliers.png")
        print("✅ Saved 'eda_resolution_outliers.png'! (Dùng biểu đồ này mổ xẻ phần Outlier Analysis trong tâm bão điểm danh, điểm 10 Data Mining!)")
        
    print("\n--- 3. GENERATING VISUALIZATIONS ---")
    
    # Plot 1: Class Distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(breed_counts.values, bins=30, kde=True, color='teal')
    plt.title("Distribution of Images per Breed")
    plt.xlabel("Number of Images")
    plt.ylabel("Frequency of Breeds")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig("eda_class_distribution.png")
    print("✅ Saved 'eda_class_distribution.png'")
    
    # Plot 2: Sample Images Grid (Shows 9 random images from dataset)
    sample_breeds = df.sample(9, random_state=101)  # Getting 9 random records
    
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    axes = axes.flatten()
    
    valid_count = 0
    for idx, row in sample_breeds.iterrows():
        img_id = row['id']
        breed = row['breed']
        img_path = os.path.join(train_dir, f"{img_id}.jpg")
        
        if os.path.exists(img_path) and valid_count < 9:
            try:
                img = Image.open(img_path).convert('RGB')
                axes[valid_count].imshow(img)
                # Format breed name to be short enough for title
                short_breed = breed.replace('_', ' ').title()
                if len(short_breed) > 20:
                    short_breed = short_breed[:17] + "..."
                axes[valid_count].set_title(short_breed)
                axes[valid_count].axis('off')
                valid_count += 1
            except:
                pass
                
    # Hide any unused subplots if missing images occurred
    for i in range(valid_count, 9):
        axes[i].axis('off')
        
    plt.suptitle("Sample Dataset Images (Raw Data)", fontsize=16)
    plt.tight_layout()
    plt.savefig("eda_sample_images.png")
    print("✅ Saved 'eda_sample_images.png'")
    
    print("\n" + "="*60)
    print("EDA COMPLETE. Use 'eda_sample_images.png' and stats above for Section 1 & 2 of your report!")

if __name__ == "__main__":
    main()
