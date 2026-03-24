"""
Data Mining & Exploratory Data Analysis
Phân tích dataset để đưa ra insights cho việc training
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm
from datetime import datetime
from collections import Counter

class Logger:
    """Logger để ghi output ra cả console và file"""
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, 'w', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()

def print_section(title):
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)

def analyze_dataset():
    print_section("DATA MINING & EXPLORATORY DATA ANALYSIS")
    
    # Create output directory
    output_dir = "data_mining_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load labels
    df = pd.read_csv("labels.csv")
    train_dir = "train"
    
    # ========================================================================
    # 1. BASIC STATISTICS
    # ========================================================================
    print_section("1. BASIC DATASET STATISTICS")
    
    total_samples = len(df)
    num_classes = df['breed'].nunique()
    
    print(f"Total samples: {total_samples}")
    print(f"Number of breeds (classes): {num_classes}")
    print(f"Columns: {df.columns.tolist()}")
    
    # ========================================================================
    # 2. CLASS DISTRIBUTION
    # ========================================================================
    print_section("2. CLASS DISTRIBUTION ANALYSIS")
    
    breed_counts = df['breed'].value_counts()
    
    print(f"\nMost common breed: {breed_counts.index[0]} ({breed_counts.iloc[0]} samples)")
    print(f"Least common breed: {breed_counts.index[-1]} ({breed_counts.iloc[-1]} samples)")
    print(f"Average samples per breed: {breed_counts.mean():.2f}")
    print(f"Median samples per breed: {breed_counts.median():.2f}")
    print(f"Imbalance ratio (max/min): {breed_counts.iloc[0] / breed_counts.iloc[-1]:.2f}")
    
    # Save distribution
    breed_counts.to_csv(f"{output_dir}/breed_distribution.csv", header=['count'])
    print(f"\n✓ Saved breed distribution to {output_dir}/breed_distribution.csv")
    
    # Visualize distribution
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    
    # Top 20 breeds
    breed_counts.head(20).plot(kind='bar', ax=axes[0], color='skyblue')
    axes[0].set_title('Top 20 Most Common Breeds', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Breed')
    axes[0].set_ylabel('Number of Samples')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Bottom 20 breeds
    breed_counts.tail(20).plot(kind='bar', ax=axes[1], color='salmon')
    axes[1].set_title('Top 20 Least Common Breeds', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Breed')
    axes[1].set_ylabel('Number of Samples')
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/class_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved class distribution plot to {output_dir}/class_distribution.png")
    
    # ========================================================================
    # 3. IMAGE PROPERTIES ANALYSIS
    # ========================================================================
    print_section("3. IMAGE PROPERTIES ANALYSIS")
    
    print("Analyzing image properties (sampling 500 images)...")
    
    sample_size = min(500, len(df))
    sample_df = df.sample(n=sample_size, random_state=42)
    
    widths = []
    heights = []
    aspect_ratios = []
    file_sizes = []
    
    for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="Processing images"):
        img_path = os.path.join(train_dir, f"{row['id']}.jpg")
        
        if os.path.exists(img_path):
            try:
                img = Image.open(img_path)
                w, h = img.size
                widths.append(w)
                heights.append(h)
                aspect_ratios.append(w / h)
                file_sizes.append(os.path.getsize(img_path) / 1024)  # KB
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
    
    print(f"\nImage Dimensions:")
    print(f"  Width  - Min: {min(widths)}, Max: {max(widths)}, Mean: {np.mean(widths):.2f}")
    print(f"  Height - Min: {min(heights)}, Max: {max(heights)}, Mean: {np.mean(heights):.2f}")
    print(f"\nAspect Ratios:")
    print(f"  Min: {min(aspect_ratios):.2f}, Max: {max(aspect_ratios):.2f}, Mean: {np.mean(aspect_ratios):.2f}")
    print(f"\nFile Sizes (KB):")
    print(f"  Min: {min(file_sizes):.2f}, Max: {max(file_sizes):.2f}, Mean: {np.mean(file_sizes):.2f}")
    
    # Visualize image properties
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].hist(widths, bins=30, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Image Width Distribution')
    axes[0, 0].set_xlabel('Width (pixels)')
    axes[0, 0].set_ylabel('Frequency')
    
    axes[0, 1].hist(heights, bins=30, color='lightcoral', edgecolor='black')
    axes[0, 1].set_title('Image Height Distribution')
    axes[0, 1].set_xlabel('Height (pixels)')
    axes[0, 1].set_ylabel('Frequency')
    
    axes[1, 0].hist(aspect_ratios, bins=30, color='lightgreen', edgecolor='black')
    axes[1, 0].set_title('Aspect Ratio Distribution')
    axes[1, 0].set_xlabel('Aspect Ratio (width/height)')
    axes[1, 0].set_ylabel('Frequency')
    
    axes[1, 1].hist(file_sizes, bins=30, color='plum', edgecolor='black')
    axes[1, 1].set_title('File Size Distribution')
    axes[1, 1].set_xlabel('File Size (KB)')
    axes[1, 1].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/image_properties.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Saved image properties plot to {output_dir}/image_properties.png")
    
    # Scatter plot: width vs height
    plt.figure(figsize=(10, 8))
    plt.scatter(widths, heights, alpha=0.5, c='steelblue')
    plt.title('Image Dimensions Scatter Plot', fontsize=14, fontweight='bold')
    plt.xlabel('Width (pixels)')
    plt.ylabel('Height (pixels)')
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/dimensions_scatter.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved dimensions scatter plot to {output_dir}/dimensions_scatter.png")
    
    # ========================================================================
    # 4. DATA QUALITY CHECK
    # ========================================================================
    print_section("4. DATA QUALITY CHECK")
    
    print("Checking for missing or corrupted images...")
    
    missing_images = []
    corrupted_images = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Checking images"):
        img_path = os.path.join(train_dir, f"{row['id']}.jpg")
        
        if not os.path.exists(img_path):
            missing_images.append(row['id'])
        else:
            try:
                img = Image.open(img_path)
                img.verify()
            except Exception:
                corrupted_images.append(row['id'])
    
    print(f"\nMissing images: {len(missing_images)}")
    print(f"Corrupted images: {len(corrupted_images)}")
    
    if len(missing_images) > 0:
        print(f"  First few missing: {missing_images[:5]}")
    if len(corrupted_images) > 0:
        print(f"  First few corrupted: {corrupted_images[:5]}")
    
    # ========================================================================
    # 5. INSIGHTS & RECOMMENDATIONS
    # ========================================================================
    print_section("5. INSIGHTS & RECOMMENDATIONS")
    
    insights = []
    
    # Class imbalance
    imbalance_ratio = breed_counts.iloc[0] / breed_counts.iloc[-1]
    if imbalance_ratio > 2:
        insights.append(f"⚠️ Class imbalance detected (ratio: {imbalance_ratio:.2f})")
        insights.append("   → Consider using weighted loss or data augmentation")
    
    # Image dimensions
    avg_width = np.mean(widths)
    avg_height = np.mean(heights)
    if avg_width > 500 or avg_height > 500:
        insights.append(f"📏 Large image dimensions (avg: {avg_width:.0f}x{avg_height:.0f})")
        insights.append("   → Resize to 224x224 or 299x299 for efficiency")
    
    # Aspect ratios
    aspect_std = np.std(aspect_ratios)
    if aspect_std > 0.3:
        insights.append(f"📐 High aspect ratio variance (std: {aspect_std:.2f})")
        insights.append("   → Use RandomResizedCrop for augmentation")
    
    # Data quality
    if len(missing_images) > 0 or len(corrupted_images) > 0:
        insights.append(f"❌ Data quality issues: {len(missing_images)} missing, {len(corrupted_images)} corrupted")
        insights.append("   → Clean dataset before training")
    else:
        insights.append("✅ No data quality issues detected")
    
    # Dataset size
    if total_samples < 5000:
        insights.append(f"📊 Small dataset ({total_samples} samples)")
        insights.append("   → Use strong augmentation and pretrained models")
    
    for insight in insights:
        print(insight)
    
    # ========================================================================
    # 6. SAVE REPORT
    # ========================================================================
    print_section("6. SAVING MINING REPORT")
    
    report = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "basic_stats": {
            "total_samples": int(total_samples),
            "num_classes": int(num_classes),
            "samples_per_class_mean": float(breed_counts.mean()),
            "samples_per_class_median": float(breed_counts.median()),
            "imbalance_ratio": float(imbalance_ratio)
        },
        "image_properties": {
            "width_mean": float(np.mean(widths)),
            "height_mean": float(np.mean(heights)),
            "aspect_ratio_mean": float(np.mean(aspect_ratios)),
            "file_size_mean_kb": float(np.mean(file_sizes))
        },
        "data_quality": {
            "missing_images": len(missing_images),
            "corrupted_images": len(corrupted_images)
        },
        "insights": insights
    }
    
    with open(f"{output_dir}/mining_report.json", 'w') as f:
        json.dump(report, f, indent=4)
    
    print(f"✓ Saved complete report to {output_dir}/mining_report.json")
    
    print_section("DATA MINING COMPLETED")
    print("\nGenerated files:")
    print(f"  - {output_dir}/mining_report.json")
    print(f"  - {output_dir}/breed_distribution.csv")
    print(f"  - {output_dir}/class_distribution.png")
    print(f"  - {output_dir}/image_properties.png")
    print(f"  - {output_dir}/dimensions_scatter.png")

if __name__ == "__main__":
    # Setup logger
    output_dir = "data_mining_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    log_file = f"{output_dir}/data_mining_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    logger = Logger(log_file)
    sys.stdout = logger
    
    print(f"Data Mining started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Log file: {log_file}\n")
    
    analyze_dataset()
    
    print(f"\nData Mining completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Log saved to: {log_file}")
    
    # Đóng logger
    sys.stdout = logger.terminal
    logger.close()
