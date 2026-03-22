"""
DATA MINING & EXPLORATORY DATA ANALYSIS
Dog Breed Classification Dataset

QUAN TRỌNG:
- Data Mining phục vụ cho CẢ DỰ ÁN, không chỉ 1 model
- Kết quả dùng chung cho VGG, ResNet, AlexNet, EfficientNet
- Chỉ cần chạy 1 LẦN trước khi train bất kỳ model nào
- Không bắt buộc nhưng HIGHLY RECOMMENDED

MỤC ĐÍCH:
1. Hiểu dataset (statistics, distribution)
2. Phát hiện vấn đề (imbalance, missing data, corrupted files)
3. Đưa ra insights và recommendations
4. Giúp quyết định: augmentation, model selection, hyperparameters

OUTPUT:
- mining_report.json: Báo cáo tổng hợp
- Visualizations: Charts và plots
- Recommendations: Đề xuất cụ thể cho training
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from collections import Counter
import json


def print_section(title):
    """Print formatted section header"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)


def analyze_dataset_statistics(df, train_dir):
    """Analyze basic dataset statistics"""
    print_section("1. DATASET STATISTICS")
    
    print(f"Total samples: {len(df)}")
    print(f"Number of unique breeds: {df['breed'].nunique()}")
    print(f"Number of features: {df.shape[1]}")
    
    # Class distribution
    breed_counts = df['breed'].value_counts()
    print(f"\nClass distribution:")
    print(f"  - Min samples per breed: {breed_counts.min()}")
    print(f"  - Max samples per breed: {breed_counts.max()}")
    print(f"  - Mean samples per breed: {breed_counts.mean():.2f}")
    print(f"  - Median samples per breed: {breed_counts.median():.2f}")
    print(f"  - Std samples per breed: {breed_counts.std():.2f}")
    
    # Check for missing values
    print(f"\nMissing values:")
    print(df.isnull().sum())
    
    # Check for duplicates
    duplicates = df.duplicated().sum()
    print(f"\nDuplicate rows: {duplicates}")
    
    return breed_counts


def analyze_class_imbalance(breed_counts):
    """Analyze class imbalance"""
    print_section("2. CLASS IMBALANCE ANALYSIS")
    
    # Calculate imbalance ratio
    max_count = breed_counts.max()
    min_count = breed_counts.min()
    imbalance_ratio = max_count / min_count
    
    print(f"Imbalance ratio: {imbalance_ratio:.2f}")
    print(f"Most common breed: {breed_counts.index[0]} ({breed_counts.iloc[0]} samples)")
    print(f"Least common breed: {breed_counts.index[-1]} ({breed_counts.iloc[-1]} samples)")
    
    # Categorize classes
    q1 = breed_counts.quantile(0.25)
    q3 = breed_counts.quantile(0.75)
    
    underrepresented = breed_counts[breed_counts < q1]
    well_represented = breed_counts[(breed_counts >= q1) & (breed_counts <= q3)]
    overrepresented = breed_counts[breed_counts > q3]
    
    print(f"\nClass categories:")
    print(f"  - Underrepresented (<Q1={q1:.0f}): {len(underrepresented)} breeds")
    print(f"  - Well-represented (Q1-Q3): {len(well_represented)} breeds")
    print(f"  - Overrepresented (>Q3={q3:.0f}): {len(overrepresented)} breeds")
    
    return {
        'imbalance_ratio': imbalance_ratio,
        'underrepresented': len(underrepresented),
        'well_represented': len(well_represented),
        'overrepresented': len(overrepresented)
    }


def analyze_image_properties(df, train_dir, sample_size=1000):
    """Analyze image properties (size, aspect ratio, color)"""
    print_section("3. IMAGE PROPERTIES ANALYSIS")
    
    print(f"Analyzing {sample_size} random images...")
    
    sample_df = df.sample(n=min(sample_size, len(df)), random_state=42)
    
    widths = []
    heights = []
    aspect_ratios = []
    file_sizes = []
    
    for idx, row in sample_df.iterrows():
        img_path = os.path.join(train_dir, row['id'] + '.jpg')
        if os.path.exists(img_path):
            img = Image.open(img_path)
            w, h = img.size
            widths.append(w)
            heights.append(h)
            aspect_ratios.append(w / h)
            file_sizes.append(os.path.getsize(img_path) / 1024)  # KB
    
    print(f"\nImage dimensions:")
    print(f"  Width:  min={min(widths)}, max={max(widths)}, mean={np.mean(widths):.2f}")
    print(f"  Height: min={min(heights)}, max={max(heights)}, mean={np.mean(heights):.2f}")
    
    print(f"\nAspect ratios:")
    print(f"  Min: {min(aspect_ratios):.2f}")
    print(f"  Max: {max(aspect_ratios):.2f}")
    print(f"  Mean: {np.mean(aspect_ratios):.2f}")
    print(f"  Median: {np.median(aspect_ratios):.2f}")
    
    print(f"\nFile sizes (KB):")
    print(f"  Min: {min(file_sizes):.2f}")
    print(f"  Max: {max(file_sizes):.2f}")
    print(f"  Mean: {np.mean(file_sizes):.2f}")
    print(f"  Median: {np.median(file_sizes):.2f}")
    
    return {
        'widths': widths,
        'heights': heights,
        'aspect_ratios': aspect_ratios,
        'file_sizes': file_sizes
    }


def analyze_data_quality(df, train_dir):
    """Check data quality issues"""
    print_section("4. DATA QUALITY ANALYSIS")
    
    missing_images = []
    corrupted_images = []
    
    print("Checking image files...")
    for idx, row in df.iterrows():
        img_path = os.path.join(train_dir, row['id'] + '.jpg')
        
        # Check if file exists
        if not os.path.exists(img_path):
            missing_images.append(row['id'])
            continue
        
        # Check if image can be opened
        try:
            img = Image.open(img_path)
            img.verify()
        except:
            corrupted_images.append(row['id'])
    
    print(f"\nData quality issues:")
    print(f"  - Missing images: {len(missing_images)}")
    print(f"  - Corrupted images: {len(corrupted_images)}")
    print(f"  - Valid images: {len(df) - len(missing_images) - len(corrupted_images)}")
    
    if missing_images:
        print(f"\nMissing image IDs (first 5): {missing_images[:5]}")
    if corrupted_images:
        print(f"\nCorrupted image IDs (first 5): {corrupted_images[:5]}")
    
    return {
        'missing': len(missing_images),
        'corrupted': len(corrupted_images),
        'valid': len(df) - len(missing_images) - len(corrupted_images)
    }


def create_visualizations(df, breed_counts, image_props):
    """Create data mining visualizations"""
    print_section("5. CREATING VISUALIZATIONS")
    
    if not os.path.exists('data_mining_results'):
        os.makedirs('data_mining_results')
    
    # 1. Class distribution
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Top 20 breeds
    top_20 = breed_counts.head(20)
    axes[0, 0].barh(range(len(top_20)), top_20.values, color='#3498db')
    axes[0, 0].set_yticks(range(len(top_20)))
    axes[0, 0].set_yticklabels(top_20.index, fontsize=8)
    axes[0, 0].set_xlabel('Number of Samples', fontweight='bold')
    axes[0, 0].set_title('Top 20 Most Common Breeds', fontweight='bold')
    axes[0, 0].invert_yaxis()
    axes[0, 0].grid(axis='x', alpha=0.3)
    
    # Bottom 20 breeds
    bottom_20 = breed_counts.tail(20)
    axes[0, 1].barh(range(len(bottom_20)), bottom_20.values, color='#e74c3c')
    axes[0, 1].set_yticks(range(len(bottom_20)))
    axes[0, 1].set_yticklabels(bottom_20.index, fontsize=8)
    axes[0, 1].set_xlabel('Number of Samples', fontweight='bold')
    axes[0, 1].set_title('Top 20 Least Common Breeds', fontweight='bold')
    axes[0, 1].invert_yaxis()
    axes[0, 1].grid(axis='x', alpha=0.3)
    
    # Distribution histogram
    axes[1, 0].hist(breed_counts.values, bins=30, color='#2ecc71', alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Samples per Breed', fontweight='bold')
    axes[1, 0].set_ylabel('Frequency', fontweight='bold')
    axes[1, 0].set_title('Distribution of Samples per Breed', fontweight='bold')
    axes[1, 0].axvline(breed_counts.mean(), color='red', linestyle='--', label=f'Mean: {breed_counts.mean():.1f}')
    axes[1, 0].axvline(breed_counts.median(), color='blue', linestyle='--', label=f'Median: {breed_counts.median():.1f}')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # Box plot
    axes[1, 1].boxplot(breed_counts.values, vert=True)
    axes[1, 1].set_ylabel('Samples per Breed', fontweight='bold')
    axes[1, 1].set_title('Box Plot of Class Distribution', fontweight='bold')
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('data_mining_results/class_distribution.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: class_distribution.png")
    
    # 2. Image properties
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Width distribution
    axes[0, 0].hist(image_props['widths'], bins=50, color='#3498db', alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('Width (pixels)', fontweight='bold')
    axes[0, 0].set_ylabel('Frequency', fontweight='bold')
    axes[0, 0].set_title('Image Width Distribution', fontweight='bold')
    axes[0, 0].grid(alpha=0.3)
    
    # Height distribution
    axes[0, 1].hist(image_props['heights'], bins=50, color='#e74c3c', alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('Height (pixels)', fontweight='bold')
    axes[0, 1].set_ylabel('Frequency', fontweight='bold')
    axes[0, 1].set_title('Image Height Distribution', fontweight='bold')
    axes[0, 1].grid(alpha=0.3)
    
    # Aspect ratio distribution
    axes[1, 0].hist(image_props['aspect_ratios'], bins=50, color='#2ecc71', alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Aspect Ratio (W/H)', fontweight='bold')
    axes[1, 0].set_ylabel('Frequency', fontweight='bold')
    axes[1, 0].set_title('Aspect Ratio Distribution', fontweight='bold')
    axes[1, 0].axvline(1.0, color='red', linestyle='--', label='Square (1:1)')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # File size distribution
    axes[1, 1].hist(image_props['file_sizes'], bins=50, color='#f39c12', alpha=0.7, edgecolor='black')
    axes[1, 1].set_xlabel('File Size (KB)', fontweight='bold')
    axes[1, 1].set_ylabel('Frequency', fontweight='bold')
    axes[1, 1].set_title('File Size Distribution', fontweight='bold')
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('data_mining_results/image_properties.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: image_properties.png")
    
    # 3. Scatter plot: Width vs Height
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(image_props['widths'], image_props['heights'], 
                        alpha=0.5, c=image_props['aspect_ratios'], 
                        cmap='viridis', s=20)
    ax.set_xlabel('Width (pixels)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Height (pixels)', fontweight='bold', fontsize=12)
    ax.set_title('Image Dimensions Scatter Plot', fontweight='bold', fontsize=14)
    ax.plot([0, max(image_props['widths'])], [0, max(image_props['widths'])], 
            'r--', alpha=0.5, label='Square (1:1)')
    ax.legend()
    ax.grid(alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Aspect Ratio', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('data_mining_results/dimensions_scatter.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: dimensions_scatter.png")
    
    plt.close('all')


def generate_insights(df, breed_counts, imbalance_info, quality_info):
    """Generate data mining insights and recommendations"""
    print_section("6. DATA MINING INSIGHTS & RECOMMENDATIONS")
    
    insights = []
    recommendations = []
    
    # Class imbalance insights
    if imbalance_info['imbalance_ratio'] > 2:
        insights.append(f"⚠️ High class imbalance detected (ratio: {imbalance_info['imbalance_ratio']:.2f})")
        recommendations.append("Consider using class weights or oversampling techniques (SMOTE, data augmentation)")
    
    if imbalance_info['underrepresented'] > len(breed_counts) * 0.3:
        insights.append(f"⚠️ {imbalance_info['underrepresented']} breeds are underrepresented")
        recommendations.append("Apply stronger augmentation for minority classes")
    
    # Data quality insights
    if quality_info['missing'] > 0:
        insights.append(f"⚠️ {quality_info['missing']} missing images detected")
        recommendations.append("Remove or replace missing images before training")
    
    if quality_info['corrupted'] > 0:
        insights.append(f"⚠️ {quality_info['corrupted']} corrupted images detected")
        recommendations.append("Clean corrupted images from dataset")
    
    # Dataset size insights
    if len(df) < 5000:
        insights.append("⚠️ Small dataset size")
        recommendations.append("Use transfer learning with pretrained models (VGG, ResNet, EfficientNet)")
        recommendations.append("Apply extensive data augmentation")
    
    if len(breed_counts) > 100:
        insights.append(f"ℹ️ Large number of classes ({len(breed_counts)})")
        recommendations.append("Consider using deeper models or ensemble methods")
        recommendations.append("Use label smoothing to prevent overfitting")
    
    # Print insights
    print("\nKey Insights:")
    for i, insight in enumerate(insights, 1):
        print(f"{i}. {insight}")
    
    print("\nRecommendations:")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")
    
    return {
        'insights': insights,
        'recommendations': recommendations
    }


def save_mining_report(df, breed_counts, imbalance_info, quality_info, insights_info):
    """Save complete data mining report"""
    print_section("7. SAVING DATA MINING REPORT")
    
    report = {
        'dataset_overview': {
            'total_samples': len(df),
            'num_classes': len(breed_counts),
            'samples_per_class': {
                'min': int(breed_counts.min()),
                'max': int(breed_counts.max()),
                'mean': float(breed_counts.mean()),
                'median': float(breed_counts.median()),
                'std': float(breed_counts.std())
            }
        },
        'class_imbalance': imbalance_info,
        'data_quality': quality_info,
        'insights': insights_info['insights'],
        'recommendations': insights_info['recommendations']
    }
    
    with open('data_mining_results/mining_report.json', 'w') as f:
        json.dump(report, f, indent=4)
    
    print("✓ Report saved: data_mining_results/mining_report.json")
    
    # Save breed distribution CSV
    breed_df = pd.DataFrame({
        'breed': breed_counts.index,
        'count': breed_counts.values
    })
    breed_df.to_csv('data_mining_results/breed_distribution.csv', index=False)
    print("✓ Distribution saved: data_mining_results/breed_distribution.csv")


if __name__ == "__main__":
    print("="*80)
    print("  DATA MINING & EXPLORATORY DATA ANALYSIS")
    print("  Dog Breed Classification Dataset")
    print("="*80)
    
    labels_path = "labels.csv"
    train_dir = "train"
    
    # Load dataset
    df = pd.read_csv(labels_path)
    
    # 1. Dataset statistics
    breed_counts = analyze_dataset_statistics(df, train_dir)
    
    # 2. Class imbalance analysis
    imbalance_info = analyze_class_imbalance(breed_counts)
    
    # 3. Image properties analysis
    image_props = analyze_image_properties(df, train_dir, sample_size=1000)
    
    # 4. Data quality analysis
    quality_info = analyze_data_quality(df, train_dir)
    
    # 5. Create visualizations
    create_visualizations(df, breed_counts, image_props)
    
    # 6. Generate insights
    insights_info = generate_insights(df, breed_counts, imbalance_info, quality_info)
    
    # 7. Save report
    save_mining_report(df, breed_counts, imbalance_info, quality_info, insights_info)
    
    print("\n" + "="*80)
    print("  DATA MINING ANALYSIS COMPLETED!")
    print("="*80)
    print("Results saved in: data_mining_results/")
    print("  - mining_report.json")
    print("  - breed_distribution.csv")
    print("  - class_distribution.png")
    print("  - image_properties.png")
    print("  - dimensions_scatter.png")
    print("="*80)
