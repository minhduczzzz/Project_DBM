"""
Error Analysis & Confusion Matrix for VGG16
Phân tích lỗi và patterns để hiểu model behavior
"""

import sys
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

from dataset import DogBreedTrainValDataset
from model_vgg import DogBreedVGG16
from transforms import get_val_transform

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

def analyze_errors():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_section("VGG16 ERROR ANALYSIS & CONFUSION MATRIX")
    
    # 1. Load model và data
    print("\n1. Loading model and data...")
    checkpoint = torch.load("training_models/best_vgg.pth", map_location=device)
    class_to_idx = checkpoint["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(class_to_idx)
    
    # Recreate test set
    df = pd.read_csv("labels.csv")
    _, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["breed"])
    
    val_transform = get_val_transform(input_size=224)
    test_dataset = DogBreedTrainValDataset("train", test_df, val_transform, class_to_idx=class_to_idx)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)
    
    # Load model
    model = DogBreedVGG16(num_classes=num_classes, pretrained=False).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    
    print(f"✓ Model loaded: {num_classes} classes")
    print(f"✓ Test set: {len(test_dataset)} samples")
    
    # 2. Get predictions
    print("\n2. Running inference on test set...")
    all_labels = []
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_dataloader, desc="Predicting"):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    print(f"✓ Predictions completed")
    
    # 3. Overall accuracy
    print_section("3. OVERALL PERFORMANCE")
    accuracy = (all_labels == all_preds).mean()
    print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Total samples: {len(all_labels)}")
    print(f"Correct predictions: {(all_labels == all_preds).sum()}")
    print(f"Wrong predictions: {(all_labels != all_preds).sum()}")
    
    # 4. Confusion Matrix
    print_section("4. CONFUSION MATRIX ANALYSIS")
    cm = confusion_matrix(all_labels, all_preds)
    
    # Find most confused pairs
    print("\nTop 10 Most Confused Pairs:")
    confused_pairs = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and cm[i, j] > 0:
                confused_pairs.append({
                    'true': idx_to_class[i],
                    'pred': idx_to_class[j],
                    'count': cm[i, j]
                })
    
    confused_pairs.sort(key=lambda x: x['count'], reverse=True)
    
    for idx, pair in enumerate(confused_pairs[:10], 1):
        print(f"{idx:2d}. {pair['true']:25s} → {pair['pred']:25s} : {pair['count']:3d} times")
    
    # 5. Per-class error analysis
    print_section("5. PER-CLASS ERROR ANALYSIS")
    
    class_errors = []
    for i in range(num_classes):
        true_count = (all_labels == i).sum()
        if true_count == 0:
            continue
            
        correct = cm[i, i]
        total = true_count
        accuracy_class = correct / total if total > 0 else 0
        
        # Find most common misclassification
        misclass_idx = np.argmax([cm[i, j] if j != i else 0 for j in range(num_classes)])
        misclass_count = cm[i, misclass_idx]
        
        class_errors.append({
            'class': idx_to_class[i],
            'total': int(total),
            'correct': int(correct),
            'wrong': int(total - correct),
            'accuracy': accuracy_class,
            'most_confused_with': idx_to_class[misclass_idx],
            'confusion_count': int(misclass_count)
        })
    
    # Sort by accuracy
    class_errors.sort(key=lambda x: x['accuracy'])
    
    print("\nWorst 10 Classes (Lowest Accuracy):")
    for idx, err in enumerate(class_errors[:10], 1):
        print(f"{idx:2d}. {err['class']:25s} - Acc: {err['accuracy']:.2%} ({err['correct']}/{err['total']}) "
              f"- Most confused with: {err['most_confused_with']} ({err['confusion_count']} times)")
    
    print("\nBest 10 Classes (Highest Accuracy):")
    for idx, err in enumerate(class_errors[-10:], 1):
        print(f"{idx:2d}. {err['class']:25s} - Acc: {err['accuracy']:.2%} ({err['correct']}/{err['total']})")
    
    # 6. Confidence analysis
    print_section("6. CONFIDENCE ANALYSIS")
    
    correct_mask = (all_labels == all_preds)
    correct_probs = all_probs[correct_mask, all_preds[correct_mask]]
    wrong_probs = all_probs[~correct_mask, all_preds[~correct_mask]]
    
    print(f"\nCorrect Predictions:")
    print(f"  Mean confidence: {correct_probs.mean():.4f}")
    print(f"  Median confidence: {np.median(correct_probs):.4f}")
    print(f"  Min confidence: {correct_probs.min():.4f}")
    print(f"  Max confidence: {correct_probs.max():.4f}")
    
    print(f"\nWrong Predictions:")
    print(f"  Mean confidence: {wrong_probs.mean():.4f}")
    print(f"  Median confidence: {np.median(wrong_probs):.4f}")
    print(f"  Min confidence: {wrong_probs.min():.4f}")
    print(f"  Max confidence: {wrong_probs.max():.4f}")
    
    # 7. Visualizations
    print_section("7. GENERATING VISUALIZATIONS")
    
    output_dir = "training_models/error_analysis"
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 7.1 Confusion Matrix Heatmap (top 20 classes)
    print("\nGenerating confusion matrix heatmap...")
    # Get top 20 most frequent classes
    class_counts = [(i, (all_labels == i).sum()) for i in range(num_classes)]
    class_counts.sort(key=lambda x: x[1], reverse=True)
    top_20_indices = [x[0] for x in class_counts[:20]]
    
    cm_top20 = cm[np.ix_(top_20_indices, top_20_indices)]
    top_20_names = [idx_to_class[i] for i in top_20_indices]
    
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm_top20, annot=True, fmt='d', cmap='Blues', 
                xticklabels=top_20_names, yticklabels=top_20_names,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix - Top 20 Most Frequent Classes', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/confusion_matrix_top20.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir}/confusion_matrix_top20.png")
    
    # 7.2 Per-class accuracy bar chart
    print("Generating per-class accuracy chart...")
    class_errors.sort(key=lambda x: x['accuracy'])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Worst 15
    worst_15 = class_errors[:15]
    ax1.barh([x['class'] for x in worst_15], [x['accuracy']*100 for x in worst_15], color='salmon')
    ax1.set_xlabel('Accuracy (%)', fontsize=12)
    ax1.set_title('15 Worst Performing Classes', fontsize=14, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    # Best 15
    best_15 = class_errors[-15:]
    ax2.barh([x['class'] for x in best_15], [x['accuracy']*100 for x in best_15], color='skyblue')
    ax2.set_xlabel('Accuracy (%)', fontsize=12)
    ax2.set_title('15 Best Performing Classes', fontsize=14, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/per_class_accuracy.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir}/per_class_accuracy.png")
    
    # 7.3 Confidence distribution
    print("Generating confidence distribution...")
    plt.figure(figsize=(12, 6))
    
    plt.hist(correct_probs, bins=50, alpha=0.7, label='Correct Predictions', color='green', edgecolor='black')
    plt.hist(wrong_probs, bins=50, alpha=0.7, label='Wrong Predictions', color='red', edgecolor='black')
    
    plt.xlabel('Prediction Confidence', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Confidence Distribution: Correct vs Wrong Predictions', fontsize=14, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/confidence_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir}/confidence_distribution.png")
    
    # 8. Save detailed report
    print_section("8. SAVING DETAILED REPORT")
    
    # Save class errors to CSV
    df_errors = pd.DataFrame(class_errors)
    df_errors.to_csv(f"{output_dir}/per_class_errors.csv", index=False)
    print(f"✓ Saved: {output_dir}/per_class_errors.csv")
    
    # Save confused pairs to CSV
    df_confused = pd.DataFrame(confused_pairs[:50])  # Top 50
    df_confused.to_csv(f"{output_dir}/confused_pairs.csv", index=False)
    print(f"✓ Saved: {output_dir}/confused_pairs.csv")
    
    print_section("ERROR ANALYSIS COMPLETED")
    print(f"\nAll results saved to: {output_dir}/")
    print("Generated files:")
    print(f"  - confusion_matrix_top20.png")
    print(f"  - per_class_accuracy.png")
    print(f"  - confidence_distribution.png")
    print(f"  - per_class_errors.csv")
    print(f"  - confused_pairs.csv")

if __name__ == "__main__":
    # Setup logger
    output_dir = "training_models/error_analysis"
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    log_file = f"{output_dir}/error_analysis_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    logger = Logger(log_file)
    sys.stdout = logger
    
    print(f"Error Analysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Log file: {log_file}\n")
    
    analyze_errors()
    
    print(f"\nError Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Log saved to: {log_file}")
    
    # Đóng logger
    sys.stdout = logger.terminal
    logger.close()
