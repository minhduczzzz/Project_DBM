import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_metrics(filepath):
    """Load metrics from JSON file"""
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return None


def print_section(title):
    """Print formatted section header"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)


if __name__ == "__main__":
    print_section("MODEL COMPARISON - DOG BREED CLASSIFICATION")
    
    # Define models to compare
    models_info = {
        "ResNet18": "training_models/resnet_test_metrics.json",
        "VGG16": "training_models/vgg_test_metrics.json",
        "AlexNet": "training_models/alexnet_test_metrics.json",
        "EfficientNet-B0": "training_models/efficientnet_test_metrics.json"
    }
    
    # Load all available metrics
    results = []
    for model_name, filepath in models_info.items():
        metrics = load_metrics(filepath)
        if metrics:
            results.append({
                "Model": model_name,
                "Accuracy": metrics.get("test_accuracy", 0),
                "Precision": metrics.get("test_precision", 0),
                "Recall": metrics.get("test_recall", 0),
                "F1-Score": metrics.get("test_f1", 0)
            })
            print(f"✓ Loaded metrics for {model_name}")
        else:
            print(f"✗ Metrics not found for {model_name}")
    
    if not results:
        print("\n⚠ No model metrics found!")
        print("Please train at least one model first.")
        exit(1)
    
    # Create comparison DataFrame
    df_comparison = pd.DataFrame(results)
    df_comparison = df_comparison.sort_values("Accuracy", ascending=False)
    
    # Display comparison table
    print("\n" + "-"*80)
    print("PERFORMANCE COMPARISON")
    print("-"*80)
    print(df_comparison.to_string(index=False))
    print("-"*80)
    
    # Find best model
    best_model = df_comparison.iloc[0]
    print("\n" + "="*80)
    print(f"  🏆 BEST MODEL: {best_model['Model']}")
    print("="*80)
    print(f"Accuracy:  {best_model['Accuracy']:.4f} ({best_model['Accuracy']*100:.2f}%)")
    print(f"Precision: {best_model['Precision']:.4f}")
    print(f"Recall:    {best_model['Recall']:.4f}")
    print(f"F1-Score:  {best_model['F1-Score']:.4f}")
    print("="*80)
    
    # Save comparison to CSV
    df_comparison.to_csv("training_models/model_comparison.csv", index=False)
    print(f"\n✓ Comparison saved to training_models/model_comparison.csv")
    
    # Create visualization
    print("\n✓ Creating comparison charts...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
    
    for idx, (ax, metric, color) in enumerate(zip(axes.flat, metrics, colors)):
        bars = ax.barh(df_comparison['Model'], df_comparison[metric], color=color, alpha=0.7)
        ax.set_xlabel(metric, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2, 
                   f'{width:.4f}', 
                   ha='left', va='center', fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('training_models/model_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Chart saved to training_models/model_comparison.png")
    
    # Create metrics comparison chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = range(len(df_comparison))
    width = 0.2
    
    ax.bar([i - 1.5*width for i in x], df_comparison['Accuracy'], width, label='Accuracy', color='#2ecc71', alpha=0.8)
    ax.bar([i - 0.5*width for i in x], df_comparison['Precision'], width, label='Precision', color='#3498db', alpha=0.8)
    ax.bar([i + 0.5*width for i in x], df_comparison['Recall'], width, label='Recall', color='#e74c3c', alpha=0.8)
    ax.bar([i + 1.5*width for i in x], df_comparison['F1-Score'], width, label='F1-Score', color='#f39c12', alpha=0.8)
    
    ax.set_xlabel('Models', fontweight='bold', fontsize=12)
    ax.set_ylabel('Score', fontweight='bold', fontsize=12)
    ax.set_title('All Metrics Comparison', fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(df_comparison['Model'], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('training_models/metrics_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Chart saved to training_models/metrics_comparison.png")
    
    print("\n" + "="*80)
    print("  COMPARISON COMPLETED!")
    print("="*80)
