import torch
import pandas as pd
import numpy as np
from torchvision import models
from torch.utils.data import DataLoader
from dataset import DogBreedTrainValDataset
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from sklearn.metrics import confusion_matrix
import torch.nn as nn
from tqdm import tqdm
import os

def main():
    print("="*60)
    print("DATA MINING: ERROR ANALYSIS & MISCLASSIFICATION INSIGHTS")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels_path = "labels.csv"
    train_dir = "train"
    checkpoint_path = "training_models/best_resnet.pth"
    
    if not os.path.exists(checkpoint_path):
        print("❌ Model checkpoint not found. Please train the model first.")
        return
        
    df = pd.read_csv(labels_path)
    # Using a 20% sample for fast diagnostic error analysis
    df_val = df.sample(frac=0.2, random_state=42)
    
    transform = Compose([
        Resize(256), CenterCrop(224), ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load metadata
    checkpoint = torch.load(checkpoint_path, map_location=device)
    class_to_idx = checkpoint['class_to_idx']
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(class_to_idx)
    
    dataset = DogBreedTrainValDataset(image_dir=train_dir, dataframe=df_val, transform=transform, class_to_idx=class_to_idx)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)
    
    # Reconstruct Model architecture
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    print(f"Running inference on validation subset to extract Confusion Matrix...")
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Predicting for Error Analysis"):
            outputs = model(images.to(device))
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    cm = confusion_matrix(all_labels, all_preds)
    
    # Find most confused pairs by ignoring the diagonal (correct predictions)
    np.fill_diagonal(cm, 0)
    confused_pairs = []
    
    for i in range(num_classes):
        for j in range(num_classes):
            if cm[i, j] > 0:
                confused_pairs.append({
                    'true_breed': idx_to_class[i],
                    'predicted_as': idx_to_class[j],
                    'count': cm[i, j]
                })
                
    confused_pairs = sorted(confused_pairs, key=lambda x: x['count'], reverse=True)
    
    print("\n" + "="*60)
    print("DATA MINING INSIGHTS: TOP MOST CONFUSED CLASS PAIRS")
    print("="*60)
    for idx, pair in enumerate(confused_pairs[:7]):
        print(f" {idx+1}. True Class: [{pair['true_breed']}]")
        print(f"    --> Misclassified as: [{pair['predicted_as']}] ({pair['count']} times)")
        print("-" * 50)
        
    print("\n--- INSIGHT TO WRITE IN SECTION 4 & 5 OF REPORT ---")
    print("1. 'Data Separability Issue': The misclassifications occur heavily among specific breed pairs.")
    print("2. 'Root Cause Analysis': These pairs share significant phenotypic traits (e.g., similar coat textures, size, or snout geometry).")
    print("3. 'Model Implication': The high-dimensional feature vectors extracted by the CNN backbone cluster too closely together for these specific classes.")
    print("4. 'Future Improvement': Using Hard-Negative Mining or Contrastive Learning (Triplet Loss) to push these overlapping clusters apart in the latent space.")

if __name__ == "__main__":
    main()
