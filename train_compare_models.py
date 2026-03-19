import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torchvision import models
from torch.utils.data import DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from dataset import DogBreedTrainValDataset
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from tqdm import tqdm

def extract_features(dataloader, model, device):
    """Pass images through the CNN to extract the 512-dimensional feature vectors."""
    features = []
    labels_list = []
    
    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Extracting Features"):
            images = images.to(device)
            # Forward pass through backbone only
            outputs = model(images)
            features.append(outputs.cpu().numpy())
            labels_list.append(labels.numpy())
            
    return np.vstack(features), np.concatenate(labels_list)

def main():
    print("="*60)
    print("DATA MINING BONUS: COMPARING ALGORITHMS ON EXTRACTED FEATURES")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Configuration
    labels_path = "labels.csv"
    train_dir = "train"
    batch_size = 64
    
    transform = Compose([
        Resize(256),
        CenterCrop(224),
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    print("\n1. Loading Data...")
    df = pd.read_csv(labels_path)
    
    # We only use a subset of the dataset here to make the ML models train faster 
    # for the demonstration of "Comparison" in Data Mining.
    # Take 20% of the dataset for this demonstration.
    try:
        df_subset, _ = train_test_split(df, train_size=0.2, random_state=42, stratify=df["breed"])
    except:
        df_subset = df.sample(frac=0.2, random_state=42)
        
    train_df, val_df = train_test_split(df_subset, test_size=0.2, random_state=42, stratify=df_subset["breed"])
    
    train_dataset = DogBreedTrainValDataset(image_dir=train_dir, dataframe=train_df, transform=transform)
    val_dataset = DogBreedTrainValDataset(image_dir=train_dir, dataframe=val_df, transform=transform, class_to_idx=train_dataset.class_to_idx)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    print("\n2. Initializing ResNet Feature Extractor...")
    # Load ResNet 18 and remove the final classification layer
    # This acts as our "Feature Engineering / Preprocessing" step in Data Mining
    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # Output of resnet.fc.in_features is 512
    # We strip the fc layer to return the 512-dim vector
    resnet.fc = nn.Identity() 
    resnet = resnet.to(device)
    
    print("Extracting training features...")
    X_train, y_train = extract_features(train_loader, resnet, device)
    print(f"Training features shape: {X_train.shape}")
    
    print("Extracting validation features...")
    X_val, y_val = extract_features(val_loader, resnet, device)
    print(f"Validation features shape: {X_val.shape}")
    
    print("\n[+] FEATURE ENGINEERING: SAVING TABULAR DATASET")
    # Save a small sample (e.g., first 500 rows) to CSV for demonstration in the report
    feature_cols = [f"feature_{i}" for i in range(X_train.shape[1])]
    df_features = pd.DataFrame(X_train[:500], columns=feature_cols)
    df_features['target_breed_idx'] = y_train[:500]
    df_features.to_csv("extracted_tabular_features.csv", index=False)
    print("✅ Saved 'extracted_tabular_features.csv'! This proves we transformed Unstructured Images into Structured Tabular Data!")
    
    print("\n3. Training Classical Machine Learning Models...")
    
    # Model 1: Random Forest Classifier
    print("-> Training Random Forest Classifier...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_val)
    
    # Model 2: Support Vector Machine (LinearSVC usually faster than SVC for many classes)
    print("-> Training Support Vector Machine (LinearSVC)...")
    svm_model = LinearSVC(C=1.0, random_state=42, max_iter=2000, dual=False)
    svm_model.fit(X_train, y_train)
    svm_preds = svm_model.predict(X_val)
    
    print("\n" + "="*60)
    print("4. EVALUATION & ALGORITHM COMPARISON RESULTS")
    print("="*60)
    
    def print_metrics(model_name, y_true, y_pred):
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        print(f"[{model_name}]")
        print(f"  Accuracy : {acc*100:.2f}%")
        print(f"  Precision: {prec*100:.2f}%")
        print(f"  Recall   : {rec*100:.2f}%")
        print(f"  F1-Score : {f1*100:.2f}%")
        print("-" * 30)

    print_metrics("Random Forest (100 Trees)", y_val, rf_preds)
    print_metrics("Linear Support Vector Machine", y_val, svm_preds)
    
    print("NOTE FOR REPORT:")
    print("Copy these results to your Data Mining report to fullfil the 'Compare Algorithms' Bonus points requirement.")
    print("="*60)

if __name__ == "__main__":
    main()
