import os
import json
import shutil
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from datetime import datetime

from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from dataset import DogBreedTrainValDataset
from model_vgg import DogBreedVGG16
from transforms import get_train_transform, get_val_transform

def print_section(title):
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)

def save_metrics(metrics, filepath):
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"✓ Metrics saved to {filepath}")

if __name__ == "__main__":
    # ========================================================================
    # STEP 1: DATASET LOADING
    # ========================================================================
    print_section("STEP 1: DATASET LOADING")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    labels_path = "labels.csv"
    train_dir = "train"
    
    df = pd.read_csv(labels_path)
    print(f"✓ Total samples: {len(df)}")
    print(f"✓ Number of breeds: {df['breed'].nunique()}")
    
    # ========================================================================
    # STEP 2 & 3: PREPROCESSING & SPLIT
    # ========================================================================
    print_section("STEP 2 & 3: PREPROCESSING & DATA SPLIT")
    
    train_transform = get_train_transform(input_size=224, augmentation=True)
    val_transform = get_val_transform(input_size=224)
    
    # First split: 80% train+val, 20% test
    train_val_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["breed"])
    
    # Second split: 80% train, 20% val
    train_df, val_df = train_test_split(train_val_df, test_size=0.2, random_state=42, stratify=train_val_df["breed"])
    
    print(f"✓ Train set: {len(train_df)} samples")
    print(f"✓ Val set: {len(val_df)} samples")
    print(f"✓ Test set: {len(test_df)} samples")
    
    train_dataset = DogBreedTrainValDataset(train_dir, train_df, train_transform)
    val_dataset = DogBreedTrainValDataset(train_dir, val_df, val_transform, train_dataset.class_to_idx)
    test_dataset = DogBreedTrainValDataset(train_dir, test_df, val_transform, train_dataset.class_to_idx)
    
    # ========================================================================
    # STEP 4: FEATURE EXTRACTION (VGG16)
    # ========================================================================
    print_section("STEP 4: MODEL SETUP (VGG16)")
    
    num_classes = len(train_dataset.class_to_idx)
    model = DogBreedVGG16(num_classes=num_classes, pretrained=True, dropout=0.3).to(device)
    
    # ĐÓNG BĂNG (FREEZE) PHẦN FEATURES (CNN Layers)
    # Giúp model train nhanh hơn và không làm hỏng trọng số ImageNet
    for param in model.backbone.features.parameters():
        param.requires_grad = False
        
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✓ Total parameters: {total_params:,}")
    print(f"✓ Trainable parameters (Classifier only): {trainable_params:,}")
    
    # ========================================================================
    # STEP 5: TRAINING
    # ========================================================================
    print_section("STEP 5: TRAINING")
    
    num_epochs = 20
    batch_size = 32 # Tăng batch_size lên vì chỉ train classifier sẽ ít tốn VRAM hơn
    learning_rate = 1e-4 # Giảm LR xuống mức an toàn cho Fine-Tuning
    weight_decay = 1e-4
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    if os.path.isdir("tensorboard_vgg"):
        shutil.rmtree("tensorboard_vgg")
    if not os.path.isdir("training_models"):
        os.mkdir("training_models")
        
    writer = SummaryWriter("tensorboard_vgg")
    
    # Chỉ truyền các parameters có requires_grad=True vào Optimizer
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    best_acc = 0
    patience_counter = 0
    patience_limit = 5
    
    print("\nStarting training...")
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        progress_bar = tqdm(train_dataloader, colour="green")
        
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            progress_bar.set_description(f"Epoch {epoch+1}/{num_epochs} | Loss: {loss.item():.4f}")
            
        avg_train_loss = train_loss / len(train_dataloader)
        
        # Validation
        model.eval()
        all_labels, all_preds = [], []
        
        with torch.no_grad():
            for images, labels in val_dataloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                preds = torch.argmax(outputs, dim=1)
                
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                
        val_acc = accuracy_score(all_labels, all_preds)
        scheduler.step(val_acc)
        
        writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        writer.add_scalar("Accuracy/Validation", val_acc, epoch)
        
        # Lưu checkpoint
        checkpoint = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "best_acc": max(best_acc, val_acc),
            "class_to_idx": train_dataset.class_to_idx 
        }
        torch.save(checkpoint, "training_models/last_vgg.pth")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(checkpoint, "training_models/best_vgg.pth")
            print(f"✓ Epoch {epoch+1} - Val Acc: {val_acc:.4f} ⭐ NEW BEST!")
            patience_counter = 0
        else:
            print(f"  Epoch {epoch+1} - Val Acc: {val_acc:.4f}")
            patience_counter += 1
            if patience_counter >= patience_limit:
                print(f"⚠️ Early stopping at epoch {epoch+1}!")
                break
                
    writer.close()

    # ========================================================================
    # STEP 6: EVALUATION
    # ========================================================================
    print_section("STEP 6: EVALUATION ON TEST SET")
    
    checkpoint = torch.load("training_models/best_vgg.pth", map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    
    all_test_labels, all_test_preds = [], []
    
    with torch.no_grad():
        for images, labels in tqdm(test_dataloader, colour="blue", desc="Testing"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            
            all_test_labels.extend(labels.cpu().numpy())
            all_test_preds.extend(preds.cpu().numpy())
            
    test_acc = accuracy_score(all_test_labels, all_test_preds)
    test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(
        all_test_labels, all_test_preds, average='weighted', zero_division=0
    )
    
    print("\n" + "="*80)
    print("  TEST SET RESULTS (VGG16)")
    print("="*80)
    print(f"Accuracy:  {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"F1-Score:  {test_f1:.4f}")
    
    metrics = {
        "accuracy": float(test_acc), "f1": float(test_f1),
        "precision": float(test_precision), "recall": float(test_recall)
    }
    save_metrics(metrics, "training_models/vgg_test_metrics.json")