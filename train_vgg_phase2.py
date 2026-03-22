import os
import json
import torch
import torch.nn as nn
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.model_selection import train_test_split

from dataset import DogBreedTrainValDataset
from model_vgg import DogBreedVGG16
from transforms import get_train_transform, get_val_transform

def print_section(title):
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_section("PHASE 2: DEEP FINE-TUNING (UNFREEZE BLOCK 5)")
    
    # 1. SETUP DATA (Giữ nguyên như cũ)
    df = pd.read_csv("labels.csv")
    train_transform = get_train_transform(input_size=224, augmentation=True)
    val_transform = get_val_transform(input_size=224)
    
    train_val_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["breed"])
    train_df, val_df = train_test_split(train_val_df, test_size=0.2, random_state=42, stratify=train_val_df["breed"])
    
    train_dataset = DogBreedTrainValDataset("train", train_df, train_transform)
    val_dataset = DogBreedTrainValDataset("train", val_df, val_transform, train_dataset.class_to_idx)
    test_dataset = DogBreedTrainValDataset("train", test_df, val_transform, train_dataset.class_to_idx)
    
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

    # 2. LOAD MODEL TỪ PHASE 1
    num_classes = len(train_dataset.class_to_idx)
    model = DogBreedVGG16(num_classes=num_classes, pretrained=False, dropout=0.3).to(device)
    
    print("✓ Loading best model from Phase 1...")
    checkpoint = torch.load("training_models/best_vgg.pth", map_location=device)
    model.load_state_dict(checkpoint["model"])
    best_acc = checkpoint["best_acc"]
    print(f"  Starting with Phase 1 Best Val Acc: {best_acc:.4f}")

    # 3. UNFREEZE BLOCK 5 & CLASSIFIER
    # Đóng băng toàn bộ trước cho chắc chắn
    for param in model.parameters():
        param.requires_grad = False
        
    # Mở khóa Classifier (để nó tiếp tục tinh chỉnh cùng Block 5)
    for param in model.backbone.classifier.parameters():
        param.requires_grad = True
        
    # Mở khóa Block 5 của VGG16 (các layers từ 24 đến cuối của features)
    for param in model.backbone.features[24:].parameters():
        param.requires_grad = True

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Trainable parameters (Block 5 + Classifier): {trainable_params:,}")

    # 4. TRAINING SETUP (LEARNING RATE CỰC NHỎ)
    num_epochs = 15
    learning_rate = 1e-5  # Giảm 10 lần so với Phase 1 (1e-4 -> 1e-5)
    weight_decay = 1e-4
    
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    
    patience_counter = 0
    patience_limit = 4

    # 5. TRAINING LOOP
    print_section("STARTING PHASE 2 TRAINING")
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
        
        checkpoint_data = {
            "epoch": epoch + 1, "model": model.state_dict(),
            "best_acc": max(best_acc, val_acc), "class_to_idx": train_dataset.class_to_idx 
        }
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(checkpoint_data, "training_models/best_vgg_phase2.pth")
            print(f"✓ Epoch {epoch+1} - Val Acc: {val_acc:.4f} ⭐ NEW BEST!")
            patience_counter = 0
        else:
            print(f"  Epoch {epoch+1} - Val Acc: {val_acc:.4f}")
            patience_counter += 1
            if patience_counter >= patience_limit:
                print(f"⚠️ Early stopping Phase 2 at epoch {epoch+1}!")
                break

    # 6. FINAL TEST SET EVALUATION
    print_section("FINAL EVALUATION ON TEST SET")
    
    best_model_path = "training_models/best_vgg_phase2.pth" if os.path.exists("training_models/best_vgg_phase2.pth") else "training_models/best_vgg.pth"
    model.load_state_dict(torch.load(best_model_path, map_location=device)["model"])
    model.eval()
    
    all_test_labels, all_test_preds = [], []
    with torch.no_grad():
        for images, labels in tqdm(test_dataloader, colour="blue", desc="Testing"):
            images, labels = images.to(device), labels.to(device)
            preds = torch.argmax(model(images), dim=1)
            all_test_labels.extend(labels.cpu().numpy())
            all_test_preds.extend(preds.cpu().numpy())
            
    test_acc = accuracy_score(all_test_labels, all_test_preds)
    _, _, test_f1, _ = precision_recall_fscore_support(all_test_labels, all_test_preds, average='weighted', zero_division=0)
    
    print("\n" + "="*80)
    print("  FINAL TEST SET RESULTS (PHASE 2)")
    print("="*80)
    print(f"Accuracy:  {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"F1-Score:  {test_f1:.4f}")