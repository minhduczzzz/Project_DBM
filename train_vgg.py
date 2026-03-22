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
    """Print formatted section header"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)


def save_metrics(metrics, filepath):
    """Save metrics to JSON file"""
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
    print(f"✓ Samples per breed (min/max): {df['breed'].value_counts().min()} / {df['breed'].value_counts().max()}")
    
    # ========================================================================
    # STEP 2: PREPROCESSING + AUGMENTATION
    # ========================================================================
    print_section("STEP 2: PREPROCESSING + AUGMENTATION")
    
    # Sử dụng transform factory để tạo transforms
    # Lợi ích: Tái sử dụng code, dễ maintain, consistent across models
    
    train_transform = get_train_transform(input_size=224, augmentation=True)
    val_transform = get_val_transform(input_size=224)
    
    print("✓ Training transforms:")
    print("  [PREPROCESSING - BẮT BUỘC]")
    print("    - Resize to 224x224")
    print("    - ToTensor")
    print("    - Normalization (ImageNet stats)")
    print("  [AUGMENTATION - Recommended]")
    print("    - Random Horizontal Flip (p=0.5)")
    print("    - Random Rotation (±15°)")
    print("    - Color Jitter (brightness, contrast, saturation)")
    
    print("\n✓ Validation transforms:")
    print("  [PREPROCESSING - BẮT BUỘC]")
    print("    - Resize to 224x224")
    print("    - ToTensor")
    print("    - Normalization (ImageNet stats)")
    print("  [AUGMENTATION]")
    print("    - None (validation không cần augmentation)")
    
    # ========================================================================
    # STEP 3: TRAIN / VAL / TEST SPLIT
    # ========================================================================
    print_section("STEP 3: TRAIN / VAL / TEST SPLIT")
    
    # First split: 80% train+val, 20% test
    train_val_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["breed"]
    )
    
    # Second split: 80% train, 20% val (from train+val)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=0.2,
        random_state=42,
        stratify=train_val_df["breed"]
    )
    
    print(f"✓ Train set: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
    print(f"✓ Validation set: {len(val_df)} samples ({len(val_df)/len(df)*100:.1f}%)")
    print(f"✓ Test set: {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)")
    
    train_dataset = DogBreedTrainValDataset(
        image_dir=train_dir,
        dataframe=train_df,
        transform=train_transform
    )
    
    val_dataset = DogBreedTrainValDataset(
        image_dir=train_dir,
        dataframe=val_df,
        transform=val_transform,
        class_to_idx=train_dataset.class_to_idx
    )
    
    test_dataset = DogBreedTrainValDataset(
        image_dir=train_dir,
        dataframe=test_df,
        transform=val_transform,
        class_to_idx=train_dataset.class_to_idx
    )
    
    # ========================================================================
    # STEP 4: FEATURE EXTRACTION (VGG16)
    # ========================================================================
    print_section("STEP 4: FEATURE EXTRACTION (VGG16)")
    
    num_classes = len(train_dataset.class_to_idx)
    model = DogBreedVGG16(num_classes=num_classes, pretrained=True).to(device)
    
    print(f"✓ Model: VGG16")
    print(f"✓ Pretrained: ImageNet weights")
    print(f"✓ Number of classes: {num_classes}")
    print(f"✓ Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"✓ Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # ========================================================================
    # STEP 5: TRAINING (FINE-TUNE)
    # ========================================================================
    print_section("STEP 5: TRAINING (FINE-TUNE)")
    
    num_epochs = 30
    batch_size = 16
    learning_rate = 1e-4
    start_epoch = 0
    best_acc = 0
    
    print(f"✓ Epochs: {num_epochs}")
    print(f"✓ Batch size: {batch_size}")
    print(f"✓ Learning rate: {learning_rate}")
    print(f"✓ Optimizer: Adam")
    print(f"✓ Loss function: CrossEntropyLoss")
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    
    if os.path.isdir("tensorboard_vgg"):
        shutil.rmtree("tensorboard_vgg")
    
    if not os.path.isdir("training_models"):
        os.mkdir("training_models")
    
    writer = SummaryWriter("tensorboard_vgg")
    
    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # Load checkpoint if exists
    if os.path.isfile("training_models/last_vgg.pth"):
        print("\n✓ Loading checkpoint...")
        ckpt = torch.load("training_models/last_vgg.pth", map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"]
        best_acc = ckpt.get("best_acc", 0)
        print(f"  Resuming from epoch {start_epoch}, best acc: {best_acc:.4f}")
    
    num_iters = len(train_dataloader)
    training_history = []
    
    print("\n" + "-"*80)
    print("Starting training...")
    print("-"*80)
    
    for epoch in range(start_epoch, num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        progress_bar = tqdm(train_dataloader, colour="green")
        
        for iteration, (images, labels) in enumerate(progress_bar):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss_value = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()
            
            train_loss += loss_value.item()
            
            progress_bar.set_description(
                f"Epoch {epoch+1}/{num_epochs}, Iter {iteration+1}/{num_iters}, Loss {loss_value.item():.4f}"
            )
            
            writer.add_scalar("Train/Loss", loss_value.item(), epoch * num_iters + iteration)
        
        avg_train_loss = train_loss / num_iters
        
        # Validation phase
        model.eval()
        all_labels = []
        all_predictions = []
        
        with torch.no_grad():
            for images, labels in val_dataloader:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                predictions = torch.argmax(outputs, dim=1)
                
                all_labels.extend(labels.cpu().numpy().tolist())
                all_predictions.extend(predictions.cpu().numpy().tolist())
        
        val_accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted', zero_division=0
        )
        
        writer.add_scalar("Validation/Accuracy", val_accuracy, epoch)
        writer.add_scalar("Validation/F1", f1, epoch)
        writer.add_scalar("Train/AvgLoss", avg_train_loss, epoch)
        
        # Save history
        training_history.append({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_accuracy": val_accuracy,
            "val_precision": precision,
            "val_recall": recall,
            "val_f1": f1
        })
        
        # Save checkpoint
        checkpoint = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_acc": best_acc,
            "class_to_idx": train_dataset.class_to_idx
        }
        
        torch.save(checkpoint, "training_models/last_vgg.pth")
        
        if val_accuracy > best_acc:
            best_acc = val_accuracy
            checkpoint["best_acc"] = best_acc
            torch.save(checkpoint, "training_models/best_vgg.pth")
            print(f"✓ Epoch {epoch+1}/{num_epochs} - Val Acc: {val_accuracy:.4f} ⭐ NEW BEST!")
        else:
            print(f"  Epoch {epoch+1}/{num_epochs} - Val Acc: {val_accuracy:.4f}")
    
    writer.close()
    
    # Save training history
    save_metrics(training_history, "training_models/vgg_training_history.json")
    
    # ========================================================================
    # STEP 6: EVALUATION (ACC, F1, PRECISION, RECALL)
    # ========================================================================
    print_section("STEP 6: EVALUATION ON TEST SET")
    
    # Load best model
    print("✓ Loading best model...")
    checkpoint = torch.load("training_models/best_vgg.pth", map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    
    all_test_labels = []
    all_test_predictions = []
    
    print("✓ Running inference on test set...")
    with torch.no_grad():
        for images, labels in tqdm(test_dataloader, colour="blue"):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)
            
            all_test_labels.extend(labels.cpu().numpy().tolist())
            all_test_predictions.extend(predictions.cpu().numpy().tolist())
    
    # Calculate metrics
    test_accuracy = accuracy_score(all_test_labels, all_test_predictions)
    test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(
        all_test_labels, all_test_predictions, average='weighted', zero_division=0
    )
    
    print("\n" + "="*80)
    print("  TEST SET RESULTS (VGG16)")
    print("="*80)
    print(f"Accuracy:  {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall:    {test_recall:.4f}")
    print(f"F1-Score:  {test_f1:.4f}")
    print("="*80)
    
    # Save test metrics
    test_metrics = {
        "model": "VGG16",
        "test_accuracy": float(test_accuracy),
        "test_precision": float(test_precision),
        "test_recall": float(test_recall),
        "test_f1": float(test_f1),
        "num_classes": num_classes,
        "test_samples": len(test_df),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    save_metrics(test_metrics, "training_models/vgg_test_metrics.json")
    
    # Classification report
    print("\n" + "-"*80)
    print("DETAILED CLASSIFICATION REPORT")
    print("-"*80)
    class_names = [train_dataset.idx_to_class[i] for i in range(num_classes)]
    print(classification_report(
        all_test_labels, 
        all_test_predictions,
        target_names=class_names,
        zero_division=0
    ))
    
    # Confusion matrix stats
    cm = confusion_matrix(all_test_labels, all_test_predictions)
    print("\n" + "-"*80)
    print("CONFUSION MATRIX STATISTICS")
    print("-"*80)
    print(f"Correctly classified: {np.trace(cm)} / {len(all_test_labels)}")
    print(f"Misclassified: {len(all_test_labels) - np.trace(cm)} / {len(all_test_labels)}")
    
    print("\n" + "="*80)
    print("  TRAINING COMPLETED!")
    print("="*80)
    print(f"✓ Best validation accuracy: {best_acc:.4f}")
    print(f"✓ Test accuracy: {test_accuracy:.4f}")
    print(f"✓ Model saved: training_models/best_vgg.pth")
    print(f"✓ Metrics saved: training_models/vgg_test_metrics.json")
    print("="*80)
