import os
import shutil
import torch
import torch.nn as nn
import pandas as pd

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import CenterCrop, Compose, RandomErasing, RandomResizedCrop, Resize, RandomHorizontalFlip, RandomRotation, ColorJitter, ToTensor, Normalize
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from dataset import DogBreedTrainValDataset
from model import DogBreedResNet

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_epochs = 50
    batch_size = 32
    start_epoch = 0
    best_acc = 0
    patience = 10

    labels_path = "labels.csv"
    train_dir = "train"

    # DATA MINING MAPPING: Handling Class Imbalance and Noise via Data Augmentation
    train_transform = Compose([
    RandomResizedCrop(224, scale=(0.8, 1.0)),  # Feature Selection (Focusing on Region of Interest)
    RandomHorizontalFlip(),
    RandomRotation(15),                        # Introducing Noise to prevent Minority Class Overfitting
    ColorJitter(0.3, 0.3, 0.3),
    ToTensor(),
    # DATA MINING MAPPING: Data Standardization / Feature Scaling
    Normalize([0.485, 0.456, 0.406],
              [0.229, 0.224, 0.225]),
    RandomErasing(p=0.25)                      # Outlier robust training simulation
    ])

    val_transform = Compose([
    Resize(256),        #  giữ tỉ lệ
    CenterCrop(224),    #  crop giữa
    ToTensor(),
    Normalize([0.485, 0.456, 0.406],
              [0.229, 0.224, 0.225])
    ])

    df = pd.read_csv(labels_path)

    print("\n[+] DATA MINING PHASE: DATA CLEANING & REDUCTION")
    # 1. Feature Engineering: Dropping exact duplicates
    initial_len = len(df)
    df = df.drop_duplicates(subset=['id']).reset_index(drop=True)
    dup_removed = initial_len - len(df)
    
    # 2. Handling Missing/Corrupted Physical Data
    # Filtering out textual records that have no valid physical image match
    valid_mask = df['id'].apply(lambda x: os.path.exists(os.path.join(train_dir, f"{x}.jpg")))
    df = df[valid_mask].reset_index(drop=True)
    missing_removed = (initial_len - dup_removed) - len(df)
    
    print(f"-> Removed {dup_removed} duplicate records.")
    print(f"-> Removed {missing_removed} missing image links.")
    print(f"-> Final Cleaned Dataset Size: {len(df)} records ready for modeling.\n")

    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["breed"]
    )

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

    if os.path.isdir("tensorboard"):
        shutil.rmtree("tensorboard")

    if not os.path.isdir("training_models"):
        os.mkdir("training_models")

    writer = SummaryWriter("tensorboard")

    num_classes = len(train_dataset.class_to_idx)
    model = DogBreedResNet(num_classes=num_classes, pretrained=True).to(device)

    optimizer = Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.3)
    criterion = nn.CrossEntropyLoss()

    if os.path.isfile("training_models/last_resnet.pth"):
        ckpt = torch.load("training_models/last_resnet.pth", map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"]
        best_acc = ckpt.get("best_acc", 0)

    num_iters = len(train_dataloader)

    for epoch in range(start_epoch, num_epochs):
        model.train()

        progress_bar = tqdm(train_dataloader, colour="green")

        for iteration, (images, labels) in enumerate(progress_bar):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss_value = criterion(outputs, labels)

            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

            preds = torch.argmax(outputs, dim=1)
            train_acc = (preds == labels).float().mean()

            progress_bar.set_description(
                f"Epoch {epoch+1}/{num_epochs}, Iteration {iteration+1}/{num_iters}, Loss {loss_value.item():.4f}, Acc {train_acc:.4f}"
            )

            writer.add_scalar("Train/Loss", loss_value.item(), epoch * num_iters + iteration)
            writer.add_scalar("Train/Accuracy", train_acc.item(), epoch * num_iters + iteration)    

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

        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='weighted', zero_division=0)
        
        writer.add_scalar("Validation/Accuracy", accuracy, epoch)
        writer.add_scalar("Validation/Precision", precision, epoch)
        writer.add_scalar("Validation/Recall", recall, epoch)
        writer.add_scalar("Validation/F1-Score", f1, epoch)

        scheduler.step(accuracy)

        checkpoint = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_acc": best_acc,
            "class_to_idx": train_dataset.class_to_idx
        }

        torch.save(checkpoint, "training_models/last_resnet.pth")

        if accuracy > best_acc:
            best_acc = accuracy
            no_improve_epochs = 0
            checkpoint["best_acc"] = best_acc
            torch.save(checkpoint, "training_models/best_resnet.pth")
            
            # Print detailed Data Mining metrics for the new best model
            print("\n" + "="*60)
            print(f"NEW BEST MODEL FOUND (Epoch {epoch+1}) - METRICS REPORT")
            print("="*60)
            print(classification_report(all_labels, all_predictions, zero_division=0))
            print("\nConfusion Matrix (First 10x10):")
            cm = confusion_matrix(all_labels, all_predictions)
            print(cm[:10, :10], "...")
            print("="*60 + "\n")
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= patience:
            print(f"Early stopping at epoch {epoch+1} due to no improvement in validation accuracy for {patience} consecutive epochs.")
            break    
        print(f"Epoch {epoch+1}/{num_epochs} - Validation Accuracy: {accuracy:.4f}")
        
    writer.close()