"""
Debug script - Kiểm tra data có đúng không
"""

import os
import pandas as pd
from PIL import Image
import torch
from dataset import DogBreedTrainValDataset
from transforms import get_train_transform

print("="*80)
print("  DEBUG DATA")
print("="*80)

# 1. Check labels.csv
print("\n1. Checking labels.csv...")
if not os.path.exists('labels.csv'):
    print("❌ labels.csv NOT FOUND!")
    exit(1)

df = pd.read_csv('labels.csv')
print(f"✓ labels.csv loaded")
print(f"  Total samples: {len(df)}")
print(f"  Columns: {df.columns.tolist()}")
print(f"  First few rows:")
print(df.head())

# 2. Check train directory
print("\n2. Checking train directory...")
if not os.path.exists('train'):
    print("❌ train/ directory NOT FOUND!")
    exit(1)

train_files = os.listdir('train')
print(f"✓ train/ directory found")
print(f"  Number of files: {len(train_files)}")
print(f"  First few files: {train_files[:5]}")

# 3. Check if images exist
print("\n3. Checking if images exist...")
missing_count = 0
for idx, row in df.head(100).iterrows():  # Check first 100
    img_path = os.path.join('train', row['id'] + '.jpg')
    if not os.path.exists(img_path):
        missing_count += 1
        if missing_count <= 5:
            print(f"  ❌ Missing: {img_path}")

if missing_count == 0:
    print(f"  ✅ All checked images exist")
else:
    print(f"  ❌ {missing_count}/100 images missing!")

# 4. Check image loading
print("\n4. Testing image loading...")
sample_row = df.iloc[0]
img_path = os.path.join('train', sample_row['id'] + '.jpg')

try:
    img = Image.open(img_path).convert('RGB')
    print(f"✓ Image loaded successfully")
    print(f"  Path: {img_path}")
    print(f"  Size: {img.size}")
    print(f"  Mode: {img.mode}")
    print(f"  Breed: {sample_row['breed']}")
except Exception as e:
    print(f"❌ Failed to load image: {e}")

# 5. Check dataset class
print("\n5. Testing Dataset class...")
transform = get_train_transform(input_size=224, augmentation=False)
dataset = DogBreedTrainValDataset(
    image_dir='train',
    dataframe=df.head(100),
    transform=transform
)

print(f"✓ Dataset created")
print(f"  Length: {len(dataset)}")
print(f"  Number of classes: {len(dataset.class_to_idx)}")
print(f"  First few classes: {list(dataset.class_to_idx.items())[:5]}")

# 6. Test __getitem__
print("\n6. Testing dataset __getitem__...")
try:
    image, label = dataset[0]
    print(f"✓ __getitem__ successful")
    print(f"  Image shape: {image.shape}")
    print(f"  Image dtype: {image.dtype}")
    print(f"  Image min/max: {image.min():.3f} / {image.max():.3f}")
    print(f"  Label: {label}")
    print(f"  Label type: {type(label)}")
    
    # Check if image is normalized
    if image.min() < -2 or image.max() > 2:
        print("  ⚠️ Image might not be normalized correctly!")
    else:
        print("  ✅ Image appears normalized")
        
except Exception as e:
    print(f"❌ __getitem__ FAILED: {e}")
    import traceback
    traceback.print_exc()

# 7. Test DataLoader
print("\n7. Testing DataLoader...")
from torch.utils.data import DataLoader

try:
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    images, labels = next(iter(dataloader))
    
    print(f"✓ DataLoader successful")
    print(f"  Batch images shape: {images.shape}")
    print(f"  Batch labels shape: {labels.shape}")
    print(f"  Labels: {labels}")
    print(f"  Unique labels in batch: {torch.unique(labels)}")
    
    # Check if all labels are same
    if len(torch.unique(labels)) == 1:
        print("  ⚠️ All labels in batch are SAME!")
    else:
        print("  ✅ Labels are diverse")
        
except Exception as e:
    print(f"❌ DataLoader FAILED: {e}")
    import traceback
    traceback.print_exc()

# 8. Check class distribution
print("\n8. Checking class distribution...")
breed_counts = df['breed'].value_counts()
print(f"  Most common breed: {breed_counts.index[0]} ({breed_counts.iloc[0]} samples)")
print(f"  Least common breed: {breed_counts.index[-1]} ({breed_counts.iloc[-1]} samples)")
print(f"  Imbalance ratio: {breed_counts.iloc[0] / breed_counts.iloc[-1]:.2f}")

print("\n" + "="*80)
print("  DEBUG COMPLETED")
print("="*80)
print("\nIf all checks passed, the problem is likely in:")
print("  1. Model architecture")
print("  2. Training loop")
print("  3. Optimizer/Loss configuration")
