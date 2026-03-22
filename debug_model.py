"""
Debug script - Kiểm tra model có hoạt động đúng không
"""

import torch
import torch.nn as nn
from model_vgg import DogBreedVGG16
from torchvision import models

print("="*80)
print("  DEBUG MODEL")
print("="*80)

# 1. Check VGG16 pretrained
print("\n1. Checking VGG16 pretrained...")
vgg_original = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
print(f"✓ VGG16 loaded")
print(f"  Original classifier: {vgg_original.classifier}")

# 2. Check custom model
print("\n2. Checking custom DogBreedVGG16...")
model = DogBreedVGG16(num_classes=120, pretrained=True, dropout=0.3)
print(f"✓ Custom model loaded")
print(f"  Custom classifier: {model.backbone.classifier}")

# 3. Test forward pass
print("\n3. Testing forward pass...")
dummy_input = torch.randn(2, 3, 224, 224)  # Batch of 2 images
try:
    output = model(dummy_input)
    print(f"✓ Forward pass successful")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Expected: (2, 120)")
    
    if output.shape == (2, 120):
        print("  ✅ Output shape correct!")
    else:
        print("  ❌ Output shape WRONG!")
        
except Exception as e:
    print(f"❌ Forward pass FAILED: {e}")

# 4. Check if weights are frozen
print("\n4. Checking trainable parameters...")
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Total parameters: {total_params:,}")
print(f"  Trainable parameters: {trainable_params:,}")

if trainable_params == 0:
    print("  ❌ NO TRAINABLE PARAMETERS! Weights are frozen!")
elif trainable_params == total_params:
    print("  ✅ All parameters trainable")
else:
    print(f"  ⚠️ Only {trainable_params/total_params*100:.1f}% parameters trainable")

# 5. Check features vs classifier
print("\n5. Checking features vs classifier...")
features_params = sum(p.numel() for p in model.backbone.features.parameters())
classifier_params = sum(p.numel() for p in model.backbone.classifier.parameters())
print(f"  Features params: {features_params:,}")
print(f"  Classifier params: {classifier_params:,}")

features_trainable = sum(p.numel() for p in model.backbone.features.parameters() if p.requires_grad)
classifier_trainable = sum(p.numel() for p in model.backbone.classifier.parameters() if p.requires_grad)
print(f"  Features trainable: {features_trainable:,}")
print(f"  Classifier trainable: {classifier_trainable:,}")

# 6. Test with actual prediction
print("\n6. Testing prediction...")
model.eval()
with torch.no_grad():
    output = model(dummy_input)
    probs = torch.softmax(output, dim=1)
    pred_classes = torch.argmax(output, dim=1)
    
    print(f"  Predictions: {pred_classes}")
    print(f"  Max probability: {probs.max(dim=1)[0]}")
    print(f"  Min probability: {probs.min(dim=1)[0]}")
    
    # Check if all predictions are same
    if pred_classes[0] == pred_classes[1]:
        print("  ⚠️ Both predictions are SAME class!")
    else:
        print("  ✅ Predictions are different")

# 7. Check loss
print("\n7. Testing loss calculation...")
criterion = nn.CrossEntropyLoss()
dummy_labels = torch.tensor([0, 1])
loss = criterion(output, dummy_labels)
print(f"  Loss: {loss.item():.4f}")
print(f"  Expected: ~4.8 (log(120) ≈ 4.79 for random)")

if loss.item() > 4.7 and loss.item() < 4.9:
    print("  ⚠️ Loss is close to random! Model not learning!")
else:
    print("  ✅ Loss looks reasonable")

print("\n" + "="*80)
print("  DEBUG COMPLETED")
print("="*80)
