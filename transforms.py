"""
Transform Factory - Tạo transforms cho các models khác nhau
Giúp tái sử dụng code và đảm bảo consistency
"""

from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torchvision.transforms import RandomHorizontalFlip, RandomRotation, ColorJitter


def get_train_transform(input_size=224, augmentation=True):
    """
    Tạo transform cho training
    
    Args:
        input_size: Kích thước input của model (224 cho VGG/ResNet, 240 cho EfficientNet)
        augmentation: Có áp dụng augmentation không
    
    Returns:
        Compose transform
    """
    transforms = [Resize((input_size, input_size))]
    
    if augmentation:
        transforms.extend([
            RandomHorizontalFlip(p=0.5),
            RandomRotation(degrees=15),
            ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
        ])
    
    transforms.extend([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return Compose(transforms)


def get_val_transform(input_size=224):
    """
    Tạo transform cho validation/test
    
    Args:
        input_size: Kích thước input của model
    
    Returns:
        Compose transform
    """
    return Compose([
        Resize((input_size, input_size)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


# Predefined transforms cho các models phổ biến
VGG_TRAIN_TRANSFORM = get_train_transform(input_size=224, augmentation=True)
VGG_VAL_TRANSFORM = get_val_transform(input_size=224)

RESNET_TRAIN_TRANSFORM = get_train_transform(input_size=224, augmentation=True)
RESNET_VAL_TRANSFORM = get_val_transform(input_size=224)

EFFICIENTNET_TRAIN_TRANSFORM = get_train_transform(input_size=240, augmentation=True)
EFFICIENTNET_VAL_TRANSFORM = get_val_transform(input_size=240)

ALEXNET_TRAIN_TRANSFORM = get_train_transform(input_size=227, augmentation=True)
ALEXNET_VAL_TRANSFORM = get_val_transform(input_size=227)


# Usage examples:
"""
# Cách 1: Dùng predefined
from transforms import VGG_TRAIN_TRANSFORM, VGG_VAL_TRANSFORM
train_dataset = DogBreedTrainValDataset(..., transform=VGG_TRAIN_TRANSFORM)

# Cách 2: Tạo custom
from transforms import get_train_transform
custom_transform = get_train_transform(input_size=256, augmentation=True)
train_dataset = DogBreedTrainValDataset(..., transform=custom_transform)

# Cách 3: Không augmentation
from transforms import get_train_transform
no_aug_transform = get_train_transform(input_size=224, augmentation=False)
train_dataset = DogBreedTrainValDataset(..., transform=no_aug_transform)
"""
