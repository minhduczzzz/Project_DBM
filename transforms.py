from torchvision.transforms import Compose, Resize, RandomResizedCrop, ToTensor, Normalize
from torchvision.transforms import RandomHorizontalFlip, RandomRotation, ColorJitter

def get_train_transform(input_size=224, augmentation=True):
    """Tạo transform cho training với Data Augmentation tối ưu"""
    transforms = []
    
    if augmentation:
        # Thay vì Resize cứng nhắc, dùng RandomResizedCrop để model học linh hoạt hơn
        transforms.extend([
            RandomResizedCrop(input_size, scale=(0.8, 1.0)),
            RandomHorizontalFlip(p=0.5),
            RandomRotation(degrees=15),
            ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        ])
    else:
        transforms.append(Resize((input_size, input_size)))
        
    transforms.extend([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return Compose(transforms)

def get_val_transform(input_size=224):
    """Tạo transform chuẩn cho validation/test"""
    return Compose([
        Resize((256, 256)), # Resize to larger first
        Resize((input_size, input_size)), # Then exact size (standard practice)
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])