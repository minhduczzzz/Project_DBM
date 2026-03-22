import torch.nn as nn
from torchvision import models

class DogBreedVGG16(nn.Module):
    """VGG16 model for Dog Breed Classification (Optimized Fine-tuning)"""
    def __init__(self, num_classes=120, pretrained=True, dropout=0.3):
        super().__init__()

        if pretrained:
            weights = models.VGG16_Weights.DEFAULT
        else:
            weights = None

        self.backbone = models.vgg16(weights=weights)

        # Cập nhật dropout để chống overfitting
        self.backbone.classifier[2].p = dropout
        self.backbone.classifier[5].p = dropout

        # CHỈ thay thế lớp Linear cuối cùng (Lớp số 6)
        # Giữ lại toàn bộ trọng số của các lớp ẩn (4096 dimensions) phía trước
        in_features = self.backbone.classifier[6].in_features
        self.backbone.classifier[6] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)