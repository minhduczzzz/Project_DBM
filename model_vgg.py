import torch.nn as nn
from torchvision import models


class DogBreedVGG16(nn.Module):
    """VGG16 model for Dog Breed Classification with Dropout"""
    def __init__(self, num_classes=120, pretrained=True, dropout=0.3):  # Giảm từ 0.5 → 0.3
        super().__init__()

        if pretrained:
            weights = models.VGG16_Weights.DEFAULT
        else:
            weights = None

        self.backbone = models.vgg16(weights=weights)

        # VGG16 classifier structure: 
        # features (conv layers) → avgpool → classifier (fc layers)
        # Chỉ thay đổi classifier, giữ nguyên features
        
        self.backbone.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)
