import torch.nn as nn
from torchvision import models


class DogBreedVGG16(nn.Module):
    """VGG16 model for Dog Breed Classification"""
    def __init__(self, num_classes=120, pretrained=True):
        super().__init__()

        if pretrained:
            weights = models.VGG16_Weights.DEFAULT
        else:
            weights = None

        self.backbone = models.vgg16(weights=weights)

        # Replace classifier
        in_features = self.backbone.classifier[6].in_features
        self.backbone.classifier[6] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)
