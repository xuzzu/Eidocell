"""Deep feature extraction processors using pretrained CNN backbones."""

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models

from core.processors.inference.feature_extraction import FeatureExtractionProcessor

# ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class MobileNetV3Extraction(FeatureExtractionProcessor):
    """Extract features using MobileNetV3-Small pretrained on ImageNet."""

    FEATURE_DIM = 576

    def __init__(self):
        self._model: nn.Module | None = None
        self._transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256, antialias=True),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def _load_model(self) -> nn.Module:
        if self._model is None:
            backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
            # Use features + avgpool + flatten as the feature extractor
            self._model = nn.Sequential(
                backbone.features,
                backbone.avgpool,
                nn.Flatten(1),
            )
            self._model.eval()
        return self._model

    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Convert BGR/grayscale image to normalized RGB tensor."""
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Ensure uint8
        if image.dtype != np.uint8:
            if image.dtype in (np.float32, np.float64):
                image = (image * 255).clip(0, 255).astype(np.uint8)
            elif image.dtype == np.uint16:
                image = (image / 256).astype(np.uint8)

        tensor = self._transform(image)
        return tensor.unsqueeze(0)  # (1, 3, 224, 224)

    def extract(self, image: np.ndarray) -> np.ndarray:
        model = self._load_model()
        tensor = self._preprocess(image)
        with torch.no_grad():
            features = model(tensor)  # (1, 576)
        return features.squeeze(0).cpu().numpy().astype(np.float32)

    def feature_dim(self) -> int:
        return self.FEATURE_DIM
