# backend/pytorch_processor.py
import logging
import os
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import tifffile
import torch
import torchvision.transforms as T
from PIL import Image

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

from backend.resources.dinov2.models import vision_transformer as vits

DINOV2_AVAILABLE = True
logging.info("DINOv2 library found and imported for model construction.")


def get_dinov2_vit_backbone(
    arch_name: str,
    patch_size: int,
    num_register_tokens: int = 0,
    **kwargs: Any,
) -> torch.nn.Module:
    """
    Creates a standalone DINOv2 Vision Transformer backbone.
    This function encapsulates the logic from your export script.
    """
    if not DINOV2_AVAILABLE:
        raise RuntimeError(
            "DINOv2 library is not available, cannot construct DINOv2 model."
        )

    vit_kwargs = dict(
        patch_size=patch_size,
        img_size=224,
        init_values=kwargs.get("layerscale", 1.0e-5),
        ffn_layer=kwargs.get("ffn_layer", "mlp"),
        block_chunks=kwargs.get("block_chunks", 0),
        num_register_tokens=num_register_tokens,
        interpolate_offset=kwargs.get("interpolate_offset", 0.0),
        interpolate_antialias=kwargs.get("interpolate_antialias", False),
    )
    vit_kwargs.update({k: v for k, v in kwargs.items() if k not in vit_kwargs})

    logging.debug(
        f"Constructing DINOv2 backbone: {arch_name}, patch_size={patch_size}, registers={num_register_tokens}, kwargs={vit_kwargs}"
    )

    model = None
    if arch_name == "vit_small":
        model = vits.vit_small(**vit_kwargs)
    elif arch_name == "vit_base":
        model = vits.vit_base(**vit_kwargs)
    elif arch_name == "vit_large":
        model = vits.vit_large(**vit_kwargs)
    elif arch_name == "vit_giant2":
        vit_kwargs["ffn_layer"] = kwargs.get("ffn_layer", "swiglufused")
        model = vits.vit_giant2(**vit_kwargs)
    else:
        raise ValueError(f"Unsupported ViT architecture for DINOv2: {arch_name}")

    return model


class PyTorchFeatureExtractor:
    def __init__(
        self,
        model_name: str,
        model_path: str,
        feature_dim: int,
        architecture: Optional[str] = None,
        device: str = "auto",
    ):
        self.model_name = model_name
        self.model_path = model_path
        self.feature_dim = feature_dim
        self.architecture = architecture
        self.device = self._determine_device(device)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"PyTorch model file not found: {model_path}")

        try:
            if self.architecture and "mobilenetv3s" in self.architecture.lower():
                full_model = torch.load(
                    model_path, map_location=self.device, weights_only=False
                )
                if hasattr(full_model, "features") and hasattr(full_model, "avgpool"):
                    self.model = torch.nn.Sequential(
                        full_model.features,
                        full_model.avgpool,
                        torch.nn.Flatten(1),  # Ensure output is (batch_size, features)
                    )
                    logging.info(
                        f"Adapted MobileNetV3-Small model '{model_name}' to use features + avgpool + flatten."
                    )
                else:
                    logging.warning(
                        f"MobileNetV3-Small model '{model_name}' does not have expected 'features' and 'avgpool'. Using as is. Ensure it outputs {feature_dim}-dim features directly or after flatten."
                    )
                    self.model = torch.nn.Sequential(full_model, torch.nn.Flatten(1))
            elif self.architecture and "dinov2" in self.architecture.lower():
                if not DINOV2_AVAILABLE:
                    raise RuntimeError(
                        "DINOv2 PyTorch model specified, but DINOv2 library components are not available."
                    )

                vit_arch_name = "vit_small"  # Default
                patch_size = 14  # Default
                num_registers = 0

                logging.info(
                    f"Reconstructing DINOv2 backbone: arch={vit_arch_name}, patch={patch_size}, registers={num_registers} for model {self.model_name}"
                )
                self.model = get_dinov2_vit_backbone(
                    vit_arch_name, patch_size, num_registers
                )
                state_dict = torch.load(model_path, map_location=self.device)
                load_msg = self.model.load_state_dict(state_dict, strict=True)
                logging.info(
                    f"Loaded state_dict into reconstructed DINOv2 model '{model_name}'. Message: {load_msg}"
                )
                if load_msg.missing_keys or load_msg.unexpected_keys:
                    logging.warning(
                        f"DINOv2 state_dict load: Missing - {load_msg.missing_keys}, Unexpected - {load_msg.unexpected_keys}"
                    )
            else:
                self.model = full_model

            if isinstance(self.model, torch.nn.Module):
                self.model.eval()
            else:
                logging.info(
                    f"Loaded model for {model_name} is not an nn.Module instance. Assuming eval mode handled."
                )

            logging.info(
                f"PyTorch model '{model_name}' (extractor part if adapted) loaded from '{model_path}' on device '{self.device}'."
            )
        except Exception as e:
            logging.error(
                f"Failed to load or adapt PyTorch model '{model_name}' from '{model_path}': {e}",
                exc_info=True,
            )
            raise RuntimeError(
                f"PyTorch model loading/adaptation failed for {model_name}"
            ) from e

        self._initialize_transforms()
        self.linear_probe = None

    def _determine_device(self, requested_device: str) -> str:
        if requested_device == "auto":
            if torch.cuda.is_available():
                logging.info("CUDA is available. Using CUDA for PyTorch models.")
                return "cuda"
            logging.info("CUDA not available. Using CPU for PyTorch models.")
            return "cpu"
        elif requested_device == "cuda":
            if not torch.cuda.is_available():
                logging.warning(
                    "CUDA requested but not available. Falling back to CPU for PyTorch models."
                )
                return "cpu"
            logging.info("Using CUDA as requested for PyTorch models.")
            return "cuda"
        logging.info("Using CPU as requested for PyTorch models.")
        return "cpu"

    def _initialize_transforms(self):
        # Common ImageNet normalization, adjust if your models need specific DINOv2 preprocessing
        if self.architecture and "dinov2" in self.architecture.lower():
            # DINOv2 often uses 224x224, but some variants might use slightly different sizes or interpolation.
            # Official DINOv2 preprocessing can be more complex. This is a standard starting point.
            self.transform = T.Compose(
                [
                    T.Resize(
                        256, interpolation=T.InterpolationMode.BICUBIC, antialias=True
                    ),
                    T.CenterCrop(224),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
            logging.info(f"Initialized DINOv2-style transforms for {self.model_name}.")
        elif self.architecture and "mobilenetv3" in self.architecture.lower():
            self.transform = T.Compose(
                [
                    T.Resize(256, antialias=True),
                    T.CenterCrop(224),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
            logging.info(
                f"Initialized MobileNetV3-style transforms for {self.model_name}."
            )
        else:
            self.transform = T.Compose(
                [
                    T.Resize(256, antialias=True),
                    T.CenterCrop(224),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
            logging.warning(
                f"Initialized generic ImageNet-style transforms for {self.model_name} (arch: {self.architecture})."
            )

    def extract_features(
        self, image_path: str
    ) -> torch.Tensor:  # Returns torch.Tensor on CPU
        pil_image = None
        try:
            if image_path.lower().endswith((".tif", ".tiff")):
                img_data_tiff = tifffile.imread(image_path)

                # Handle multi-page TIFFs (take first page)
                if (
                    img_data_tiff.ndim > 2
                    and img_data_tiff.shape[0] > 1
                    and img_data_tiff.ndim == 3
                ):
                    if img_data_tiff.shape[0] <= 4:
                        img_data_tiff = np.moveaxis(img_data_tiff, 0, -1)
                    else:
                        img_data_tiff = img_data_tiff[0]
                        logging.info(
                            f"Multi-page TIFF detected for {image_path}. Using first page."
                        )
                elif img_data_tiff.ndim > 3 and img_data_tiff.shape[0] > 1:
                    img_data_tiff = img_data_tiff[0]
                    logging.info(
                        f"Multi-page TIFF detected for {image_path}. Using first page."
                    )

                # Normalize to 0-255 uint8 if necessary
                if img_data_tiff.dtype != np.uint8:
                    if np.issubdtype(img_data_tiff.dtype, np.floating):
                        img_data_tiff = (
                            (img_data_tiff - np.min(img_data_tiff))
                            / (np.max(img_data_tiff) - np.min(img_data_tiff) + 1e-9)
                            * 255
                        )
                    elif img_data_tiff.max() > 255:
                        img_data_tiff = (
                            img_data_tiff / (img_data_tiff.max() / 255.0)
                        ).astype(np.uint8)
                    img_data_tiff = img_data_tiff.astype(np.uint8)

                # Ensure 3 channels (RGB) for PIL
                if img_data_tiff.ndim == 2:  # Grayscale
                    pil_image = Image.fromarray(img_data_tiff, mode="L").convert("RGB")
                elif img_data_tiff.ndim == 3:
                    if img_data_tiff.shape[-1] == 1:  # Grayscale with channel dim
                        pil_image = Image.fromarray(
                            img_data_tiff.squeeze(axis=-1), mode="L"
                        ).convert("RGB")
                    elif img_data_tiff.shape[-1] == 3:  # RGB
                        pil_image = Image.fromarray(img_data_tiff, mode="RGB")
                    elif img_data_tiff.shape[-1] == 4:  # RGBA
                        pil_image = Image.fromarray(img_data_tiff, mode="RGBA").convert(
                            "RGB"
                        )  # Convert RGBA to RGB
                    else:
                        logging.error(
                            f"Unsupported TIFF channel count for {image_path}: {img_data_tiff.shape[-1]}."
                        )
                        return torch.zeros(self.feature_dim, device="cpu")
                else:
                    logging.error(
                        f"Unsupported TIFF dimensions for {image_path}: {img_data_tiff.ndim}."
                    )
                    return torch.zeros(self.feature_dim, device="cpu")
            else:  # For non-TIFF files
                pil_image = Image.open(image_path).convert("RGB")

            if (
                pil_image is None
            ):  # Should be caught by exceptions above if loading failed
                raise ValueError("Failed to load image into PIL format.")

            img_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                features = self.model(img_tensor)

            if features.ndim != 2 or features.shape[0] != 1:
                logging.warning(
                    f"PyTorch model '{self.model_name}' output unexpected shape {features.shape}. Attempting to flatten."
                )
                features = features.view(1, -1)

            if features.shape[1] != self.feature_dim:
                logging.error(
                    f"PyTorch feature dimension mismatch for {self.model_name}. Expected {self.feature_dim}, got {features.shape[1]}."
                )
                return torch.zeros(self.feature_dim, device="cpu")

            return features.squeeze(0).cpu()
        except Exception as e:
            logging.error(
                f"Error extracting PyTorch features for {image_path} using {self.model_name}: {e}",
                exc_info=True,
            )
            return torch.zeros(self.feature_dim, device="cpu")

    def train_linear_probe(
        self,
        features_list: List[torch.Tensor],
        integer_labels_list: List[int],
        num_classes: int,
        class_id_to_int_label_map: Dict,  # Pass the map
        epochs: int = 100,
        lr: float = 0.001,
        batch_size: int = 32,
        patience: int = -1,  # -1 means no early stopping
        progress_callback=None,
    ):
        if (
            not features_list
            or not integer_labels_list
            or len(features_list) != len(integer_labels_list)
        ):
            logging.error("LP Train: Empty/mismatched features/labels.")
            return
        if num_classes <= 0:
            logging.error(f"LP Train: Invalid num_classes ({num_classes}).")
            return

        self.class_id_to_int_label_map = class_id_to_int_label_map
        self.int_label_to_class_id_map = {
            v: k for k, v in class_id_to_int_label_map.items()
        }

        try:
            X = torch.stack(features_list).to(self.device)
        except RuntimeError as e:
            logging.error(f"LP Train: Error stacking features: {e}")
            return
        y = torch.tensor(integer_labels_list, dtype=torch.long).to(self.device)

        input_dim = X.shape[1]
        if input_dim != self.feature_dim:
            logging.error(
                f"LP Train: Feature dim mismatch ({input_dim} vs {self.feature_dim})."
            )
            return

        self.linear_probe = torch.nn.Linear(input_dim, num_classes).to(self.device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            self.linear_probe.parameters(), lr=lr, weight_decay=1e-4
        )

        actual_batch_size = min(batch_size, len(X))
        if actual_batch_size <= 0:
            actual_batch_size = 1  # Handle case with very few samples

        logging.info(
            f"LP Train Start: model '{self.model_name}', {len(X)} samples, {num_classes} classes, device '{self.device}'."
        )
        self.linear_probe.train()

        best_loss = float("inf")
        epochs_no_improve = 0

        for epoch in range(epochs):
            permutation = torch.randperm(X.size(0))
            epoch_loss = 0.0
            num_batches = 0
            for i in range(0, X.size(0), actual_batch_size):
                optimizer.zero_grad()
                indices = permutation[i : i + actual_batch_size]
                batch_X, batch_y = X[indices], y[indices]
                outputs = self.linear_probe(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                num_batches += 1

            avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
            if progress_callback:
                progress_callback(epoch + 1, avg_epoch_loss)
            if (epoch + 1) % 10 == 0:
                logging.info(
                    f"LP Train - Epoch {epoch+1}/{epochs}, Avg Loss: {avg_epoch_loss:.4f}"
                )

            # Early stopping logic
            if patience > 0:
                if avg_epoch_loss < best_loss:
                    best_loss = avg_epoch_loss
                    epochs_no_improve = 0
                    # Optionally save best model state_dict here
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= patience:
                    logging.info(
                        f"LP Train: Early stopping triggered at epoch {epoch+1} due to no improvement for {patience} epochs."
                    )
                    break

        self.linear_probe.eval()
        logging.info(f"LP Train End: model '{self.model_name}'.")

    def classify_with_probe(
        self, features: torch.Tensor
    ) -> Optional[int]:  # Returns single predicted int label
        if self.linear_probe is None:
            logging.warning(f"LP Classify: Probe not trained for {self.model_name}.")
            return None
        if features.ndim == 1:
            features = features.unsqueeze(0)
        if features.shape[1] != self.feature_dim:
            logging.error(
                f"LP Classify: Input feature dim {features.shape[1]} != probe's expected dim {self.feature_dim}."
            )
            return None

        features_on_device = features.to(self.device)
        self.linear_probe.eval()
        with torch.no_grad():
            outputs = self.linear_probe(
                features_on_device
            )  # Should be (1, num_classes)
            _, predicted_class_index = torch.max(outputs, 1)
        return predicted_class_index.cpu().item()  # Return single int
