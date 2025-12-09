from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
from PIL import Image
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


@dataclass
class DetectionResult:
    label: str
    confidence: float
    bbox: Dict[str, float]
    robot_coords: Dict[str, float]

    def to_dict(self) -> Dict[str, float]:
        return {
            "label": self.label,
            "confidence": self.confidence,
            "bbox": self.bbox,
            "robot_coords": self.robot_coords,
        }


class GarlicDetector:
    """
    Wrapper around a Faster R-CNN backbone for garlic-root detection.
    The detector expects Pascal VOC-style training; provide a weights_path
    after you fine-tune on your dataset.
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        confidence_threshold: float = 0.5,
        label_map: Optional[Dict[int, str]] = None,
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.confidence_threshold = confidence_threshold
        self.label_map = label_map or {1: "garlic_root"}
        self.weights_path = weights_path
        self.model = None  # Lazy initialization
        self.transform = transforms.Compose([transforms.ToTensor()])
        print(f"GarlicDetector initialized (device: {self.device}, weights: {weights_path})")
    
    def _ensure_model_loaded(self) -> None:
        """Lazy load the model only when needed."""
        if self.model is not None:
            return
        
        try:
            print("🔧 Loading Faster R-CNN model...")
            # Always build with pretrained weights first, then load custom weights if available
            self.model = self._build_model(num_classes=len(self.label_map) + 1, use_pretrained=True)
            
            if self.weights_path and Path(self.weights_path).exists():
                try:
                    print(f"📦 Loading custom weights from {self.weights_path}...")
                    self._load_weights(self.weights_path)
                    print("✅ Custom weights loaded successfully")
                except Exception as e:
                    print(f"⚠️ Warning: Could not load weights from {self.weights_path}: {e}. Using pretrained model.")
            else:
                print("ℹ️ No custom weights provided, using pretrained model")
            
            print(f"📤 Moving model to {self.device}...")
            self.model.to(self.device)
            self.model.eval()
            print("✅ Model loaded and ready for inference")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _build_model(self, num_classes: int, use_pretrained: bool = True):
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if use_pretrained else None
        model = fasterrcnn_resnet50_fpn(weights=weights)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        return model

    def _load_weights(self, weights_path: str) -> None:
        path = Path(weights_path)
        if not path.exists():
            raise FileNotFoundError(f"Model weights not found at {path}")
        state_dict = torch.load(path, map_location=self.device)
        if isinstance(state_dict, dict) and "model" in state_dict:
            state_dict = state_dict["model"]
        self.model.load_state_dict(state_dict)

    def detect(self, image_path: Path) -> List[DetectionResult]:
        # Ensure model is loaded before detection
        self._ensure_model_loaded()
        
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image).to(self.device)

        with torch.no_grad():
            outputs = self.model([tensor])[0]

        detections: List[DetectionResult] = []
        width, height = image.size

        for bbox, score, label_idx in zip(
            outputs["boxes"], outputs["scores"], outputs["labels"]
        ):
            confidence = float(score.cpu().item())
            if confidence < self.confidence_threshold:
                continue

            label_name = self.label_map.get(int(label_idx.cpu().item()), "garlic_root")
            bbox_array = bbox.cpu().tolist()
            bbox_dict = {
                "x_min": float(bbox_array[0]),
                "y_min": float(bbox_array[1]),
                "x_max": float(bbox_array[2]),
                "y_max": float(bbox_array[3]),
            }
            robot_coords = self._estimate_robot_coords(bbox_dict, width, height)

            detections.append(
                DetectionResult(
                    label=label_name,
                    confidence=confidence,
                    bbox=bbox_dict,
                    robot_coords=robot_coords,
                )
            )

        return detections

    def _estimate_robot_coords(
        self, bbox: Dict[str, float], width: int, height: int
    ) -> Dict[str, float]:
        x_center = (bbox["x_min"] + bbox["x_max"]) / 2
        y_center = (bbox["y_min"] + bbox["y_max"]) / 2

        normalized_x = (x_center / width) * 2 - 1  # map to [-1, 1]
        normalized_y = (y_center / height) * 2 - 1

        return {
            "x": round(normalized_x, 4),
            "y": round(normalized_y, 4),
            "z": 0.0,  # placeholder until depth calibration is available
        }

