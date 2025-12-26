from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from inference_sdk import InferenceHTTPClient
from PIL import Image


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
    Detector that calls Roboflow Hosted Inference API via inference-sdk.

    Required env vars:
      - ROBOFLOW_API_KEY
      - ROBOFLOW_MODEL_ID        (e.g., "garlic-root-detection")
      - ROBOFLOW_MODEL_VERSION   (e.g., "1")
    Optional:
      - ROBOFLOW_API_URL         (default: https://serverless.roboflow.com)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_id: Optional[str] = None,
        version: Optional[str] = None,
        confidence_threshold: float = 0.5,
        api_url: Optional[str] = None,
    ) -> None:
        self.api_key = api_key or os.getenv("ROBOFLOW_API_KEY")
        self.model_id = model_id or os.getenv("ROBOFLOW_MODEL_ID")
        self.version = version or os.getenv("ROBOFLOW_MODEL_VERSION", "1")
        self.confidence_threshold = confidence_threshold
        self.api_url = api_url or os.getenv("ROBOFLOW_API_URL", "https://serverless.roboflow.com")

        if not self.api_key or not self.model_id:
            raise ValueError("ROBOFLOW_API_KEY and ROBOFLOW_MODEL_ID must be set.")

        # Model reference for inference-sdk: allow either full "model/version" in MODEL_ID,
        # or separate MODEL_ID + MODEL_VERSION.
        if "/" in self.model_id:
            self.model_ref = self.model_id
        else:
            self.model_ref = f"{self.model_id}/{self.version}"

        self.client = InferenceHTTPClient(
            api_url=self.api_url,
            api_key=self.api_key,
        )
        print(
            f"GarlicDetector configured for Roboflow model "
            f"{self.model_ref} via {self.api_url} (threshold={self.confidence_threshold})"
        )

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

    def _bbox_from_prediction(self, pred: Dict[str, float]) -> Dict[str, float]:
        # Roboflow returns center x/y and width/height in pixels
        x_center = float(pred.get("x", 0.0))
        y_center = float(pred.get("y", 0.0))
        w = float(pred.get("width", 0.0))
        h = float(pred.get("height", 0.0))

        x_min = x_center - w / 2
        x_max = x_center + w / 2
        y_min = y_center - h / 2
        y_max = y_center + h / 2

        return {
            "x_min": x_min,
            "y_min": y_min,
            "x_max": x_max,
            "y_max": y_max,
        }

    def detect(self, image_path: Path) -> List[DetectionResult]:
        # Load image to get dimensions (for robot coords)
        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        # inference-sdk handles file upload internally
        data = self.client.infer(str(image_path), model_id=self.model_ref)

        # Roboflow may return "predictions" or "objects" depending on model type
        predictions = data.get("predictions", []) or data.get("objects", [])
        detections: List[DetectionResult] = []

        for pred in predictions:
            confidence = float(pred.get("confidence", 0.0))
            if confidence < self.confidence_threshold:
                continue

            bbox = self._bbox_from_prediction(pred)
            robot_coords = self._estimate_robot_coords(bbox, width, height)

            detections.append(
                DetectionResult(
                    label=str(pred.get("class", "object")),
                    confidence=confidence,
                    bbox=bbox,
                    robot_coords=robot_coords,
                )
            )

        return detections

