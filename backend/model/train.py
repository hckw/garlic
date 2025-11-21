from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import box_iou


def collate_fn(batch: List[Tuple[torch.Tensor, Dict]]) -> Tuple:
    return tuple(zip(*batch))


class GarlicVOCDataset(Dataset):
    def __init__(self, root: Path, transforms_fn=None):
        self.root = Path(root)
        self.transforms_fn = transforms_fn or transforms.ToTensor()
        self.image_ids = sorted(
            {path.stem for path in self.root.glob("*.jpg")} | {path.stem for path in self.root.glob("*.png")}
        )
        if not self.image_ids:
            raise ValueError(f"No image files found in {self.root}")

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int):
        img_id = self.image_ids[idx]
        image_path = self.root / f"{img_id}.jpg"
        if not image_path.exists():
            image_path = self.root / f"{img_id}.png"
        annot_path = self.root / f"{img_id}.xml"

        if not image_path.exists() or not annot_path.exists():
            raise FileNotFoundError(f"Missing files for {img_id} in {self.root}")

        image = Image.open(image_path).convert("RGB")
        boxes = []

        import xml.etree.ElementTree as ET

        tree = ET.parse(annot_path)
        root = tree.getroot()
        for obj in root.findall("object"):
            bbox = obj.find("bndbox")
            xmin = float(bbox.find("xmin").text)
            ymin = float(bbox.find("ymin").text)
            xmax = float(bbox.find("xmax").text)
            ymax = float(bbox.find("ymax").text)
            boxes.append([xmin, ymin, xmax, ymax])

        boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
        labels_tensor = torch.ones((len(boxes),), dtype=torch.int64)
        area_tensor = (
            (boxes_tensor[:, 2] - boxes_tensor[:, 0]) * (boxes_tensor[:, 3] - boxes_tensor[:, 1])
            if len(boxes) > 0
            else torch.tensor([])
        )
        iscrowd_tensor = torch.zeros((len(boxes),), dtype=torch.int64)

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([idx]),
            "area": area_tensor,
            "iscrowd": iscrowd_tensor,
        }

        if self.transforms_fn:
            image = self.transforms_fn(image)

        return image, target


def build_model(num_classes: int = 2, pretrained: bool = True) -> torch.nn.Module:
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    model = fasterrcnn_resnet50_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def evaluate(
    model: torch.nn.Module,
    dataset: GarlicVOCDataset,
    device: torch.device,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.5,
) -> Dict[str, float]:
    model.eval()
    tp = fp = fn = 0

    with torch.no_grad():
        for image, target in dataset:
            image = image.to(device)
            gt_boxes = target["boxes"].to(device)

            prediction = model([image])[0]
            scores = prediction["scores"].cpu()
            keep = scores > score_threshold

            pred_boxes = prediction["boxes"][keep].to(device)
            if len(pred_boxes) == 0:
                fn += len(gt_boxes)
                continue

            ious = box_iou(pred_boxes, gt_boxes)
            matched_gt = set()

            for i, row in enumerate(ious):
                max_iou, idx = torch.max(row, dim=0)
                if max_iou.item() >= iou_threshold and idx.item() not in matched_gt:
                    tp += 1
                    matched_gt.add(idx.item())
                else:
                    fp += 1

            fn += len(gt_boxes) - len(matched_gt)

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
    }


def train(
    train_dir: Path,
    val_dir: Path,
    output_path: Path,
    epochs: int = 20,
    batch_size: int = 2,
    lr: float = 0.002,
    momentum: float = 0.9,
    weight_decay: float = 0.0005,
    confidence_threshold: float = 0.5,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.ToTensor()
    train_dataset = GarlicVOCDataset(train_dir, transforms_fn=transform)
    val_dataset = GarlicVOCDataset(val_dir, transforms_fn=transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    model = build_model(num_classes=2, pretrained=True)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)

    metrics_history: List[Dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            total_loss += losses.item()

        metrics = evaluate(model, val_dataset, device, score_threshold=confidence_threshold)
        metrics_history.append({"epoch": epoch, "loss": total_loss, **metrics})

        print(
            f"[Epoch {epoch}/{epochs}] "
            f"Loss: {total_loss:.4f} | "
            f"Precision: {metrics['precision']:.4f} | "
            f"Recall: {metrics['recall']:.4f}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    history_path = output_path.with_suffix(".metrics.json")
    with history_path.open("w", encoding="utf-8") as fp:
        json.dump(metrics_history, fp, indent=2)

    print(f"Training complete. Weights saved to {output_path}")
    print(f"Metrics log saved to {history_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Faster R-CNN on garlic VOC dataset.")
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=Path("data/dataset/garlic alfina.v4i.voc/train"),
        help="Path to training dataset directory (Pascal VOC format).",
    )
    parser.add_argument(
        "--val-dir",
        type=Path,
        default=Path("data/dataset/garlic alfina.v4i.voc/valid"),
        help="Path to validation dataset directory (Pascal VOC format).",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("models/garlic_frcnn.pth"),
        help="Where to save the trained weights.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Score threshold when computing validation metrics.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        output_path=args.output_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        confidence_threshold=args.confidence_threshold,
    )

