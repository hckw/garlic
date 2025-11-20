from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Dict, Literal, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image, ImageDraw, ImageFont

ROOT_DIR = Path(__file__).resolve().parents[1]
UPLOAD_DIR = ROOT_DIR / "data" / "uploads"
ANNOTATED_DIR = ROOT_DIR / "data" / "annotated"

for directory in (UPLOAD_DIR, ANNOTATED_DIR):
    directory.mkdir(parents=True, exist_ok=True)


@dataclass
class ImageRecord:
    image_id: str
    original_path: Path
    uploaded_at: datetime
    status: Literal["uploaded", "processing", "completed", "accepted", "needs_new_image"] = "uploaded"
    annotated_path: Optional[Path] = None
    reject_count: int = 0
    history: list[str] = field(default_factory=list)


class ImageStore:
    def __init__(self) -> None:
        self._records: Dict[str, ImageRecord] = {}
        self._lock = Lock()

    def create(self, record: ImageRecord) -> None:
        with self._lock:
            self._records[record.image_id] = record

    def get(self, image_id: str) -> ImageRecord:
        record = self._records.get(image_id)
        if record is None:
            raise KeyError(image_id)
        return record

    def update(self, image_id: str, **fields) -> ImageRecord:
        with self._lock:
            record = self.get(image_id)
            for key, value in fields.items():
                setattr(record, key, value)
            return record


store = ImageStore()
app = FastAPI(title="Garlic Detector API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"]
    ,
    allow_headers=["*"],
)


class UploadResponse(BaseModel):
    image_id: str
    status: str


class ProcessResponse(BaseModel):
    image_id: str
    status: str
    message: str


class ResultResponse(BaseModel):
    image_id: str
    status: str
    annotated_image_base64: str


class FeedbackRequest(BaseModel):
    image_id: str
    decision: Literal["accept", "reject"]


class FeedbackResponse(BaseModel):
    image_id: str
    action: Literal["completed", "reprocess", "reset"]
    reject_count: int
    message: Optional[str] = None


def _generate_image_id() -> str:
    return datetime.utcnow().strftime("%Y%m%d%H%M%S%f")


def _save_upload(image_id: str, file: UploadFile) -> Path:
    suffix = Path(file.filename or "image.png").suffix or ".png"
    output_path = UPLOAD_DIR / f"{image_id}{suffix}"
    with output_path.open("wb") as buffer:
        content = file.file.read()
        buffer.write(content)
    return output_path


def _annotate_image(original_path: Path, annotated_path: Path) -> None:
    image = Image.open(original_path).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")

    width, height = image.size
    box_padding = int(min(width, height) * 0.1)
    box_coords = [
        box_padding,
        box_padding,
        width - box_padding,
        height - box_padding,
    ]
    draw.rectangle(box_coords, outline=(34, 197, 94), width=6)
    draw.rectangle(box_coords, fill=(34, 197, 94, 40))

    try:
        font = ImageFont.truetype("Arial.ttf", size=32)
    except OSError:
        font = ImageFont.load_default()

    text = "Garlic detected"
    text_size = draw.textbbox((0, 0), text, font=font)
    text_width = text_size[2] - text_size[0]
    text_height = text_size[3] - text_size[1]
    text_position = (box_padding + 10, box_padding + 10)
    text_bg_coords = [
        text_position[0] - 8,
        text_position[1] - 4,
        text_position[0] + text_width + 8,
        text_position[1] + text_height + 4,
    ]
    draw.rectangle(text_bg_coords, fill=(15, 23, 42, 200))
    draw.text(text_position, text, font=font, fill=(255, 255, 255))

    image.save(annotated_path)


async def _simulate_processing(image_id: str, record: ImageRecord) -> None:
    annotated_path = ANNOTATED_DIR / f"{Path(record.original_path).stem}_annotated.png"
    store.update(image_id, status="processing", history=record.history + ["processing"])
    await asyncio.sleep(1.0)
    _annotate_image(record.original_path, annotated_path)
    store.update(
        image_id,
        status="completed",
        annotated_path=annotated_path,
        history=record.history + ["processing", "completed"],
    )


@app.post("/api/upload", response_model=UploadResponse)
async def upload_image(file: UploadFile = File(...)):
    image_id = _generate_image_id()
    original_path = _save_upload(image_id, file)
    record = ImageRecord(
        image_id=image_id,
        original_path=original_path,
        uploaded_at=datetime.utcnow(),
        history=["uploaded"],
    )
    store.create(record)
    return UploadResponse(image_id=image_id, status=record.status)


@app.post("/api/process/{image_id}", response_model=ProcessResponse)
async def process_image(image_id: str):
    try:
        record = store.get(image_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Image not found")

    if record.status == "completed":
        return ProcessResponse(image_id=image_id, status=record.status, message="Already processed")

    await _simulate_processing(image_id, record)
    return ProcessResponse(image_id=image_id, status="completed", message="Processing finished")


@app.get("/api/result/{image_id}", response_model=ResultResponse)
async def get_result(image_id: str):
    try:
        record = store.get(image_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Image not found")

    if record.annotated_path is None or not record.annotated_path.exists():
        raise HTTPException(status_code=400, detail="Image is not ready yet")

    with record.annotated_path.open("rb") as fp:
        encoded = base64.b64encode(fp.read()).decode("utf-8")
    return ResultResponse(image_id=image_id, status=record.status, annotated_image_base64=encoded)


@app.post("/api/feedback", response_model=FeedbackResponse)
async def submit_feedback(payload: FeedbackRequest):
    try:
        record = store.get(payload.image_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Image not found")

    if payload.decision == "accept":
        store.update(payload.image_id, status="accepted")
        return FeedbackResponse(
            image_id=payload.image_id,
            action="completed",
            reject_count=record.reject_count,
            message="Thanks for your confirmation!",
        )

    # reject path
    new_reject_count = record.reject_count + 1
    store.update(payload.image_id, reject_count=new_reject_count)

    if new_reject_count > 5:
        store.update(payload.image_id, status="needs_new_image")
        return FeedbackResponse(
            image_id=payload.image_id,
            action="reset",
            reject_count=new_reject_count,
            message=(
                "You’ve rejected this image more than 5 times. It may be unsuitable for detection. "
                "Please upload a clearer image."
            ),
        )

    store.update(payload.image_id, status="uploaded")
    return FeedbackResponse(
        image_id=payload.image_id,
        action="reprocess",
        reject_count=new_reject_count,
        message="We will reprocess this image for you.",
    )


@app.get("/health")
async def health_check():
    return {"status": "ok"}
