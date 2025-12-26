from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass, field
from datetime import datetime
import os
from pathlib import Path
from threading import Lock
from typing import Dict, Literal, Optional, List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image, ImageDraw, ImageFont

from .model.inference import GarlicDetector

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
    detections: List[Dict] = field(default_factory=list)


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

# Initialize detector with error handling (Roboflow)
try:
    detector = GarlicDetector(
        api_key=os.getenv("ROBOFLOW_API_KEY"),
        model_id=os.getenv("ROBOFLOW_MODEL_ID"),
        version=os.getenv("ROBOFLOW_MODEL_VERSION"),
        confidence_threshold=float(os.getenv("GARLIC_CONFIDENCE_THRESHOLD", "0.5")),
    )
    print("✅ GarlicDetector initialized successfully (Roboflow)")
except Exception as e:
    print(f"⚠️ Warning: Could not initialize GarlicDetector: {e}")
    print("⚠️ Using fallback - detector will be initialized on first use")
    detector = None

# CORS configuration - update with your Netlify domain in production
allowed_origins = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,http://localhost:8080,http://localhost:8501"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins + ["*"] if os.getenv("ENVIRONMENT") == "development" else allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
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
    detections: List[Dict]


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


def _annotate_with_detections(original_path: Path, annotated_path: Path, detections: List[Dict]) -> None:
    if not detections:
        _annotate_image(original_path, annotated_path)
        return

    image = Image.open(original_path).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")

    try:
        font = ImageFont.truetype("Arial.ttf", size=24)
    except OSError:
        font = ImageFont.load_default()

    for detection in detections:
        bbox = detection["bbox"]
        coords = [
            bbox["x_min"],
            bbox["y_min"],
            bbox["x_max"],
            bbox["y_max"],
        ]
        draw.rectangle(coords, outline=(34, 197, 94), width=4)
        draw.rectangle(coords, fill=(34, 197, 94, 40))

        label = detection["label"]
        confidence = detection["confidence"] * 100
        text = f"{label} {confidence:.1f}%"
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_height = text_bbox[3] - text_bbox[1]
        text_width = text_bbox[2] - text_bbox[0]
        
        # Calculate background coordinates, ensuring y1 >= y0
        y0 = max(0, coords[1] - text_height - 6)
        y1 = max(y0 + text_height + 4, coords[1] - 2)
        
        bg_coords = [
            coords[0],
            y0,
            coords[0] + text_width + 12,
            y1,
        ]
        draw.rectangle(bg_coords, fill=(15, 23, 42, 200))
        draw.text((bg_coords[0] + 6, bg_coords[1] + 2), text, font=font, fill=(255, 255, 255))

    image.save(annotated_path)


async def _process_with_model(image_id: str, record: ImageRecord) -> None:
    annotated_path = ANNOTATED_DIR / f"{Path(record.original_path).stem}_annotated.png"
    
    # Ensure annotated directory exists
    ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)
    
    # Mark as processing
    store.update(image_id, status="processing", history=record.history + ["processing"])

    # Initialize detector if not already initialized
    global detector
    if detector is None:
        try:
            print(f"🔧 Initializing GarlicDetector instance...")
            detector = GarlicDetector(
                api_key=os.getenv("ROBOFLOW_API_KEY"),
                model_id=os.getenv("ROBOFLOW_MODEL_ID"),
                version=os.getenv("ROBOFLOW_MODEL_VERSION"),
                confidence_threshold=float(os.getenv("GARLIC_CONFIDENCE_THRESHOLD", "0.5")),
            )
            print("✅ GarlicDetector instance created successfully (Roboflow)")
        except Exception as e:
            print(f"❌ Error initializing detector: {e}")
            import traceback
            traceback.print_exc()
            store.update(image_id, status="uploaded", history=record.history + ["processing", "error"])
            raise HTTPException(status_code=500, detail=f"Failed to initialize detector: {str(e)}")
    else:
        print("✅ Reusing existing GarlicDetector instance (Roboflow client already initialized)")

    try:
        print(f"🔍 Running detection on {record.original_path}...")
        detection_results = await asyncio.to_thread(detector.detect, record.original_path)
        print(f"✅ Detection completed: {len(detection_results)} detections found")
    except Exception as e:
        print(f"❌ Error during detection: {e}")
        import traceback
        traceback.print_exc()
        store.update(image_id, status="uploaded", history=record.history + ["processing", "error"])
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")
    
    try:
        detection_dicts = [result.to_dict() for result in detection_results]
        print(f"📝 Annotating image...")
        _annotate_with_detections(record.original_path, annotated_path, detection_dicts)
        print(f"✅ Annotation saved to {annotated_path}")
    except Exception as e:
        print(f"❌ Error during annotation: {e}")
        import traceback
        traceback.print_exc()
        store.update(image_id, status="uploaded", history=record.history + ["processing", "error"])
        raise HTTPException(status_code=500, detail=f"Annotation failed: {str(e)}")

    store.update(
        image_id,
        status="completed",
        annotated_path=annotated_path,
        detections=detection_dicts,
        history=record.history + ["processing", "completed"],
    )
    print(f"✅ Processing completed for {image_id}")


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

    # Verify image file exists
    if not record.original_path.exists():
        raise HTTPException(status_code=400, detail=f"Image file not found: {record.original_path}")

    # Run processing synchronously so the caller gets the result or an error
    try:
        await _process_with_model(image_id, record)
        return ProcessResponse(image_id=image_id, status="completed", message="Processing finished")
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Unexpected error in process_image: {e}")
        import traceback
        traceback.print_exc()
        store.update(image_id, status="uploaded", history=record.history + ["processing", "error"])
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


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
    return ResultResponse(
        image_id=image_id,
        status=record.status,
        annotated_image_base64=encoded,
        detections=record.detections,
    )


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

    if new_reject_count > 2:
        store.update(payload.image_id, status="needs_new_image")
        return FeedbackResponse(
            image_id=payload.image_id,
            action="reset",
            reject_count=new_reject_count,
            message=(
                "You've rejected this image more than 2 times. It may be unsuitable for detection. "
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


@app.get("/")
async def root():
    return {
        "message": "Garlic Detector API",
        "version": "0.1.0",
        "endpoints": {
            "health": "/health",
            "upload": "/api/upload",
            "process": "/api/process/{image_id}",
            "result": "/api/result/{image_id}",
            "feedback": "/api/feedback"
        }
    }


@app.get("/health")
async def health_check():
    return {"status": "ok"}
