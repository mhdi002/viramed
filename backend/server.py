import os
import io
import base64
import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from pymongo import MongoClient, ASCENDING
from dotenv import load_dotenv

# IMPORTANT: All routes must be prefixed with '/api'
API_PREFIX = "/api"

app = FastAPI(title="YOLO Inference Service", openapi_url=f"{API_PREFIX}/openapi.json")

# CORS (frontend domain will be handled by kube ingress)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Folders
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Load .env from backend dir to populate process env if supervisor didn't export
load_dotenv(dotenv_path=os.path.join(BASE_DIR, ".env"), override=False)

STORAGE_DIR = os.path.join(BASE_DIR, "storage")
MODELS_DIR = os.path.join(STORAGE_DIR, "models")
UPLOADS_DIR = os.path.join(STORAGE_DIR, "uploads")
RESULTS_DIR = os.path.join(STORAGE_DIR, "results")
for d in [STORAGE_DIR, MODELS_DIR, UPLOADS_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

# MongoDB (MUST use MONGO_URL)
MONGO_URL = os.environ.get("MONGO_URL")
if not MONGO_URL:
    # Respect environment rules: Must use MONGO_URL, so fail early with clear message
    raise RuntimeError("MONGO_URL not set in backend environment. Please set backend/.env with MONGO_URL.")

mongo_client = MongoClient(MONGO_URL)
db = mongo_client["yolo_app"]
models_col = db["models"]
models_col.create_index([("name", ASCENDING)], unique=True)
models_col.create_index([("id", ASCENDING)], unique=True)

class ModelInfo(BaseModel):
    model_config = {"protected_namespaces": ()}
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    filename: str
    format: str = Field(description="pt|pth|onnx")
    model_type: Optional[str] = Field(default=None, description="e.g. yolov8, yolov5, custom")
    task: Optional[str] = Field(default="detect", description="detect|segment|classify")
    input_size: Optional[int] = Field(default=640, description="Square input size")
    classes: Optional[List[str]] = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    def to_mongo(self) -> Dict[str, Any]:
        d = self.model_dump()
        # Ensure plain types
        d["created_at"] = self.created_at
        return d


def pil_to_data_url(pil_img: Image.Image, format: str = "PNG") -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format=format)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/{format.lower()};base64,{b64}"


def annotate_image(image: Image.Image, boxes: List[Dict[str, Any]]) -> Image.Image:
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    for b in boxes:
        x1, y1, x2, y2 = b.get("x1"), b.get("y1"), b.get("x2"), b.get("y2")
        label = b.get("label", "obj")
        conf = b.get("confidence")
        color = (255, 0, 0)
        draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=2)
        text = f"{label}{f' {conf:.2f}' if conf is not None else ''}"
        if font:
            text_size = draw.textbbox((0, 0), text, font=font)
            tw = text_size[2] - text_size[0]
            th = text_size[3] - text_size[1]
            draw.rectangle([(x1, y1 - th - 4), (x1 + tw + 4, y1)], fill=color)
            draw.text((x1 + 2, y1 - th - 2), text, fill=(255, 255, 255), font=font)
        else:
            draw.text((x1, y1), text, fill=color)
    return image


@app.get(f"{API_PREFIX}/health")
def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}


@app.get(f"{API_PREFIX}/models")
def list_models():
    docs = list(models_col.find({}, {"_id": 0}).sort("name", ASCENDING))
    return {"models": docs}


@app.post(f"{API_PREFIX}/models/refresh")
def refresh_models():
    """Scan MODELS_DIR for model files (.pt/.pth/.onnx) and upsert into Mongo."""
    supported_ext = {".pt", ".pth", ".onnx"}
    found = []
    for fname in os.listdir(MODELS_DIR):
        fpath = os.path.join(MODELS_DIR, fname)
        if not os.path.isfile(fpath):
            continue
        ext = os.path.splitext(fname)[1].lower()
        if ext not in supported_ext:
            continue
        name = os.path.splitext(fname)[0]
        fmt = ext[1:]
        info = ModelInfo(
            name=name,
            filename=fname,
            format=fmt,
            model_type="auto",
        )
        # Upsert by name
        existing = models_col.find_one({"name": name})
        if existing:
            models_col.update_one({"name": name}, {"$set": info.to_mongo()})
            info.id = existing.get("id", info.id)
        else:
            models_col.insert_one(info.to_mongo())
        found.append(info.model_dump())
    return {"count": len(found), "models": found}


@app.post(f"{API_PREFIX}/models/register")
async def register_model(
    file: UploadFile = File(...),
    name: Optional[str] = Form(None),
    model_type: Optional[str] = Form("auto"),
    task: Optional[str] = Form("detect"),
    input_size: Optional[int] = Form(640),
):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".pt", ".pth", ".onnx"]:
        raise HTTPException(status_code=400, detail="Unsupported model format. Use .pt, .pth, or .onnx")

    safe_name = name or os.path.splitext(file.filename)[0]
    dest_path = os.path.join(MODELS_DIR, file.filename)
    with open(dest_path, "wb") as f:
        f.write(await file.read())

    info = ModelInfo(
        name=safe_name,
        filename=file.filename,
        format=ext[1:],
        model_type=model_type,
        task=task,
        input_size=input_size,
    )
    existing = models_col.find_one({"name": safe_name})
    if existing:
        models_col.update_one({"name": safe_name}, {"$set": info.to_mongo()})
        info.id = existing.get("id", info.id)
    else:
        models_col.insert_one(info.to_mongo())

    return {"model": info.model_dump()}


class InferResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    boxes: List[Dict[str, Any]]
    image: Optional[str] = None  # data URL base64
    meta: Dict[str, Any]


# Simple in-process cache for loaded models
_loaded_models: Dict[str, Any] = {}


def _load_ultralytics(path: str):
    try:
        from ultralytics import YOLO  # noqa: F401
    except Exception as e:
        raise HTTPException(status_code=501, detail=f"Ultralytics not installed or failed to import: {e}")
    from ultralytics import YOLO
    if path not in _loaded_models:
        _loaded_models[path] = YOLO(path)
    return _loaded_models[path]


@app.post(f"{API_PREFIX}/infer", response_model=InferResponse)
async def infer(
    image: UploadFile = File(...),
    model_name: str = Form(...),
    conf: float = Form(0.25),
    iou: float = Form(0.45),
    return_image: bool = Form(True),
):
    # Find model by name
    model_doc = models_col.find_one({"name": model_name}, {"_id": 0})
    if not model_doc:
        raise HTTPException(status_code=404, detail="Model not found. Try /api/models/refresh or register.")
    fmt = model_doc.get("format")
    model_path = os.path.join(MODELS_DIR, model_doc.get("filename"))
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model file missing on server.")

    img_bytes = await image.read()
    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    boxes: List[Dict[str, Any]] = []
    annotated_data_url: Optional[str] = None

    if fmt in ("pt",):
        # Use ultralytics
        model = _load_ultralytics(model_path)
        # Predict
        res = model.predict(source=np.array(pil_img), imgsz=640, conf=conf, iou=iou, verbose=False)
        if not res:
            boxes = []
        else:
            r0 = res[0]
            if hasattr(r0, "boxes") and r0.boxes is not None:
                for b in r0.boxes:
                    # YOLO outputs tensors; convert to float
                    xyxy = b.xyxy[0].tolist()
                    conf_v = float(b.conf[0]) if hasattr(b, "conf") and b.conf is not None else None
                    cls_v = int(b.cls[0]) if hasattr(b, "cls") and b.cls is not None else None
                    label = None
                    names = getattr(model, "names", None)
                    if names is not None and cls_v is not None and cls_v in names:
                        label = str(names[cls_v])
                    boxes.append({
                        "x1": float(xyxy[0]),
                        "y1": float(xyxy[1]),
                        "x2": float(xyxy[2]),
                        "y2": float(xyxy[3]),
                        "confidence": conf_v,
                        "class_id": cls_v,
                        "label": label or f"cls_{cls_v}",
                    })
            else:
                boxes = []
    elif fmt in ("onnx",):
        # Placeholder: You can use onnxruntime here if needed in future
        raise HTTPException(status_code=501, detail="ONNX runtime not implemented in FastAPI backend. Use Node.js backend path or add onnxruntime.")
    elif fmt in ("pth",):
        raise HTTPException(status_code=501, detail=".pth custom formats not supported here. Place files in models folder; Node.js backend can call Python adapters.")
    else:
        raise HTTPException(status_code=400, detail="Unsupported model format")

    if return_image:
        ann = annotate_image(pil_img.copy(), boxes)
        annotated_data_url = pil_to_data_url(ann)

    meta = {
        "model_name": model_name,
        "format": fmt,
        "count": len(boxes),
    }
    return InferResponse(boxes=boxes, image=annotated_data_url, meta=meta)


# Root note to remind users
@app.get("/")
def root_notice():
    return JSONResponse({
        "message": "This is backend service. Use /api/* routes. Frontend served separately.",
    })