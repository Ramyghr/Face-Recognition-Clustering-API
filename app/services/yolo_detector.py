# # app/services/yolo_detector.py
import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import cv2
import numpy as np
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

from app.core.config import settings

logger = logging.getLogger(__name__)

# Single global
YOLO_MODEL: Optional[YOLO] = None


def _try_load_from(path: str) -> Optional[YOLO]:
    """Try to load a YOLO model from a local path; return None on failure."""
    try:
        if path and os.path.exists(path):
            model = YOLO(path)
            logger.info(f"[YOLO] Loaded model from: {path}")
            return model
    except Exception as e:
        logger.warning(f"[YOLO] Failed to load from {path}: {e}")
    return None


def _download_from_hf() -> YOLO:
    """
    Download the face model from Hugging Face cache (or reuse cache),
    then load with Ultralytics.
    """
    logger.info("[YOLO] Downloading/using cached HF model…")
    model_path = hf_hub_download(
        repo_id="arnabdhar/YOLOv8-Face-Detection",
        filename="model.pt",
        cache_dir="./app/ai_models",
    )
    model_path = Path(model_path).as_posix()
    logger.info(f"[YOLO] HF model path: {model_path}")
    return YOLO(model_path)


def init_yolo_model() -> YOLO:
    """
    Initialize the global YOLO model once.
    Order:
      1) exact snapshot path if present (the one you logged earlier)
      2) settings.YOLO_MODEL_PATH if present
      3) Hugging Face download/cache
    """
    global YOLO_MODEL
    if YOLO_MODEL is not None:
        return YOLO_MODEL

    # 1) Snapshot path that appears in your logs
    snapshot_path = (
        "./app/ai_models/models--arnabdhar--YOLOv8-Face-Detection/"
        "snapshots/52fa54977207fa4f021de949b515fb19dcab4488/model.pt"
    )
    YOLO_MODEL = _try_load_from(snapshot_path)
    if YOLO_MODEL is not None:
        return YOLO_MODEL

    # 2) settings path
    settings_path = getattr(settings, "YOLO_MODEL_PATH", "")
    if settings_path and settings_path != snapshot_path:
        YOLO_MODEL = _try_load_from(settings_path)
        if YOLO_MODEL is not None:
            return YOLO_MODEL

    # 3) Hugging Face
    try:
        YOLO_MODEL = _download_from_hf()
        return YOLO_MODEL
    except Exception as e:
        logger.error(f"[YOLO] Initialization failed: {e}")
        raise RuntimeError("Couldn't initialize face detector") from e



def detect_faces(
    image_bgr: np.ndarray,
    conf_thres: float = 0.5,
    max_faces: int = 10,
    min_face_size: int = None,
) -> Tuple[List[Dict[str, Any]], bool]:
    """
    Detect faces in a BGR image using Ultralytics YOLOv8 face model.
    Returns (faces, single_face_flag).

    Each face dict:
      {
        "bbox": (x, y, w, h),  # ints
        "face": np.ndarray(BGR),
        "confidence": float
      }
    """
    global YOLO_MODEL
    if YOLO_MODEL is None:
        YOLO_MODEL = init_yolo_model()

    if min_face_size is None:
        min_face_size = getattr(settings, "MIN_FACE_SIZE", 30)

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = image_rgb.shape[:2]

    results = YOLO_MODEL.predict(
        source=image_rgb,
        conf=conf_thres,
        verbose=False,
        device="cpu",
        max_det=max_faces,
    )

    faces: List[Dict[str, Any]] = []
    for res in results:
        if res.boxes is None:
            continue

        boxes = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()

        for i, (x1, y1, x2, y2) in enumerate(boxes):
            conf = float(confs[i])
            if conf < conf_thres:
                continue

            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(w, x2); y2 = min(h, y2)
            bw, bh = x2 - x1, y2 - y1

            if bw < min_face_size or bh < min_face_size:
                continue

            face_crop_bgr = image_bgr[y1:y2, x1:x2]
            faces.append({
                "bbox": (x1, y1, bw, bh),
                "face": face_crop_bgr,
                "confidence": conf,
            })

            if len(faces) >= max_faces:
                break

    return faces, (len(faces) == 1)



# keep this helper (some code calls predict)
def predict(image_bgr: np.ndarray, conf_thres: float = 0.5):
    faces, _ = detect_faces(image_bgr, conf_thres=conf_thres)
    return faces

# Change the function to be synchronous since YOLO loading is blocking I/O
def load_yolo_from_hf():
    """
    Downloads YOLOv8 face model from Hugging Face and initializes it using ultralytics.
    """
    global YOLO_MODEL
    if YOLO_MODEL is not None:
        return YOLO_MODEL

    try:
        logger.info("[YOLO] Loading YOLOv8 face model from cache or downloading...")
        model_path = hf_hub_download(
            repo_id="arnabdhar/YOLOv8-Face-Detection",
            filename="model.pt",
            cache_dir="./app/ai_models"
        )
        model_path = Path(model_path).as_posix()
        logger.info(f"[YOLO] Model loaded from {model_path}")

        YOLO_MODEL = YOLO(model_path)
        # Optimize for inference
        YOLO_MODEL.model.eval()
        return YOLO_MODEL
    except Exception as e:
        logger.error(f"[YOLO] Failed to load model: {e}")
        raise RuntimeError(f"Failed to load YOLOv8 model: {str(e)}")