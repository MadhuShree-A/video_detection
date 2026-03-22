"""
face_detector.py — YOLOv8-based face detector
"""

import logging
import numpy as np
import cv2
from typing import List, Tuple

logger = logging.getLogger(__name__)

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    logger.warning("Ultralytics not installed. Run: pip install ultralytics")


class FaceDetector:
    """
    YOLOv8 face detector.
    Uses 'yolov8n-face.pt' (Ultralytics face variant) by default.
    Falls back to the standard yolov8n.pt class-0 (person) detection if needed.
    """

    def __init__(self, model_path: str = "yolov8n-face.pt",
                 confidence: float = 0.5,
                 iou_threshold: float = 0.45,
                 min_face_size: int = 40):

        if not ULTRALYTICS_AVAILABLE:
            raise RuntimeError(
                "Ultralytics YOLOv8 required. Install with: pip install ultralytics"
            )

        logger.info(f"Loading YOLO model: {model_path}")
        self.model = YOLO("yolov8n-face.pt")
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.min_face_size = min_face_size
        logger.info("YOLO face detector loaded.")

    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """
        Returns list of (x1, y1, x2, y2, confidence) for detected faces.
        Filters out boxes smaller than min_face_size in either dimension.
        """
        results = self.model.predict(
            frame,
            conf=self.confidence,
            iou=self.iou_threshold,
            imgsz=1280,
            verbose=False
        )
        detections = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                w, h = x2 - x1, y2 - y1
                if w < self.min_face_size or h < self.min_face_size:
                    continue
                detections.append((x1, y1, x2, y2, conf))
        return detections
