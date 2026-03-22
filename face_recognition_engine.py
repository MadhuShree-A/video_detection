"""
face_recognition_engine.py — InsightFace-based face recognition + embedding
"""

import logging
import warnings
import os
import numpy as np
import cv2
from typing import List, Optional, Tuple

# Suppress onnxruntime and insightface startup warnings
warnings.filterwarnings("ignore")
os.environ["ORT_LOGGING_LEVEL"] = "3"

logger = logging.getLogger(__name__)

try:
    import insightface
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    logger.warning("InsightFace not installed. Run: pip install insightface onnxruntime")


class FaceRecognitionEngine:
    """
    Wraps InsightFace FaceAnalysis to:
      - detect faces
      - generate 512-d ArcFace embeddings
      - compare embeddings via cosine similarity
    """

    def __init__(self, model_name: str = "buffalo_l",
                 ctx_id: int = 0,
                 det_size: Tuple[int, int] = (640, 640),
                 similarity_threshold: float = 0.45):

        self.similarity_threshold = similarity_threshold

        if not INSIGHTFACE_AVAILABLE:
            raise RuntimeError(
                "InsightFace is required. Install with: pip install insightface onnxruntime"
            )

        logger.info(f"Loading InsightFace model: {model_name}")
        self.app = FaceAnalysis(
            name=model_name,
            allowed_modules=["detection", "recognition"],
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        # Use smaller det_size=(320,320) — works better on small face crops
        self.app.prepare(ctx_id=ctx_id, det_size=(320, 320))
        logger.info("InsightFace loaded successfully.")

    def get_faces(self, frame: np.ndarray) -> List:
        """
        Returns a list of InsightFace Face objects, each with:
          .bbox   — [x1, y1, x2, y2]
          .normed_embedding — 512-d L2-normalised ArcFace embedding
          .det_score — detection confidence
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = self.app.get(rgb)
        return faces

    def cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Cosine similarity between two L2-normalised embeddings → [-1, 1]."""
        emb1 = emb1 / (np.linalg.norm(emb1) + 1e-8)
        emb2 = emb2 / (np.linalg.norm(emb2) + 1e-8)
        return float(np.dot(emb1, emb2))

    def match_embedding(
        self,
        query_emb: np.ndarray,
        registered: List[Tuple[str, np.ndarray]]
    ) -> Tuple[Optional[str], float]:
        """
        Find the best match for query_emb among registered (uuid, embedding) pairs.
        Returns (best_uuid, best_similarity) or (None, 0.0) if no match exceeds threshold.
        """
        best_uuid, best_sim = None, -1.0
        for uuid_, stored_emb in registered:
            sim = self.cosine_similarity(query_emb, stored_emb)
            if sim > best_sim:
                best_sim = sim
                best_uuid = uuid_
        if best_sim >= self.similarity_threshold:
            return best_uuid, best_sim
        return None, best_sim

    @staticmethod
    def crop_face(frame: np.ndarray, bbox, padding: float = 0.2) -> np.ndarray:
        """Return a cropped (with padding) BGR face image."""
        x1, y1, x2, y2 = map(int, bbox)
        h, w = frame.shape[:2]
        pad_x = int((x2 - x1) * padding)
        pad_y = int((y2 - y1) * padding)
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)
        return frame[y1:y2, x1:x2].copy()