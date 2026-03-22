"""
face_tracker.py — Core pipeline: Detection → Recognition → Tracking → Logging
"""

import logging
import uuid
import cv2
import numpy as np
from typing import Optional

from face_detector import FaceDetector
from face_recognition_engine import FaceRecognitionEngine
from tracker import CentroidTracker
from database import DatabaseManager
from logger_manager import EventLogger

logger = logging.getLogger(__name__)


class FaceTrackerPipeline:
    def __init__(self, config: dict):
        det_cfg  = config["detection"]
        rec_cfg  = config["recognition"]
        trk_cfg  = config["tracking"]
        db_cfg   = config["database"]
        log_cfg  = config["logging"]

        # Sub-systems
        self.detector = FaceDetector(
            model_path=det_cfg.get("model", "yolov8n-face.pt"),
            confidence=det_cfg["confidence_threshold"],
            iou_threshold=det_cfg["iou_threshold"],
            min_face_size=trk_cfg["min_face_size"],
        )
        self.recognizer = FaceRecognitionEngine(
            model_name=rec_cfg["model_name"],
            ctx_id=rec_cfg["ctx_id"],
            det_size=tuple(rec_cfg["det_size"]),
            similarity_threshold=rec_cfg["embedding_similarity_threshold"],
        )
        self.tracker = CentroidTracker(
            max_disappeared=trk_cfg["max_disappeared_frames"],
            max_distance=trk_cfg["max_tracking_distance"],
        )
        self.db = DatabaseManager(db_cfg["path"])
        self.event_logger = EventLogger(
            base_dir=log_cfg["image_base_dir"],
            log_file=log_cfg["log_file"],
        )

        self.detection_skip = det_cfg["detection_skip_frames"]
        self.frame_no = 0

        # In-memory cache of embeddings to avoid repeated DB queries
        self._emb_cache: list = []   # list of (uuid, np.ndarray)
        self._refresh_cache()

    def _refresh_cache(self):
        self._emb_cache = self.db.get_all_embeddings()
        logger.debug(f"Embedding cache refreshed: {len(self._emb_cache)} entries")

    # -------------------------------------------------------------- public API

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process one video frame. Returns annotated frame.
        Resizes high-res frames before detection so faces are large enough to detect.
        """
        self.frame_no += 1
        run_detection = (self.frame_no % (self.detection_skip + 1) == 0)

        # Downscale to 1280px wide for detection (handles high-res CCTV like 2688x1520)
        orig_h, orig_w = frame.shape[:2]
        detect_w = 1280
        scale = detect_w / orig_w
        detect_h = int(orig_h * scale)
        small_frame = cv2.resize(frame, (detect_w, detect_h))

        if run_detection:
            raw_detections = self.detector.detect(small_frame)
            # Scale bounding boxes back to original frame coordinates
            detections = []
            for (x1, y1, x2, y2, conf) in raw_detections:
                detections.append((
                    int(x1 / scale), int(y1 / scale),
                    int(x2 / scale), int(y2 / scale),
                    conf
                ))
        else:
            detections = []

        active_tracks = self.tracker.update(detections, self.frame_no)

        # Handle exits for disappearing tracks
        disappeared_ids = self.tracker.get_disappeared_ids()
        for tid in disappeared_ids:
            track = self.tracker.tracks.get(tid)
            if track and track.face_uuid and track.entry_logged:
                self._handle_exit(frame, track)

        # Process each active track
        if run_detection:
            for track in active_tracks:
                if not track.is_registered:
                    self._identify_and_register(frame, track)
                else:
                    self.event_logger.log_tracking(
                        track.face_uuid, track.track_id, self.frame_no
                    )

        # Annotate frame
        annotated = self._annotate(frame, active_tracks)
        return annotated

    def get_stats(self) -> dict:
        return self.db.get_summary()

    def close(self):
        self.db.close()
        logger.info("Pipeline shut down.")

    # ----------------------------------------------------------- internal logic

    def _identify_and_register(self, frame: np.ndarray, track):
        crop = FaceRecognitionEngine.crop_face(frame, track.bbox, padding=0.3)
        if crop.size == 0:
            return

        # Ensure crop is large enough for InsightFace (min 80px wide)
        ch, cw = crop.shape[:2]
        if cw < 80:
            scale_up = 80 / cw
            crop = cv2.resize(crop, (80, int(ch * scale_up)))

        # Get InsightFace embedding
        faces = self.recognizer.get_faces(crop)
        if not faces:
            logger.debug(f"InsightFace found no face in crop for track {track.track_id}, "
                         f"crop size={crop.shape[:2]}")
            return

        face = max(faces, key=lambda f: f.det_score)
        embedding = face.normed_embedding  # 512-d float32
        logger.debug(f"Track {track.track_id}: embedding extracted, det_score={face.det_score:.3f}")

        # Try to match against known faces
        matched_uuid, similarity = self.recognizer.match_embedding(
            embedding, self._emb_cache
        )

        if matched_uuid:
            # Known face — re-identification
            track.face_uuid = matched_uuid
            track.embedding = embedding
            track.is_registered = True
            self.db.update_face_last_seen(matched_uuid)
            self.event_logger.log_recognition(matched_uuid, similarity, self.frame_no)
        else:
            # New face — auto-register
            new_uuid = str(uuid.uuid4())[:8]
            track.face_uuid = new_uuid
            track.embedding = embedding
            track.is_registered = True

            img_path = self.event_logger.save_face_image(crop, new_uuid, "entries")
            self.db.register_face(new_uuid, embedding, img_path)
            self.event_logger.log_registration(new_uuid, self.frame_no)
            self.event_logger.log_embedding(new_uuid)
            self._refresh_cache()

        # Log entry event (once per track)
        if not track.entry_logged:
            img_path = self.event_logger.save_face_image(crop, track.face_uuid, "entries")
            self.db.log_event(track.face_uuid, "entry", img_path, self.frame_no)
            self.event_logger.log_entry(track.face_uuid, img_path, self.frame_no)
            track.entry_logged = True

    def _handle_exit(self, frame: np.ndarray, track):
        crop = FaceRecognitionEngine.crop_face(frame, track.bbox)
        img_path = self.event_logger.save_face_image(
            crop if crop.size > 0 else np.zeros((64, 64, 3), np.uint8),
            track.face_uuid, "exits"
        )
        self.db.log_event(track.face_uuid, "exit", img_path, self.frame_no)
        self.event_logger.log_exit(track.face_uuid, img_path, self.frame_no)

    def _annotate(self, frame: np.ndarray, tracks) -> np.ndarray:
        out = frame.copy()
        stats = self.db.get_summary()

        for track in tracks:
            if track.disappeared > 0:
                continue
            x1, y1, x2, y2 = track.bbox
            color = (0, 255, 0) if track.is_registered else (0, 165, 255)
            label = track.face_uuid if track.face_uuid else f"ID:{track.track_id}"

            # Bounding box
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)

            # Label background
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.7, 2)
            cv2.rectangle(out, (x1, y1 - th - 12), (x1 + tw + 8, y1), color, -1)
            cv2.putText(out, label, (x1 + 4, y1 - 6),
                        cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 2)

        # HUD — semi-transparent background
        h, w = out.shape[:2]
        hud_w, hud_h = 420, 110
        overlay = out.copy()
        cv2.rectangle(overlay, (0, 0), (hud_w, hud_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, out, 0.3, 0, out)

        # HUD border
        cv2.rectangle(out, (0, 0), (hud_w, hud_h), (0, 200, 100), 2)

        # HUD text
        cv2.putText(out, f"Unique Visitors : {stats['unique_visitors']}",
                    (12, 35), cv2.FONT_HERSHEY_DUPLEX, 0.85, (0, 255, 150), 2)
        cv2.putText(out, f"Entries : {stats['total_entries']}     Exits : {stats['total_exits']}",
                    (12, 68), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 220, 100), 2)
        cv2.putText(out, f"Frame : {self.frame_no}",
                    (12, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

        return out