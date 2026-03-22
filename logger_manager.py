"""
logger_manager.py — Filesystem + log-file event logger for face tracker
"""

import logging
import os
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path


def setup_logger(log_file: str, log_level: str = "INFO") -> logging.Logger:
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    level = getattr(logging, log_level.upper(), logging.INFO)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    root = logging.getLogger()
    root.setLevel(level)

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    root.addHandler(ch)

    return root


class EventLogger:
    """
    Saves cropped face images to a structured directory and
    writes human-readable event entries to events.log.
    """

    def __init__(self, base_dir: str, log_file: str):
        self.base_dir = Path(base_dir)
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self._logger = logging.getLogger("EventLogger")

    def _date_dir(self, event_type: str) -> Path:
        today = datetime.now().strftime("%Y-%m-%d")
        folder = self.base_dir / event_type / today
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def save_face_image(self, face_crop: np.ndarray,
                        face_uuid: str, event_type: str) -> str:
        """
        Save a cropped face image and return its path.
        event_type: 'entries' | 'exits'
        """
        folder = self._date_dir(event_type)
        ts = datetime.now().strftime("%H%M%S_%f")
        filename = f"{face_uuid}_{ts}.jpg"
        filepath = folder / filename
        cv2.imwrite(str(filepath), face_crop)
        self._logger.debug(f"Saved face image: {filepath}")
        return str(filepath)

    def log_entry(self, face_uuid: str, image_path: str, frame_no: int):
        self._write(f"ENTRY  | face={face_uuid} | frame={frame_no} | img={image_path}")

    def log_exit(self, face_uuid: str, image_path: str, frame_no: int):
        self._write(f"EXIT   | face={face_uuid} | frame={frame_no} | img={image_path}")

    def log_registration(self, face_uuid: str, frame_no: int):
        self._write(f"REGISTER | face={face_uuid} | frame={frame_no}")

    def log_recognition(self, face_uuid: str, similarity: float, frame_no: int):
        self._write(
            f"RECOGNIZE | face={face_uuid} | similarity={similarity:.4f} | frame={frame_no}"
        )

    def log_embedding(self, face_uuid: str):
        self._write(f"EMBED  | face={face_uuid}")

    def log_tracking(self, face_uuid: str, track_id: int, frame_no: int):
        self._write(
            f"TRACK  | face={face_uuid} | track_id={track_id} | frame={frame_no}"
        )

    def _write(self, message: str):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        line = f"{ts} | {message}\n"
        with open(self.log_file, "a") as f:
            f.write(line)
        self._logger.info(message)