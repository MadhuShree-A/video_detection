"""
tracker.py — Lightweight centroid + IoU tracker (no external dependency needed)
Optionally integrates with DeepSort / ByteTrack if installed.
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import OrderedDict

logger = logging.getLogger(__name__)


@dataclass
class TrackedFace:
    track_id: int
    face_uuid: Optional[str]          # None until recognised/registered
    bbox: Tuple[int, int, int, int]   # x1, y1, x2, y2
    centroid: Tuple[int, int]
    disappeared: int = 0
    embedding: Optional[np.ndarray] = None
    is_registered: bool = False
    entry_logged: bool = False
    last_frame: int = 0


def _centroid(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter + 1e-6
    return inter / union


class CentroidTracker:
    """
    Simple centroid + IoU-based tracker.
    Assigns stable track IDs across frames without external libraries.
    """

    def __init__(self, max_disappeared: int = 30, iou_threshold: float = 0.3,
                 max_distance: int = 100):
        self.next_id = 0
        self.tracks: Dict[int, TrackedFace] = OrderedDict()
        self.max_disappeared = max_disappeared
        self.iou_threshold = iou_threshold
        self.max_distance = max_distance

    def update(self, detections: List[Tuple], frame_no: int) -> List[TrackedFace]:
        """
        detections: list of (x1, y1, x2, y2, conf)
        Returns the current list of active TrackedFace objects.
        """
        if not detections:
            for tid in list(self.tracks.keys()):
                self.tracks[tid].disappeared += 1
            self._prune()
            return list(self.tracks.values())

        det_bboxes = [(d[0], d[1], d[2], d[3]) for d in detections]

        if not self.tracks:
            for bbox in det_bboxes:
                self._register(bbox, frame_no)
            return list(self.tracks.values())

        track_ids = list(self.tracks.keys())
        track_bboxes = [self.tracks[t].bbox for t in track_ids]

        # Build IoU cost matrix
        matched_tracks = set()
        matched_dets = set()

        # Greedy IoU matching
        iou_matrix = np.zeros((len(track_ids), len(det_bboxes)))
        for ti, tbbox in enumerate(track_bboxes):
            for di, dbbox in enumerate(det_bboxes):
                iou_matrix[ti, di] = _iou(tbbox, dbbox)

        # Greedy assignment: highest IoU first
        pairs = []
        flat = [(iou_matrix[ti, di], ti, di)
                for ti in range(len(track_ids))
                for di in range(len(det_bboxes))]
        flat.sort(reverse=True)
        for score, ti, di in flat:
            if score < self.iou_threshold:
                break
            if ti in matched_tracks or di in matched_dets:
                continue
            matched_tracks.add(ti)
            matched_dets.add(di)
            pairs.append((ti, di))

        # Update matched
        for ti, di in pairs:
            tid = track_ids[ti]
            self.tracks[tid].bbox = det_bboxes[di]
            self.tracks[tid].centroid = _centroid(det_bboxes[di])
            self.tracks[tid].disappeared = 0
            self.tracks[tid].last_frame = frame_no

        # Increment disappeared for unmatched tracks
        for ti, tid in enumerate(track_ids):
            if ti not in matched_tracks:
                self.tracks[tid].disappeared += 1

        # Register new detections
        for di, dbbox in enumerate(det_bboxes):
            if di not in matched_dets:
                self._register(dbbox, frame_no)

        self._prune()
        return list(self.tracks.values())

    def _register(self, bbox, frame_no):
        tf = TrackedFace(
            track_id=self.next_id,
            face_uuid=None,
            bbox=bbox,
            centroid=_centroid(bbox),
            last_frame=frame_no
        )
        self.tracks[self.next_id] = tf
        self.next_id += 1

    def _prune(self):
        for tid in list(self.tracks.keys()):
            if self.tracks[tid].disappeared > self.max_disappeared:
                del self.tracks[tid]

    def get_disappeared_ids(self) -> List[int]:
        """Return track IDs that have just exceeded the disappeared threshold."""
        return [
            tid for tid, t in self.tracks.items()
            if t.disappeared == self.max_disappeared
        ]
