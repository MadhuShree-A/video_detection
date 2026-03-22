"""
debug_detection.py — Check how many faces are detected per frame
Run: python debug_detection.py
"""

import cv2
import json
from face_detector import FaceDetector

with open('config.json') as f:
    config = json.load(f)

detector = FaceDetector(
    model_path=config['detection']['model'],
    confidence=config['detection']['confidence_threshold'],
    iou_threshold=config['detection']['iou_threshold'],
    min_face_size=config['tracking']['min_face_size']
)

cap = cv2.VideoCapture(config['camera']['video_source'])
orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
scale = 1280 / orig_w

print(f"Video: {orig_w}x{orig_h}, scale factor: {scale:.3f}")
print(f"Confidence threshold: {config['detection']['confidence_threshold']}")
print(f"Min face size: {config['tracking']['min_face_size']}")
print("-" * 60)

frame_no = 0
max_faces_seen = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_no += 1

    # Check every 10 frames
    if frame_no % 10 == 0:
        small = cv2.resize(frame, (1280, int(orig_h * scale)))
        dets = detector.detect(small)
        count = len(dets)

        if count > max_faces_seen:
            max_faces_seen = count

        if count > 0:
            sizes = [f"{int((d[2]-d[0])/scale)}x{int((d[3]-d[1])/scale)}px" for d in dets]
            print(f"Frame {frame_no:5d}: {count} face(s) → sizes: {sizes}")

cap.release()
print("-" * 60)
print(f"Max faces detected in any single frame: {max_faces_seen}")
print("Done.")
