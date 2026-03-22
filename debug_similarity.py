"""
debug_similarity.py — Extract faces from frames where 2 people are visible
and print the actual cosine similarity score between them.
Run: python debug_similarity.py
"""

import cv2
import json
import numpy as np
from face_detector import FaceDetector
from face_recognition_engine import FaceRecognitionEngine

with open('config.json') as f:
    config = json.load(f)

detector = FaceDetector(
    model_path=config['detection']['model'],
    confidence=config['detection']['confidence_threshold'],
    iou_threshold=config['detection']['iou_threshold'],
    min_face_size=config['tracking']['min_face_size']
)

recognizer = FaceRecognitionEngine(
    model_name=config['recognition']['model_name'],
    ctx_id=config['recognition']['ctx_id'],
    det_size=tuple(config['recognition']['det_size']),
    similarity_threshold=0.0  # disable threshold — we want raw scores
)

cap = cv2.VideoCapture(config['camera']['video_source'])
orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
scale = 1280 / orig_w

print(f"Video: {orig_w}x{orig_h}")
print("Looking for frames with 2 faces to measure similarity...")
print("-" * 60)

checked = 0

for target_frame in [770, 780, 790, 800, 810, 820, 830, 850]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ret, frame = cap.read()
    if not ret:
        continue

    small = cv2.resize(frame, (1280, int(orig_h * scale)))
    dets = detector.detect(small)

    if len(dets) < 2:
        print(f"Frame {target_frame}: only {len(dets)} face(s) detected, skipping")
        continue

    # Get embeddings for each detected face
    embeddings = []
    for i, (x1, y1, x2, y2, conf) in enumerate(dets):
        # Scale back to original
        ox1, oy1 = int(x1/scale), int(y1/scale)
        ox2, oy2 = int(x2/scale), int(y2/scale)

        # Add padding
        pad = 20
        h, w = frame.shape[:2]
        ox1 = max(0, ox1 - pad)
        oy1 = max(0, oy1 - pad)
        ox2 = min(w, ox2 + pad)
        oy2 = min(h, oy2 + pad)

        crop = frame[oy1:oy2, ox1:ox2]
        if crop.size == 0:
            continue

        # Save crop for visual inspection
        cv2.imwrite(f"debug_face_{target_frame}_person{i+1}.jpg", crop)

        faces = recognizer.get_faces(crop)
        if faces:
            face = max(faces, key=lambda f: f.det_score)
            embeddings.append((i+1, face.normed_embedding, conf))
            print(f"Frame {target_frame} Person {i+1}: bbox=({x1},{y1},{x2},{y2}) conf={conf:.3f}")

    if len(embeddings) >= 2:
        emb1 = embeddings[0][1]
        emb2 = embeddings[1][1]
        sim = recognizer.cosine_similarity(emb1, emb2)
        print(f"  >>> SIMILARITY between Person 1 and Person 2: {sim:.4f}")
        print(f"  >>> At threshold 0.55 they would be: {'SAME person ❌' if sim >= 0.55 else 'DIFFERENT people ✓'}")
        print(f"  >>> At threshold 0.45 they would be: {'SAME person ❌' if sim >= 0.45 else 'DIFFERENT people ✓'}")
        print(f"  >>> At threshold 0.35 they would be: {'SAME person ❌' if sim >= 0.35 else 'DIFFERENT people ✓'}")
        print()
        checked += 1

    if checked >= 3:
        break

cap.release()
print("-" * 60)
print("Face crops saved as debug_face_*.jpg — check them to confirm correct cropping.")
print("Done.")
