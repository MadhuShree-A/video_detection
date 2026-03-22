"""
debug_similarity_v2.py — Uses YOLO to find 2 faces, then InsightFace for embeddings
Run: python debug_similarity_v2.py
"""

import cv2
import json
import numpy as np
from face_detector import FaceDetector

with open('config.json') as f:
    config = json.load(f)

# Load YOLO detector
detector = FaceDetector(
    model_path=config['detection']['model'],
    confidence=config['detection']['confidence_threshold'],
    iou_threshold=config['detection']['iou_threshold'],
    min_face_size=config['tracking']['min_face_size']
)

# Load InsightFace with smaller det_size so it works on crops
import insightface
from insightface.app import FaceAnalysis
app = FaceAnalysis(
    name=config['recognition']['model_name'],
    allowed_modules=["detection", "recognition"],
    providers=["CPUExecutionProvider"]
)
app.prepare(ctx_id=0, det_size=(320, 320))  # smaller size for crops

def get_embedding(crop):
    """Get ArcFace embedding from a face crop."""
    if crop is None or crop.size == 0:
        return None
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    # Resize crop to reasonable size for InsightFace
    h, w = rgb.shape[:2]
    if w < 60:
        rgb = cv2.resize(rgb, (120, int(h * 120 / w)))
    faces = app.get(rgb)
    if not faces:
        return None
    face = max(faces, key=lambda f: f.det_score)
    return face.normed_embedding

def cosine_sim(e1, e2):
    e1 = e1 / (np.linalg.norm(e1) + 1e-8)
    e2 = e2 / (np.linalg.norm(e2) + 1e-8)
    return float(np.dot(e1, e2))

cap = cv2.VideoCapture(config['camera']['video_source'])
orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
scale = 1280 / orig_w

print(f"Video: {orig_w}x{orig_h}, scale: {scale:.3f}")
print("-" * 60)

results = []

for target_frame in [770, 780, 790, 800, 810, 820, 830, 850]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ret, frame = cap.read()
    if not ret:
        continue

    small = cv2.resize(frame, (1280, int(orig_h * scale)))
    dets = detector.detect(small)

    print(f"Frame {target_frame}: {len(dets)} YOLO detection(s)")

    if len(dets) < 2:
        continue

    embeddings = []
    for i, (x1, y1, x2, y2, conf) in enumerate(dets[:2]):
        # Scale back to original resolution
        ox1 = max(0, int(x1 / scale) - 30)
        oy1 = max(0, int(y1 / scale) - 30)
        ox2 = min(orig_w, int(x2 / scale) + 30)
        oy2 = min(orig_h, int(y2 / scale) + 30)

        crop = frame[oy1:oy2, ox1:ox2].copy()

        # Save for visual check
        save_path = f"debug_face_v2_{target_frame}_p{i+1}.jpg"
        cv2.imwrite(save_path, crop)
        print(f"  Person {i+1}: crop={crop.shape}, conf={conf:.3f} → saved {save_path}")

        emb = get_embedding(crop)
        if emb is not None:
            embeddings.append(emb)
            print(f"  Person {i+1}: embedding extracted ✓ (shape={emb.shape})")
        else:
            print(f"  Person {i+1}: InsightFace couldn't extract embedding from crop ✗")

    if len(embeddings) == 2:
        sim = cosine_sim(embeddings[0], embeddings[1])
        print(f"  >>> SIMILARITY score: {sim:.4f}")
        print(f"  >>> threshold 0.60 → {'SAME ❌' if sim >= 0.60 else 'DIFFERENT ✓'}")
        print(f"  >>> threshold 0.55 → {'SAME ❌' if sim >= 0.55 else 'DIFFERENT ✓'}")
        print(f"  >>> threshold 0.50 → {'SAME ❌' if sim >= 0.50 else 'DIFFERENT ✓'}")
        print(f"  >>> threshold 0.45 → {'SAME ❌' if sim >= 0.45 else 'DIFFERENT ✓'}")
        print(f"  >>> threshold 0.40 → {'SAME ❌' if sim >= 0.40 else 'DIFFERENT ✓'}")
        results.append(sim)
        print()

    if len(results) >= 3:
        break

cap.release()

if results:
    avg_sim = sum(results) / len(results)
    print("=" * 60)
    print(f"Average similarity across {len(results)} frame(s): {avg_sim:.4f}")
    print(f"Recommended threshold to set: {avg_sim - 0.05:.4f}")
    print("=" * 60)
else:
    print("Could not extract embeddings. Check the saved debug_face_v2_*.jpg images.")
    print("If crops look wrong (wrong region), the bbox scaling needs adjustment.")
