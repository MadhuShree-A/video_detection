"""
debug_registration.py — Check why second person is not getting registered
Run: python debug_registration.py
"""

import cv2
import json
import numpy as np
from face_detector import FaceDetector
import insightface
from insightface.app import FaceAnalysis

with open('config.json') as f:
    config = json.load(f)

detector = FaceDetector(
    model_path=config['detection']['model'],
    confidence=config['detection']['confidence_threshold'],
    iou_threshold=config['detection']['iou_threshold'],
    min_face_size=config['tracking']['min_face_size']
)

# Try different det_sizes to find what works
for det_size in [(320, 320), (640, 640), (160, 160)]:
    print(f"\nTesting InsightFace det_size={det_size}")
    app = FaceAnalysis(
        name=config['recognition']['model_name'],
        allowed_modules=["detection", "recognition"],
        providers=["CPUExecutionProvider"]
    )
    app.prepare(ctx_id=0, det_size=det_size)

    cap = cv2.VideoCapture(config['camera']['video_source'])
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = 1280 / orig_w

    cap.set(cv2.CAP_PROP_POS_FRAMES, 800)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Could not read frame")
        continue

    small = cv2.resize(frame, (1280, int(orig_h * scale)))
    dets = detector.detect(small)
    print(f"  YOLO found {len(dets)} face(s)")

    for i, (x1, y1, x2, y2, conf) in enumerate(dets):
        ox1 = max(0, int(x1/scale) - 40)
        oy1 = max(0, int(y1/scale) - 40)
        ox2 = min(orig_w, int(x2/scale) + 40)
        oy2 = min(orig_h, int(y2/scale) + 40)
        crop = frame[oy1:oy2, ox1:ox2].copy()

        # Save crop
        cv2.imwrite(f"reg_debug_p{i+1}_det{det_size[0]}.jpg", crop)

        # Try getting embedding
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        # Try with resize
        for resize_w in [None, 160, 200, 256]:
            if resize_w:
                h, w = rgb.shape[:2]
                resized = cv2.resize(rgb, (resize_w, int(h * resize_w / w)))
            else:
                resized = rgb

            faces = app.get(resized)
            emb_ok = len(faces) > 0
            print(f"  Person {i+1} | crop={crop.shape[:2]} | "
                  f"resize_w={resize_w} | InsightFace found: {len(faces)} face(s) "
                  f"{'✓ embedding OK' if emb_ok else '✗ no embedding'}")
            if emb_ok:
                print(f"    det_score={faces[0].det_score:.3f}")
                break
