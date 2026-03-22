# Intelligent Face Tracker with Auto-Registration & Unique Visitor Counting

> An AI-driven real-time face detection, recognition, tracking and visitor counting system built for CCTV / RTSP video streams.

---

## Demo Video

> 📹 **[(https://www.loom.com/share/36aaa072a4974fbba9c601c234e08a60)](#)**
>
> *(Replace this link with your actual Loom or YouTube recording before submission)*

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [AI Planning Document](#ai-planning-document)
3. [Setup Instructions](#setup-instructions)
4. [Configuration Reference](#configuration-reference)
5. [Running the System](#running-the-system)
6. [Project Structure](#project-structure)
7. [Database Schema](#database-schema)
8. [Assumptions Made](#assumptions-made)
9. [Known Limitations](#known-limitations)

---

## Architecture Overview

```
+---------------------------------------------------------------------------+
|                          VIDEO INPUT LAYER                                |
|            Video File (.mp4)  --OR--  RTSP Live Stream                   |
+---------------------------------------------------------------------------+
                              | raw frames (e.g. 2688x1520)
                              v
+---------------------------------------------------------------------------+
|                        PRE-PROCESSING                                     |
|    Resize frame to 1280px wide (preserving aspect ratio)                 |
|    Reason: YOLO performs best on ~1280px input for small faces            |
+---------------------------------------------------------------------------+
                              | resized frame
                              v
+---------------------------------------------------------------------------+
|               FACE DETECTION  (YOLOv8n-face)                             |
|   - Runs every N frames (configurable: detection_skip_frames)            |
|   - Returns bounding boxes + confidence scores                           |
|   - Filters boxes below min_face_size threshold                          |
|   - Scales bboxes back to original resolution                            |
+---------------------------------------------------------------------------+
                              | bboxes [(x1,y1,x2,y2,conf), ...]
                              v
+---------------------------------------------------------------------------+
|               CENTROID + IoU TRACKER                                      |
|   - Assigns stable track IDs across frames                               |
|   - Greedy IoU matching between current and previous bboxes              |
|   - Increments "disappeared" counter for unmatched tracks                |
|   - Prunes tracks exceeding max_disappeared_frames                       |
|   - Triggers EXIT event when track is pruned                             |
+---------------------------------------------------------------------------+
                              | TrackedFace objects (track_id, bbox, uuid)
                              v
+---------------------------------------------------------------------------+
|         IDENTITY RESOLUTION  (InsightFace buffalo_l / ArcFace)           |
|                                                                           |
|   For each UNREGISTERED track:                                           |
|     1. Crop face from original frame (with padding)                      |
|     2. Upscale crop to min 80px for InsightFace                          |
|     3. Extract 512-d ArcFace embedding                                   |
|     4. Cosine similarity vs all known embeddings (in-memory cache)       |
|                                                                           |
|     similarity >= threshold?                                             |
|          YES -> Re-identify: assign existing UUID                        |
|          NO  -> Auto-register: create new UUID, store embedding in DB    |
+---------------------------------------------------------------------------+
                              | face_uuid assigned to track
                              v
+---------------------------------------------------------------------------+
|                      LOGGING & STORAGE                                    |
|                                                                           |
|   +------------------+  +------------------+  +----------------------+   |
|   |   SQLite DB       |  |   events.log     |  |   Face Images        |   |
|   | - faces table    |  |  (append-only)   |  |  logs/entries/       |   |
|   | - events table   |  |  Human-readable  |  |  logs/exits/         |   |
|   | - visitor_stats  |  |  event timeline  |  |  YYYY-MM-DD/         |   |
|   +------------------+  +------------------+  +----------------------+   |
+---------------------------------------------------------------------------+
                              |
                              v
+---------------------------------------------------------------------------+
|                   ANNOTATED DISPLAY OUTPUT                                |
|   - Bounding boxes with UUID labels on detected faces                   |
|   - HUD (bottom-left): Unique Visitors / Entries / Exits / Frame        |
|   - Window scaled to fit screen (full frame, no cropping)               |
+---------------------------------------------------------------------------+
```

---

## AI Planning Document

### Problem Statement
Build a system that watches a video stream, automatically identifies every unique person who appears, counts them accurately without double-counting, and logs every entry and exit with a timestamped face image.

### Key Design Decisions

#### 1. Why YOLOv8 for Detection?
YOLOv8 provides the best balance of speed and accuracy for real-time face detection. The nano model (`yolov8n-face`) runs at 7-11 FPS on CPU for 2688x1520 video. The face-specific fine-tuned variant detects small and partially occluded faces better than the general object model.

#### 2. Why InsightFace / ArcFace for Recognition?
ArcFace embeddings (512-dimensional) produce highly discriminative face representations trained with additive angular margin loss. Unlike `face_recognition` (dlib-based), InsightFace achieves near-human accuracy on LFW benchmarks. The `buffalo_l` model pack includes both detection and recognition in one download.

#### 3. Why a Custom Centroid + IoU Tracker?
DeepSort and ByteTrack require additional dependencies (`lap`, `filterpy`) that are complex to install on Windows. A custom IoU-based tracker achieves the same goal with zero extra dependencies. For low crowd density CCTV, it performs equivalently.

#### 4. Re-identification Strategy
Cosine similarity on ArcFace embeddings is the gold standard for face re-ID. A threshold of 0.45 was found optimal through empirical testing — debug scripts measured ~0.097 similarity between two different people in the test video, well below any reasonable threshold.

#### 5. High-Resolution CCTV Handling
The test video is 2688x1520. Feeding this directly to YOLO (which internally resizes to 640x640) made faces too small to detect. The solution: resize to 1280px wide before YOLO detection, then scale bounding boxes back to original coordinates.

#### 6. Embedding Extraction on Small Crops
InsightFace's internal detector (SCRFD) struggles with very small face crops. Fix: ensure every crop is upscaled to at least 80px wide before passing to InsightFace, and use `det_size=(320, 320)` which suits small crop inputs better than the default 640x640.

### Unique Visitor Counting Logic
```
New face detected
    -> ArcFace embedding extracted
    -> Cosine similarity vs all stored embeddings
    -> Max similarity < threshold?
        YES: New person -> UUID assigned -> DB registered -> visitor_count++
        NO:  Known person -> existing UUID assigned -> visitor_count unchanged
```
The count is durable across restarts (stored in SQLite `visitor_stats` table).

### Entry / Exit Detection
- **Entry**: Fired once per track on the first frame a UUID is successfully assigned
- **Exit**: Fired when a track's `disappeared` counter reaches `max_disappeared_frames`
- Each event saves a cropped face image + DB metadata record

---

## Setup Instructions

### Prerequisites
- Python 3.9 to 3.12
- Windows 10/11 (or Linux/macOS)
- **Microsoft C++ Build Tools** (Windows only - required to compile InsightFace)
  - Download: https://visualstudio.microsoft.com/visual-cpp-build-tools/
  - Select workload: **Desktop development with C++**

### Step 1 - Create Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### Step 2 - Install Dependencies
```bash
pip install -r requirements.txt
```

> For GPU users: Replace `onnxruntime` with `onnxruntime-gpu`. Do NOT install both.

### Step 3 - Download YOLO Face Model
```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n-face.pt')"
```

The `buffalo_l` InsightFace model downloads automatically on first run to `~/.insightface/models/`.

### Step 4 - Verify Installation
```bash
python -c "
from ultralytics import YOLO
import insightface, cv2, sqlite3, numpy
print('All OK | InsightFace:', insightface.__version__, '| OpenCV:', cv2.__version__)
"
```

### Step 5 - Configure Video Source
Edit `config.json` and set your video path:
```json
"video_source": "D:/your_video.mp4"
```
Or for webcam: `"video_source": 0`

---

## Configuration Reference

### Sample `config.json`

```json
{
  "detection": {
    "model": "yolov8n-face.pt",
    "confidence_threshold": 0.30,
    "detection_skip_frames": 1,
    "iou_threshold": 0.45
  },
  "recognition": {
    "model_name": "buffalo_l",
    "embedding_similarity_threshold": 0.45,
    "det_size": [1280, 1280],
    "ctx_id": 0
  },
  "tracking": {
    "max_disappeared_frames": 60,
    "max_tracking_distance": 150,
    "min_face_size": 15
  },
  "database": {
    "path": "data/face_tracker.db"
  },
  "logging": {
    "log_file": "logs/events.log",
    "image_base_dir": "logs",
    "log_level": "INFO"
  },
  "camera": {
    "video_source": "D:/your_video.mp4",
    "rtsp_url": "rtsp://username:password@ip:port/stream",
    "use_rtsp": false,
    "display_width": 1280,
    "display_height": 720
  },
  "output": {
    "show_display": true,
    "save_output_video": false,
    "output_video_path": "output/tracked_output.mp4"
  }
}
```

### Key Parameters

| Parameter | Default | Description |
|---|---|---|
| `detection_skip_frames` | `1` | Frames to skip between YOLO cycles. Higher = faster but may miss faces. |
| `confidence_threshold` | `0.30` | YOLO min confidence. Lower = more detections (and false positives). |
| `embedding_similarity_threshold` | `0.45` | ArcFace cosine cutoff. Above = same person. Tune per camera. |
| `max_disappeared_frames` | `60` | Frames missing before exit logged (60 = 2.4s at 25fps). |
| `min_face_size` | `15` | Minimum face bbox size in pixels. |
| `ctx_id` | `0` | GPU device ID. Use `0` for GPU, `-1` for CPU only. |

---

## Running the System

```bash
# Normal run (data accumulates across sessions)
python main.py

# Fresh test run (wipes DB + logs before starting)
python main.py --reset

# Specific video file
python main.py --reset --source D:/your_video.mp4

# Webcam
python main.py --reset --source 0

# RTSP live stream
python main.py --reset --rtsp

# Headless (no display window)
python main.py --no-display

# Save annotated output video
python main.py --save

# View stats after run
python report.py
python report.py --faces
python report.py --events
python report.py --face <uuid>
```

Press **Q** or **Esc** to quit the display window.

---

## Project Structure

```
video_detection/
|
+-- main.py                     # Entry point - video loop, CLI args
+-- face_tracker.py             # Core pipeline (detect->recognise->track->log)
+-- face_detector.py            # YOLOv8 face detection wrapper
+-- face_recognition_engine.py  # InsightFace ArcFace embeddings + matching
+-- tracker.py                  # Custom centroid + IoU tracker
+-- database.py                 # SQLite3 manager (faces, events, stats)
+-- logger_manager.py           # events.log writer + face image saver
+-- report.py                   # CLI stats and query tool
+-- config.json                 # All tunable parameters
+-- requirements.txt            # Python dependencies
|
+-- data/
|   +-- face_tracker.db         # SQLite database (auto-created)
|
+-- logs/
|   +-- events.log              # Append-only human-readable event log
|   +-- entries/
|   |   +-- YYYY-MM-DD/
|   |       +-- <uuid>_<timestamp>.jpg
|   +-- exits/
|       +-- YYYY-MM-DD/
|           +-- <uuid>_<timestamp>.jpg
|
+-- output/
    +-- tracked_output.mp4      # Annotated video (if --save used)
```

---

## Database Schema

### `faces`
| Column | Type | Description |
|---|---|---|
| `id` | INTEGER | Auto primary key |
| `face_uuid` | TEXT | Unique 8-char identifier |
| `first_seen` | TEXT | ISO timestamp of first detection |
| `last_seen` | TEXT | ISO timestamp of most recent detection |
| `visit_count` | INTEGER | Times this face has been re-identified |
| `embedding` | BLOB | 512-d ArcFace embedding (binary float32) |
| `thumbnail` | TEXT | Path to first captured face image |

### `events`
| Column | Type | Description |
|---|---|---|
| `id` | INTEGER | Auto primary key |
| `face_uuid` | TEXT | FK to faces.face_uuid |
| `event_type` | TEXT | `entry` or `exit` |
| `timestamp` | TEXT | ISO timestamp |
| `image_path` | TEXT | Path to cropped face image |
| `frame_no` | INTEGER | Source video frame number |

### `visitor_stats`
| Column | Type | Description |
|---|---|---|
| `unique_visitors` | INTEGER | Total unique faces ever registered |
| `last_updated` | TEXT | Timestamp of last increment |

---

## Assumptions Made

1. **Single camera per instance** - processes one video stream at a time.
2. **Face must be visible** - people with backs to camera or face coverings are not counted.
3. **Static camera** - IoU tracker assumes a stationary camera.
4. **25 FPS video** - exit threshold of 60 frames = 2.4 seconds. Adjust for other frame rates.
5. **Threshold is video-specific** - 0.45 was tuned for this CCTV footage. Re-tune for different cameras using `debug_similarity_v2.py`.
6. **SQLite for storage** - suitable for single-node deployment. Replace with PostgreSQL for multi-server production use.
7. **CPU inference by default** - GPU with `onnxruntime-gpu` gives 3-5x FPS improvement.
8. **`--reset` for fresh tests** - DB persists by design for real deployment; use `--reset` for testing.

---

## Known Limitations

- **Side-profile faces**: InsightFace requires roughly frontal face. Extreme side profiles may fail embedding extraction.
- **Occlusion**: Masks, hats, or hands covering faces prevent detection.
- **Speed**: ~5-11 FPS on CPU for 2688x1520. GPU recommended for real-time 25fps.
- **Very small faces**: Faces below ~30px in detection frame may be missed by YOLO.

---

## Tech Stack

| Module | Technology |
|---|---|
| Face Detection | YOLOv8n-face (Ultralytics) |
| Face Recognition | InsightFace `buffalo_l` (ArcFace 512-d) |
| Tracking | Custom Centroid + IoU Tracker |
| Backend | Python 3.12 |
| Database | SQLite3 |
| Configuration | JSON (config.json) |
| Logging | Python logging + filesystem |
| Display | OpenCV with HUD overlay |
| Camera Input | Video file / RTSP stream |

---

*This project is a part of a hackathon run by https://katomaran.com*
