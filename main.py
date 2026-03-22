"""
main.py — Entry point for the Intelligent Face Tracker
Usage:
    python main.py                        # uses config.json defaults
    python main.py --source video.mp4     # override video source
    python main.py --rtsp                 # use RTSP stream from config
    python main.py --reset                # wipe DB + logs before running (fresh test)
    python main.py --reset --source video.mp4  # reset + custom source
"""

import argparse
import json
import logging
import os
import shutil
import sys
import warnings
import cv2
import time
from pathlib import Path

# Suppress noisy third-party warnings
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from logger_manager import setup_logger
from face_tracker import FaceTrackerPipeline


def load_config(path: str = "config.json") -> dict:
    with open(path) as f:
        return json.load(f)


def reset_all(config: dict):
    """Wipe database, log file, and saved face images for a clean test run."""
    db_path = config["database"]["path"]
    log_file = config["logging"]["log_file"]
    image_base = config["logging"]["image_base_dir"]

    # Delete database
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"[RESET] Deleted database: {db_path}")
    else:
        print(f"[RESET] No database found at {db_path} (skipping)")

    # Delete log file
    if os.path.exists(log_file):
        os.remove(log_file)
        print(f"[RESET] Deleted log file: {log_file}")
    else:
        print(f"[RESET] No log file found at {log_file} (skipping)")

    # Delete saved face images (entries/ and exits/ folders)
    for folder in ["entries", "exits"]:
        folder_path = Path(image_base) / folder
        if folder_path.exists():
            shutil.rmtree(folder_path)
            print(f"[RESET] Deleted image folder: {folder_path}")

    print("[RESET] Clean reset complete. Starting fresh...\n")


def run(config: dict, source: str, show: bool, save_output: bool, reset: bool = False):
    if reset:
        reset_all(config)

    log_cfg = config["logging"]
    setup_logger(log_cfg["log_file"], log_cfg.get("log_level", "INFO"))
    logger = logging.getLogger("main")

    logger.info("=" * 60)
    logger.info("Intelligent Face Tracker — Starting")
    logger.info(f"Video source : {source}")
    logger.info("=" * 60)

    # Init pipeline
    pipeline = FaceTrackerPipeline(config)

    # Open video source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error(f"Cannot open video source: {source}")
        sys.exit(1)

    fps   = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info(f"Stream opened — {width}x{height} @ {fps:.1f} fps")

    # Optional output writer
    writer = None
    if save_output:
        out_path = config["output"]["output_video_path"]
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        logger.info(f"Saving output video to {out_path}")

    frame_count = 0
    t_start = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.info("End of stream reached.")
                break

            frame_count += 1
            annotated = pipeline.process_frame(frame)

            if writer:
                writer.write(annotated)

            if show:
                # Scale down to fit screen while keeping full frame (aspect ratio preserved)
                screen_w = config["camera"].get("display_width", 1280)
                screen_h = config["camera"].get("display_height", 720)
                scale_w = screen_w / width
                scale_h = screen_h / height
                scale_disp = min(scale_w, scale_h)  # fit entire frame on screen
                disp_w = int(width * scale_disp)
                disp_h = int(height * scale_disp)
                display_frame = cv2.resize(annotated, (disp_w, disp_h))

                cv2.namedWindow("Face Tracker", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Face Tracker", disp_w, disp_h)
                cv2.imshow("Face Tracker", display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    logger.info("User quit.")
                    break

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")

    finally:
        elapsed = time.time() - t_start
        stats = pipeline.get_stats()

        logger.info("=" * 60)
        logger.info(f"Session complete — {frame_count} frames in {elapsed:.1f}s "
                    f"({frame_count/max(elapsed,1):.1f} fps)")
        logger.info(f"Unique visitors  : {stats['unique_visitors']}")
        logger.info(f"Total entries    : {stats['total_entries']}")
        logger.info(f"Total exits      : {stats['total_exits']}")
        logger.info("=" * 60)

        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        pipeline.close()

    return stats


def main():
    parser = argparse.ArgumentParser(description="Intelligent Face Tracker")
    parser.add_argument("--config", default="config.json", help="Path to config.json")
    parser.add_argument("--source", default=None, help="Override video source path")
    parser.add_argument("--rtsp", action="store_true", help="Use RTSP stream from config")
    parser.add_argument("--no-display", action="store_true", help="Disable GUI window")
    parser.add_argument("--save", action="store_true", help="Save annotated output video")
    parser.add_argument("--reset", action="store_true", help="Wipe DB + logs before running (fresh test)")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.rtsp:
        source = config["camera"]["rtsp_url"]
    elif args.source:
        source = args.source
    else:
        source = config["camera"]["video_source"]

    show = config["output"]["show_display"] and not args.no_display
    save = args.save or config["output"]["save_output_video"]

    run(config, source, show, save, reset=args.reset)


if __name__ == "__main__":
    main()