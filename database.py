"""
database.py — SQLite database manager for face tracker
"""

import sqlite3
import numpy as np
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Tuple, Dict

logger = logging.getLogger(__name__)


class DatabaseManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()
        logger.info(f"Database initialized at {db_path}")

    def _create_tables(self):
        cursor = self.conn.cursor()

        # Registered faces
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS faces (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                face_uuid   TEXT    NOT NULL UNIQUE,
                first_seen  TEXT    NOT NULL,
                last_seen   TEXT,
                visit_count INTEGER DEFAULT 1,
                embedding   BLOB    NOT NULL,
                thumbnail   TEXT
            )
        """)

        # Entry / Exit events
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                face_uuid   TEXT    NOT NULL,
                event_type  TEXT    NOT NULL CHECK(event_type IN ('entry','exit')),
                timestamp   TEXT    NOT NULL,
                image_path  TEXT,
                frame_no    INTEGER,
                FOREIGN KEY (face_uuid) REFERENCES faces(face_uuid)
            )
        """)

        # Visitor counter (single-row summary)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS visitor_stats (
                id              INTEGER PRIMARY KEY CHECK(id = 1),
                unique_visitors INTEGER DEFAULT 0,
                last_updated    TEXT
            )
        """)
        cursor.execute("""
            INSERT OR IGNORE INTO visitor_stats (id, unique_visitors, last_updated)
            VALUES (1, 0, ?)
        """, (datetime.now().isoformat(),))

        self.conn.commit()

    # ------------------------------------------------------------------ faces

    def register_face(self, face_uuid: str, embedding: np.ndarray, thumbnail_path: str = None) -> bool:
        try:
            emb_blob = embedding.tobytes()
            now = datetime.now().isoformat()
            self.conn.execute("""
                INSERT INTO faces (face_uuid, first_seen, last_seen, embedding, thumbnail)
                VALUES (?, ?, ?, ?, ?)
            """, (face_uuid, now, now, emb_blob, thumbnail_path))
            self.conn.execute("""
                UPDATE visitor_stats SET unique_visitors = unique_visitors + 1, last_updated = ?
                WHERE id = 1
            """, (now,))
            self.conn.commit()
            logger.info(f"Registered new face: {face_uuid}")
            return True
        except sqlite3.IntegrityError:
            logger.warning(f"Face {face_uuid} already registered.")
            return False

    def update_face_last_seen(self, face_uuid: str):
        now = datetime.now().isoformat()
        self.conn.execute("""
            UPDATE faces SET last_seen = ?, visit_count = visit_count + 1
            WHERE face_uuid = ?
        """, (now, face_uuid))
        self.conn.commit()

    def get_all_embeddings(self) -> List[Tuple[str, np.ndarray]]:
        cursor = self.conn.execute("SELECT face_uuid, embedding FROM faces")
        results = []
        for row in cursor.fetchall():
            emb = np.frombuffer(row["embedding"], dtype=np.float32)
            results.append((row["face_uuid"], emb))
        return results

    def get_face(self, face_uuid: str) -> Optional[Dict]:
        row = self.conn.execute(
            "SELECT * FROM faces WHERE face_uuid = ?", (face_uuid,)
        ).fetchone()
        return dict(row) if row else None

    def face_exists(self, face_uuid: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM faces WHERE face_uuid = ?", (face_uuid,)
        ).fetchone()
        return row is not None

    # ----------------------------------------------------------------- events

    def log_event(self, face_uuid: str, event_type: str,
                  image_path: str = None, frame_no: int = None):
        now = datetime.now().isoformat()
        self.conn.execute("""
            INSERT INTO events (face_uuid, event_type, timestamp, image_path, frame_no)
            VALUES (?, ?, ?, ?, ?)
        """, (face_uuid, event_type, now, image_path, frame_no))
        self.conn.commit()
        logger.info(f"Event logged: {event_type} — {face_uuid}")

    def get_events(self, face_uuid: str = None) -> List[Dict]:
        if face_uuid:
            rows = self.conn.execute(
                "SELECT * FROM events WHERE face_uuid = ? ORDER BY timestamp", (face_uuid,)
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM events ORDER BY timestamp"
            ).fetchall()
        return [dict(r) for r in rows]

    # --------------------------------------------------------------- counters

    def get_unique_visitor_count(self) -> int:
        row = self.conn.execute(
            "SELECT unique_visitors FROM visitor_stats WHERE id = 1"
        ).fetchone()
        return row["unique_visitors"] if row else 0

    def get_summary(self) -> Dict:
        visitors = self.get_unique_visitor_count()
        entries = self.conn.execute(
            "SELECT COUNT(*) AS c FROM events WHERE event_type='entry'"
        ).fetchone()["c"]
        exits = self.conn.execute(
            "SELECT COUNT(*) AS c FROM events WHERE event_type='exit'"
        ).fetchone()["c"]
        return {
            "unique_visitors": visitors,
            "total_entries": entries,
            "total_exits": exits,
        }

    def close(self):
        self.conn.close()
        logger.info("Database connection closed.")
