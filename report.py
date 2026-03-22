"""
report.py — Query the SQLite database and print visitor statistics
Usage:
    python report.py
    python report.py --events          # show all events
    python report.py --face <uuid>     # show one face's history
"""

import argparse
import sqlite3
import json
from pathlib import Path


def load_config(path="config.json"):
    with open(path) as f:
        return json.load(f)


def print_summary(conn):
    row = conn.execute("SELECT * FROM visitor_stats WHERE id=1").fetchone()
    print("\n" + "=" * 50)
    print("  VISITOR SUMMARY")
    print("=" * 50)
    if row:
        print(f"  Unique visitors : {row[1]}")
        print(f"  Last updated    : {row[2]}")

    entries = conn.execute(
        "SELECT COUNT(*) FROM events WHERE event_type='entry'"
    ).fetchone()[0]
    exits = conn.execute(
        "SELECT COUNT(*) FROM events WHERE event_type='exit'"
    ).fetchone()[0]
    print(f"  Total entries   : {entries}")
    print(f"  Total exits     : {exits}")
    print("=" * 50 + "\n")


def print_faces(conn):
    rows = conn.execute(
        "SELECT face_uuid, first_seen, last_seen, visit_count FROM faces ORDER BY first_seen"
    ).fetchall()
    print(f"\n{'UUID':<12} {'First Seen':<25} {'Last Seen':<25} {'Visits'}")
    print("-" * 75)
    for r in rows:
        print(f"{r[0]:<12} {r[1]:<25} {str(r[2]):<25} {r[3]}")
    print()


def print_events(conn, face_uuid=None):
    if face_uuid:
        rows = conn.execute(
            "SELECT * FROM events WHERE face_uuid=? ORDER BY timestamp", (face_uuid,)
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM events ORDER BY timestamp").fetchall()

    print(f"\n{'ID':<5} {'Face UUID':<12} {'Type':<8} {'Timestamp':<25} {'Frame'}")
    print("-" * 72)
    for r in rows:
        print(f"{r[0]:<5} {r[1]:<12} {r[2]:<8} {r[3]:<25} {r[5]}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Face Tracker Report")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--events", action="store_true", help="List all events")
    parser.add_argument("--faces", action="store_true", help="List all registered faces")
    parser.add_argument("--face", default=None, help="Show events for a specific face UUID")
    args = parser.parse_args()

    config = load_config(args.config)
    db_path = config["database"]["path"]

    if not Path(db_path).exists():
        print(f"Database not found: {db_path}")
        return

    conn = sqlite3.connect(db_path)
    print_summary(conn)

    if args.faces:
        print_faces(conn)
    if args.events:
        print_events(conn)
    if args.face:
        print_events(conn, args.face)
    if not (args.events or args.faces or args.face):
        print_faces(conn)

    conn.close()


if __name__ == "__main__":
    main()
