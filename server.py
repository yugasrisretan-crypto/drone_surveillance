"""
server.py
=========
FastAPI Command Center Server — Autonomous Drone Campus Surveillance
High-Speed 3-Thread Native 30 FPS Architecture.

Key Endpoints:
  - GET  /video_feed     -> Native 30 FPS MJPEG Stream (multipart/x-mixed-replace)
  - WS   /ws/telemetry    -> Lightweight JSON telemetry sent once every 200ms
  - WS   /ws/stream       -> Backwards-compatibility alias for telemetry
  - POST /api/enroll_live -> 1-Click Live Self-Calibration (<1s reload)
  - POST /api/enroll      -> Upload ID photo and rebuild index
  - POST /api/source      -> Switch live source ('webcam' or 'demo')
  - GET  /api/status      -> Server and biometric health status
  - GET  /api/alerts      -> Historical alerts from SQLite
"""

from __future__ import annotations

import os
import sys
import subprocess
from pathlib import Path

# ---------------------------------------------------------------------------
# Auto-dispatch to project .venv if invoked with system python
# ---------------------------------------------------------------------------
_venv_python = Path(__file__).resolve().parent / ".venv" / "Scripts" / "python.exe"
if _venv_python.exists() and Path(sys.executable).resolve() != _venv_python.resolve():
    sys.exit(subprocess.call([str(_venv_python)] + sys.argv))

os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import asyncio
import json
import logging
import sqlite3
import threading
import time
import traceback
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from vision_engine import (
    DroneSurveillanceEngine,
    SNAPSHOT_DIR,
    ThreadedCamera,
    TrackStatus,
)

# ---------------------------------------------------------------------------
# Paths & Settings
# ---------------------------------------------------------------------------
ROOT         = Path(__file__).parent
VIDEO_PATH   = ROOT / "data" / "videos" / "campus_drone.mp4"
FAISS_PATH   = ROOT / "storage" / "campus.faiss"
DB_PATH      = ROOT / "storage" / "campus.db"
ID_CARDS_DIR = ROOT / "data" / "id_cards"
BADGE_TEMPLATE_PATH = ROOT / "storage" / "badge_template.jpg"
STATIC_DIR   = ROOT / "static"

SOURCE_WEBCAM = "webcam"
SOURCE_DEMO   = "demo"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("server")

# ---------------------------------------------------------------------------
# Global State & Primitives
# ---------------------------------------------------------------------------
engine         = DroneSurveillanceEngine()
camera: Optional[ThreadedCamera] = None
shutdown_event = threading.Event()
reload_lock    = threading.Lock()

_source_lock       = threading.Lock()
_current_source    = SOURCE_WEBCAM
_source_changed    = threading.Event()

_latest_annotated_jpeg: Optional[bytes] = None
_latest_jpeg_lock = threading.Lock()
_new_jpeg_event   = threading.Event()

_latest_raw_frame: Optional[np.ndarray] = None
_raw_frame_lock   = threading.Lock()

_fps_counter       = 0.0
_prev_statuses: Dict[int, str] = {}
_event_loop: Optional[asyncio.AbstractEventLoop] = None


# ---------------------------------------------------------------------------
# WebSocket Connection Manager (Lightweight JSON Only)
# ---------------------------------------------------------------------------
class ConnectionManager:
    """Thread-safe WebSocket manager broadcasting lightweight telemetry JSON."""

    def __init__(self):
        self._conns: Dict[int, Tuple[WebSocket, asyncio.Queue]] = {}
        self._mu = threading.Lock()

    async def connect(self, ws: WebSocket) -> asyncio.Queue:
        await ws.accept()
        q: asyncio.Queue = asyncio.Queue(maxsize=4)
        with self._mu:
            self._conns[id(ws)] = (ws, q)
        log.info(f"WebSocket client connected (total={len(self._conns)})")
        return q

    async def disconnect(self, ws: WebSocket):
        with self._mu:
            self._conns.pop(id(ws), None)
        log.info(f"WebSocket client disconnected (total={len(self._conns)})")

    def broadcast_from_thread(self, data: dict):
        global _event_loop
        if _event_loop is None:
            return
        with self._mu:
            if not self._conns:
                return
            items = list(self._conns.values())

        msg = json.dumps(data, ensure_ascii=False)

        def _safe_put(target_q, payload):
            try:
                target_q.put_nowait(payload)
            except (asyncio.QueueFull, Exception):
                pass

        for _ws, q in items:
            try:
                _event_loop.call_soon_threadsafe(_safe_put, q, msg)
            except Exception:
                pass


manager = ConnectionManager()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _enrolled_count() -> int:
    if not DB_PATH.exists():
        return 0
    try:
        conn = sqlite3.connect(str(DB_PATH))
        count = conn.execute("SELECT COUNT(*) FROM persons").fetchone()[0]
        conn.close()
        return int(count)
    except Exception:
        return 0


def _detect_alerts(persons: list) -> list:
    global _prev_statuses
    alerts = []
    current_ids = set()

    for p in persons:
        if p.track_id <= 0:
            continue
        current_ids.add(p.track_id)
        status_str = p.status.name
        prev = _prev_statuses.get(p.track_id)

        if status_str in ("CONFIRMED_UNKNOWN", "EVASIVE_ALERT") and prev != status_str:
            snaps = sorted(SNAPSHOT_DIR.glob(f"intruder_track{p.track_id}_*.jpg"))
            snap_url = f"/snapshots/{snaps[-1].name}" if snaps else None
            ts = time.time()
            alerts.append({
                "type":          "alert",
                "is_intruder":   True,
                "track_id":      p.track_id,
                "status":        status_str,
                "person_name":   p.person_name or "Unknown Intruder",
                "snapshot_url":  snap_url,
                "timestamp":     ts,
                "timestamp_str": time.strftime("%H:%M:%S", time.localtime(ts)),
                "s1":            round(float(p.s1), 3),
                "z_score":       round(float(p.z_score), 2),
            })
            # Persist alert to database
            try:
                conn = sqlite3.connect(str(DB_PATH))
                conn.execute(
                    "INSERT INTO alerts (track_id, alert_type, top1_person, s1, z_score, snapshot_path, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (p.track_id, status_str, p.person_name or "Unknown", p.s1, p.z_score, snap_url or "", ts),
                )
                conn.commit()
                conn.close()
            except Exception as e:
                log.warning(f"Could not record alert: {e}")

        elif status_str == "KNOWN_NO_BADGE" and prev != status_str:
            ts = time.time()
            alerts.append({
                "type":          "badge_warning",
                "is_intruder":   False,
                "track_id":      p.track_id,
                "status":        status_str,
                "person_name":   p.person_name or "Authorized Person",
                "snapshot_url":  None,
                "timestamp":     ts,
                "timestamp_str": time.strftime("%H:%M:%S", time.localtime(ts)),
                "s1":            round(float(p.s1), 3),
                "z_score":       round(float(p.z_score), 2),
            })
            try:
                conn = sqlite3.connect(str(DB_PATH))
                conn.execute(
                    "INSERT INTO alerts (track_id, alert_type, top1_person, s1, z_score, snapshot_path, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (p.track_id, status_str, p.person_name or "Authorized Person", p.s1, p.z_score, "", ts),
                )
                conn.commit()
                conn.close()
            except Exception as e:
                log.warning(f"Could not record badge warning: {e}")

        _prev_statuses[p.track_id] = status_str

    for old_id in list(_prev_statuses.keys()):
        if old_id not in current_ids:
            _prev_statuses.pop(old_id, None)

    return alerts


# ---------------------------------------------------------------------------
# Thread B — Main Video Processing Loop
# ---------------------------------------------------------------------------
def processing_loop():
    global camera, _fps_counter, _latest_annotated_jpeg, _latest_raw_frame

    log.info("Starting Video Processing Loop...")
    with _source_lock:
        src = _current_source

    camera = ThreadedCamera(
        src=0 if src == SOURCE_WEBCAM else str(VIDEO_PATH),
        is_file=(src == SOURCE_DEMO),
    ).start()

    last_telemetry_time = 0.0

    while not shutdown_event.is_set():
        if _source_changed.is_set():
            _source_changed.clear()
            with _source_lock:
                src = _current_source
            log.info(f"Switching video source to: {src}")
            if camera:
                camera.stop()
            camera = ThreadedCamera(
                src=0 if src == SOURCE_WEBCAM else str(VIDEO_PATH),
                is_file=(src == SOURCE_DEMO),
            ).start()
            time.sleep(0.1)
            continue

        ret, frame = camera.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue

        with _raw_frame_lock:
            _latest_raw_frame = frame

        # Process frame through surveillance engine
        now = time.time()
        try:
            with reload_lock:
                res = engine.process_frame(frame, timestamp=now)
        except Exception as e:
            traceback.print_exc()
            H, W = frame.shape[:2]
            err_msg = f"SERVER PROCESS ERROR: {e}"
            cv2.rectangle(frame, (0, 0), (W, 44), (0, 0, 180), -1)
            cv2.putText(frame, err_msg[:80], (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            res = FrameResult(frame_idx=0, timestamp=now, annotated_frame=frame, fps=0.0)

        _fps_counter = res.fps

        # Compress to JPEG for native /video_feed stream
        if res.annotated_frame is not None:
            _, buf = cv2.imencode(".jpg", res.annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            with _latest_jpeg_lock:
                _latest_annotated_jpeg = buf.tobytes()
            _new_jpeg_event.set()

        # Emit any immediate new alerts
        new_alerts = _detect_alerts(res.persons)
        for al in new_alerts:
            manager.broadcast_from_thread(al)

        # Telemetry updates sent every 200ms (~5 Hz)
        if now - last_telemetry_time >= 0.20:
            last_telemetry_time = now
            persons_json = [
                {
                    "track_id":      p.track_id,
                    "status":        p.status.name,
                    "person_name":   p.person_name or "",
                    "s1":            round(float(p.s1), 3),
                    "z_score":       round(float(p.z_score), 2),
                    "bbox":          [int(x) for x in p.bbox_xyxy[:4]],
                    "radar_xy":      [round(float(p.radar_xy[0]), 1), round(float(p.radar_xy[1]), 1)],
                    "yaw_deg":       round(float(p.yaw_deg), 1),
                    "pitch_deg":     round(float(p.pitch_deg), 1),
                    "has_id_card":   bool(p.has_id_card),
                    "card_bbox":     [int(x) for x in p.card_bbox[:4]] if p.card_bbox is not None else None,
                    "card_conf":     round(float(p.card_conf), 2),
                }
                for p in res.persons
            ]

            id_count = sum(1 for p in res.persons if p.status == TrackStatus.ID_CARD_VERIFIED)
            known_count = sum(1 for p in res.persons if p.status == TrackStatus.KNOWN_NO_BADGE)
            unknown_count = sum(1 for p in res.persons if p.status in (TrackStatus.CONFIRMED_UNKNOWN, TrackStatus.EVASIVE_ALERT))

            telemetry_data = {
                "type":                 "telemetry",
                "fps":                  round(res.fps, 1),
                "persons":              persons_json,
                "enrolled_count":       _enrolled_count(),
                "id_card_count":        id_count,
                "known_no_badge_count": known_count,
                "unknown_intruder_count": unknown_count,
                "frame_idx":            res.frame_idx,
                "source":               src,
            }
            manager.broadcast_from_thread(telemetry_data)

    if camera:
        camera.stop()
    engine.close()
    log.info("Processing loop terminated.")


# ---------------------------------------------------------------------------
# Lifespan Context Manager
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _event_loop
    _event_loop = asyncio.get_running_loop()

    # Pre-load FAISS and SQLite
    if FAISS_PATH.exists() and DB_PATH.exists():
        log.info("Pre-loading FAISS and SQLite into DroneSurveillanceEngine...")
        engine.load(FAISS_PATH, DB_PATH)
    else:
        log.warning("Storage not found. Run python enroll_ids.py first.")

    # Pre-create alerts table in DB
    if DB_PATH.exists():
        try:
            conn = sqlite3.connect(str(DB_PATH))
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    track_id      INTEGER,
                    alert_type    TEXT,
                    top1_person   TEXT,
                    s1            REAL,
                    z_score       REAL,
                    snapshot_path TEXT,
                    created_at    REAL
                )
            """)
            conn.commit()
            conn.close()
        except Exception as e:
            log.warning(f"DB init warning: {e}")

    # Launch background processing thread
    proc_thread = threading.Thread(target=processing_loop, daemon=True, name="video-proc")
    proc_thread.start()

    log.info("Autonomous Drone Surveillance Server running on http://127.0.0.1:8000")
    yield

    shutdown_event.set()
    _new_jpeg_event.set()
    proc_thread.join(timeout=3.0)
    log.info("Server shutdown complete.")


# ---------------------------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------------------------
app = FastAPI(title="Campus Guardian", version="2.0", lifespan=lifespan)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
async def root():
    return FileResponse(str(STATIC_DIR / "index.html"))


# ---------------------------------------------------------------------------
# Native 30 FPS MJPEG Stream Endpoint
# ---------------------------------------------------------------------------
@app.get("/video_feed", include_in_schema=False)
def video_feed():
    """Streams multipart JPEG frames with zero lag directly to <img> tags."""
    def iter_frames():
        while not shutdown_event.is_set():
            _new_jpeg_event.wait(timeout=0.04)
            _new_jpeg_event.clear()
            with _latest_jpeg_lock:
                frame_bytes = _latest_annotated_jpeg
            if frame_bytes is not None:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                )
    return StreamingResponse(
        iter_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ---------------------------------------------------------------------------
# 1-Click Live Self-Calibration Endpoint
# ---------------------------------------------------------------------------
@app.post("/api/enroll_live")
async def api_enroll_live():
    """
    Captures live webcam face + 4 micro-augmentations, appends them to FAISS,
    and reloads the index in <1 second.
    """
    with _raw_frame_lock:
        if _latest_raw_frame is None:
            raise HTTPException(400, "No live webcam frame available. Check camera.")
        frame_copy = _latest_raw_frame.copy()

    def do_calibrate():
        with reload_lock:
            return engine.calibrate_live_face(
                frame_copy,
                faiss_path=FAISS_PATH,
                db_path=DB_PATH,
                person_id="EMP001",
                person_name="Authorized User",
            )

    result = await asyncio.to_thread(do_calibrate)
    if not result.get("success", False):
        raise HTTPException(400, result.get("message", "Calibration failed"))
    return result


# ---------------------------------------------------------------------------
# 1-Click Worn ID Card Calibration Endpoint
# ---------------------------------------------------------------------------
@app.post("/api/calibrate_badge")
async def api_calibrate_badge():
    """
    Crops the center of card_search_roi, saves to storage/badge_template.jpg,
    extracts its ORB descriptors and color profile on the fly, and uses it immediately.
    """
    with _raw_frame_lock:
        if _latest_raw_frame is None:
            raise HTTPException(400, "No live webcam frame available. Check camera.")
        frame_copy = _latest_raw_frame.copy()

    def do_calibrate_badge():
        with reload_lock:
            return engine.calibrate_badge_template(frame_copy, BADGE_TEMPLATE_PATH)

    result = await asyncio.to_thread(do_calibrate_badge)
    if not result.get("success", False):
        raise HTTPException(400, result.get("message", "Badge template calibration failed"))
    return result


# ---------------------------------------------------------------------------
# Standard Photo Upload Enrollment Endpoint
# ---------------------------------------------------------------------------
@app.post("/api/enroll")
async def api_enroll(
    person_id:   str        = Form(...),
    person_name: str        = Form(...),
    file:        UploadFile = File(...),
):
    pid  = person_id.strip().upper()
    name = person_name.strip()
    if not pid or not name:
        raise HTTPException(400, "person_id and person_name are required")

    safe_name = name.replace(" ", "_")
    dest = ID_CARDS_DIR / f"{pid}_{safe_name}.jpg"
    content = await file.read()
    if not content:
        raise HTTPException(400, "Uploaded file is empty")
    dest.write_bytes(content)

    def run_enrollment():
        return subprocess.run(
            [sys.executable, str(ROOT / "enroll_ids.py"),
             "--id_cards_dir", str(ID_CARDS_DIR),
             "--storage_dir",  str(ROOT / "storage")],
            capture_output=True,
            text=True,
        )

    proc = await asyncio.to_thread(run_enrollment)
    if proc.returncode != 0:
        raise HTTPException(500, f"Enrollment failed: {proc.stderr[-300:]}")

    # Hot-reload FAISS into running engine
    def do_reload():
        with reload_lock:
            engine.load(FAISS_PATH, DB_PATH)

    await asyncio.to_thread(do_reload)
    enrolled = _enrolled_count()
    return {
        "success": True,
        "person_id": pid,
        "person_name": name,
        "enrolled_total": enrolled,
        "message": f"{pid} ({name}) enrolled successfully.",
    }


# ---------------------------------------------------------------------------
# Video Source Switching
# ---------------------------------------------------------------------------
@app.post("/api/source")
async def api_set_source(source: str):
    global _current_source
    if source not in (SOURCE_WEBCAM, SOURCE_DEMO):
        raise HTTPException(400, f"Invalid source '{source}'. Use 'webcam' or 'demo'.")
    with _source_lock:
        _current_source = source
    _source_changed.set()
    log.info(f"Video source switched to: {source}")
    return {"success": True, "active_source": source}


# ---------------------------------------------------------------------------
# Status and Alerts Endpoints
# ---------------------------------------------------------------------------
@app.get("/api/status")
async def api_status():
    with _source_lock:
        src = _current_source
    return {
        "fps":            round(_fps_counter, 1),
        "enrolled_count": _enrolled_count(),
        "faiss_loaded":   engine._matcher.is_ready,
        "active_source":  src,
    }


@app.get("/api/alerts")
async def api_alerts(limit: int = 50):
    if not DB_PATH.exists():
        return []
    try:
        conn = sqlite3.connect(str(DB_PATH))
        rows = conn.execute(
            "SELECT track_id, alert_type, top1_person, s1, z_score, snapshot_path, created_at "
            "FROM alerts ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        conn.close()
        return [
            {
                "track_id":      r[0],
                "alert_type":    r[1],
                "top1_person":   r[2],
                "s1":            r[3],
                "z_score":       r[4],
                "snapshot_url":  r[5] if r[5] else None,
                "timestamp":     r[6],
                "timestamp_str": time.strftime("%H:%M:%S", time.localtime(r[6])),
            }
            for r in rows
        ]
    except Exception as e:
        log.error(f"Alerts query error: {e}")
        return []


@app.get("/api/export_csv", include_in_schema=False)
async def api_export_csv():
    """Download Security Audit Log as CSV for college evaluation."""
    import io
    import csv
    from fastapi.responses import Response

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "Record ID",
        "Track ID",
        "Alert Type",
        "Identified Person",
        "Cosine Similarity (S1)",
        "Cohort Z-Score",
        "Snapshot URL",
        "Timestamp",
    ])
    if DB_PATH.exists():
        try:
            conn = sqlite3.connect(str(DB_PATH))
            rows = conn.execute(
                "SELECT id, track_id, alert_type, top1_person, s1, z_score, snapshot_path, created_at "
                "FROM alerts ORDER BY created_at DESC"
            ).fetchall()
            conn.close()
            for r in rows:
                t_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r[7])) if r[7] else ""
                writer.writerow([r[0], r[1], r[2], r[3], round(float(r[4] or 0), 4), round(float(r[5] or 0), 2), r[6], t_str])
        except Exception as e:
            log.warning(f"Export CSV read error: {e}")

    return Response(
        content=output.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=campus_security_audit.csv"},
    )


@app.get("/snapshots/{filename}", include_in_schema=False)
async def get_snapshot(filename: str):
    p = SNAPSHOT_DIR / filename
    if not p.exists() or not p.is_file():
        raise HTTPException(404, "Snapshot not found")
    return FileResponse(str(p), media_type="image/jpeg")


# ---------------------------------------------------------------------------
# Lightweight Telemetry WebSocket Endpoint
# ---------------------------------------------------------------------------
@app.websocket("/ws/telemetry")
@app.websocket("/ws/stream")
async def ws_telemetry(websocket: WebSocket):
    q = await manager.connect(websocket)

    async def sender():
        while True:
            try:
                msg = await asyncio.wait_for(q.get(), timeout=2.0)
                await websocket.send_text(msg)
            except asyncio.TimeoutError:
                await websocket.send_text('{"type":"ping"}')

    async def receiver():
        while True:
            await websocket.receive_text()

    try:
        done, pending = await asyncio.wait(
            {asyncio.create_task(sender()), asyncio.create_task(receiver())},
            return_when=asyncio.FIRST_EXCEPTION,
        )
        for task in pending:
            task.cancel()
    except WebSocketDisconnect:
        pass
    except Exception as e:
        log.warning(f"WebSocket closed: {e}")
    finally:
        await manager.disconnect(websocket)


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info", access_log=False)
