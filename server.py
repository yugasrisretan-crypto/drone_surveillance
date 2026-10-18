"""
server.py
=========
FastAPI Command Center Server — Autonomous Drone Campus Surveillance

Endpoints:
  GET  /                → static/index.html
  WS   /ws/stream       → live JSON stream: annotated frames + telemetry + alerts
  POST /api/enroll      → upload new ID card photo, hot-reload FAISS index
  GET  /snapshots/{f}   → serve intruder snapshot JPEGs
  GET  /api/status      → system health JSON
  GET  /api/alerts      → historical alerts from SQLite (last N)

Usage:
    python server.py
    # or
    uvicorn server:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# CPU setup
# ---------------------------------------------------------------------------
import os, sys, subprocess
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
import base64
import json
import logging
import sqlite3
import subprocess
import sys
import threading
import time
import traceback
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import cv2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from vision_engine import DroneSurveillanceEngine, SNAPSHOT_DIR

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT         = Path(__file__).parent
VIDEO_PATH   = ROOT / "data" / "videos" / "campus_drone.mp4"
FAISS_PATH   = ROOT / "storage" / "campus.faiss"
DB_PATH      = ROOT / "storage" / "campus.db"
ID_CARDS_DIR = ROOT / "data" / "id_cards"
STATIC_DIR   = ROOT / "static"

STREAM_FPS   = 10          # target WebSocket frame rate
STREAM_WIDTH = 960         # resize width before base64 encode
JPEG_QUALITY = 72          # JPEG encode quality (lower = smaller WS messages)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("server")

# ---------------------------------------------------------------------------
# Video source state
# ---------------------------------------------------------------------------
# SOURCE_WEBCAM  → cv2.VideoCapture(0, cv2.CAP_DSHOW)  (laptop built-in cam)
# SOURCE_DEMO    → cv2.VideoCapture(str(VIDEO_PATH))   (demo drone video)
SOURCE_WEBCAM = "webcam"
SOURCE_DEMO   = "demo"

_source_lock       = threading.Lock()
_current_source    = SOURCE_WEBCAM          # default: laptop webcam
_source_changed    = threading.Event()      # signals loop to reopen cap


def _open_capture(source: str):
    """Open the appropriate cv2.VideoCapture for the given source string."""
    if source == SOURCE_WEBCAM:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # CAP_DSHOW = fast startup on Windows
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS,          30)
        label = "Laptop Webcam (0) [CAP_DSHOW]"
    else:
        cap = cv2.VideoCapture(str(VIDEO_PATH))
        label = f"Demo Drone Video ({VIDEO_PATH.name})"
    return cap, label


# ---------------------------------------------------------------------------
# Global engine + synchronisation primitives
# ---------------------------------------------------------------------------
engine         = DroneSurveillanceEngine()
shutdown_event = threading.Event()
reload_lock    = threading.Lock()          # guards engine.load() vs process_frame()
_event_loop: Optional[asyncio.AbstractEventLoop] = None
_fps_counter   = 0.0
_latest_annotated_jpeg: Optional[bytes] = None
_latest_jpeg_lock = threading.Lock()


# ===========================================================================
# WebSocket Connection Manager
# ===========================================================================
class ConnectionManager:
    """Thread-safe manager for all active WebSocket connections."""

    def __init__(self):
        # ws_id → (WebSocket, asyncio.Queue[str])
        self._conns: dict[int, tuple[WebSocket, asyncio.Queue]] = {}
        self._mu = threading.Lock()

    async def connect(self, ws: WebSocket) -> asyncio.Queue:
        await ws.accept()
        q: asyncio.Queue = asyncio.Queue(maxsize=3)
        with self._mu:
            self._conns[id(ws)] = (ws, q)
        log.info(f"WS client connected (total={len(self._conns)})")
        return q

    async def disconnect(self, ws: WebSocket):
        with self._mu:
            self._conns.pop(id(ws), None)
        log.info(f"WS client disconnected (total={len(self._conns)})")

    def broadcast_from_thread(self, data: dict):
        """
        Push a JSON message to every connected client.
        Safe to call from any background thread.
        """
        global _event_loop
        if _event_loop is None:
            return
        with self._mu:
            if not self._conns:
                return
            items = list(self._conns.values())

        # If all client queues are currently full, skip JSON serialization entirely
        available_queues = [q for _ws, q in items if not q.full()]
        if not available_queues:
            return

        msg = json.dumps(data, ensure_ascii=False)

        def _safe_put(target_q, payload):
            try:
                target_q.put_nowait(payload)
            except (asyncio.QueueFull, Exception):
                pass

        for q in available_queues:
            try:
                _event_loop.call_soon_threadsafe(_safe_put, q, msg)
            except Exception:
                pass


manager = ConnectionManager()


# ===========================================================================
# Processing-thread helpers
# ===========================================================================
_prev_statuses: dict[int, str] = {}


def _encode_frame(frame) -> tuple[str, bytes]:
    """JPEG-encode an annotated frame → (base64 ASCII, raw jpeg bytes)."""
    try:
        if frame is None or frame.size == 0:
            raise ValueError("Empty frame")
        h, w = frame.shape[:2]
        if w > STREAM_WIDTH:
            nh = int(h * STREAM_WIDTH / w)
            target = cv2.resize(frame, (STREAM_WIDTH, nh), interpolation=cv2.INTER_AREA)
        else:
            target = frame
        _, buf = cv2.imencode(".jpg", target, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        raw_bytes = buf.tobytes()
        return base64.b64encode(raw_bytes).decode("ascii"), raw_bytes
    except Exception as exc:
        log.warning(f"Frame encode warning: {exc}")
        dummy = np.zeros((20, 20, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", dummy)
        raw_bytes = buf.tobytes()
        return base64.b64encode(raw_bytes).decode("ascii"), raw_bytes


def _persons_to_json(persons) -> list[dict]:
    out = []
    for p in persons:
        out.append({
            "track_id":      p.track_id,
            "status":        p.status.name,
            "person_name":   p.person_name or "",
            "s1":            round(float(p.s1), 4),
            "z_score":       round(float(p.z_score), 4),
            "bbox":          [int(x) for x in p.bbox_xyxy[:4]],
            "radar_xy":      [round(float(p.radar_xy[0]), 1),
                             round(float(p.radar_xy[1]), 1)],
            "yaw_deg":       round(float(p.yaw_deg), 1),
            "pitch_deg":     round(float(p.pitch_deg), 1),
            "has_id_card":   bool(p.has_id_card),
            "card_bbox":     [int(x) for x in p.card_bbox[:4]] if p.card_bbox is not None else None,
            "card_conf":     round(float(p.card_conf), 2),
        })
    return out


def _detect_new_alerts(persons) -> list[dict]:
    """Compare this frame's statuses to previous — emit alerts / badge warnings on transitions."""
    global _prev_statuses
    alerts, current_ids = [], set()
    for p in persons:
        if p.track_id <= 0:
            continue
        current_ids.add(p.track_id)
        status = p.status.name
        prev   = _prev_statuses.get(p.track_id)

        # Branch 2B / Evasive: Unregistered intruder alert
        if status in ("CONFIRMED_UNKNOWN", "EVASIVE_ALERT") and prev != status:
            snaps    = sorted(SNAPSHOT_DIR.glob(f"intruder_track{p.track_id}_*.jpg"))
            snap_url = f"/snapshots/{snaps[-1].name}" if snaps else None
            ts = time.time()
            alerts.append({
                "type":          "alert",
                "is_intruder":   True,
                "track_id":      p.track_id,
                "status":        status,
                "person_name":   p.person_name or "Unknown Person",
                "snapshot_url":  snap_url,
                "timestamp":     ts,
                "timestamp_str": time.strftime("%H:%M:%S", time.localtime(ts)),
                "s1":            round(float(p.s1), 4),
                "z_score":       round(float(p.z_score), 4),
            })

        # Branch 2A: Minor badge-compliance warning (face verified, but wearing no ID card)
        elif status == "KNOWN_NO_BADGE" and prev != status:
            ts = time.time()
            alerts.append({
                "type":          "badge_warning",
                "is_intruder":   False,
                "track_id":      p.track_id,
                "status":        status,
                "person_name":   p.person_name or "Authorized Person",
                "snapshot_url":  None,
                "timestamp":     ts,
                "timestamp_str": time.strftime("%H:%M:%S", time.localtime(ts)),
                "s1":            round(float(p.s1), 4),
                "z_score":       round(float(p.z_score), 4),
            })

        _prev_statuses[p.track_id] = status
    for gone in set(_prev_statuses) - current_ids:
        _prev_statuses.pop(gone, None)
    return alerts


def _enrolled_count() -> int:
    try:
        if not DB_PATH.exists():
            return 0
        conn = sqlite3.connect(str(DB_PATH))
        n = conn.execute("SELECT COUNT(*) FROM persons").fetchone()[0]
        conn.close()
        return int(n)
    except Exception:
        return 0


# ===========================================================================
# Video Processing Thread
# ===========================================================================
def processing_loop():
    """
    Background daemon thread.  Reads frames from the currently-selected
    source (webcam or demo video), runs the full DroneSurveillanceEngine
    pipeline, and broadcasts results to all WebSocket clients.

    The source can be changed at runtime by setting _current_source and
    firing _source_changed — the loop will reopen the capture device on
    the next iteration without restarting the thread.
    """
    global _fps_counter, _current_source

    log.info("Processing thread starting…")

    interval   = 1.0 / STREAM_FPS
    fps_window: deque = deque(maxlen=30)

    cap: Optional[cv2.VideoCapture] = None
    active_source = ""

    def open_source():
        global _current_source
        nonlocal cap, active_source
        if cap is not None:
            cap.release()
            cap = None

        with _source_lock:
            src = _current_source

        # For demo-video source, wait up to 60 s for the file to appear
        if src == SOURCE_DEMO:
            for _ in range(30):
                if VIDEO_PATH.exists() or shutdown_event.is_set():
                    break
                log.info(f"Waiting for demo video: {VIDEO_PATH}")
                time.sleep(2)
            if not VIDEO_PATH.exists():
                log.warning(f"Demo video not found: {VIDEO_PATH} — falling back to webcam")
                with _source_lock:
                    _current_source = SOURCE_WEBCAM
                src = SOURCE_WEBCAM

        new_cap, label = _open_capture(src)
        if not new_cap.isOpened():
            log.error(f"Cannot open source: {label}")
            new_cap.release()
            return False

        cap = new_cap
        active_source = src
        source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        log.info(f"Video source opened: {label}  src_fps={source_fps:.0f}")
        return True

    if not open_source():
        log.error("Could not open any video source — processing thread exiting.")
        return

    frame_pos = 0

    while not shutdown_event.is_set():
        # ---------------------------------------------------------------
        # Check if the user requested a source switch
        # ---------------------------------------------------------------
        if _source_changed.is_set():
            _source_changed.clear()
            frame_pos = 0
            if not open_source():
                time.sleep(1)
                continue

        t0 = time.time()

        try:
            ret, raw_frame = cap.read()
        except Exception as read_exc:
            log.warning(f"VideoCapture read warning: {read_exc}")
            time.sleep(0.05)
            continue

        frame_pos += 1

        if not ret or raw_frame is None:
            if active_source == SOURCE_DEMO:
                # Loop demo video
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_pos = 0
            else:
                # Webcam stalled — brief pause and retry
                time.sleep(0.05)
            continue

        # ---------------------------------------------------------------
        # Run engine (hold reload_lock so hot-reload doesn't race)
        # ---------------------------------------------------------------
        try:
            with reload_lock:
                result = engine.process_frame(
                    raw_frame, timestamp=time.time(), annotate=True
                )
        except Exception as exc:
            log.error(f"Frame processing error: {exc}")
            traceback.print_exc()
            del raw_frame
            continue

        # Periodic memory cleanup every 30 frames
        if frame_pos % 30 == 0:
            import gc
            gc.collect()

        # ---------------------------------------------------------------
        # Encode & build WebSocket payload
        # ---------------------------------------------------------------
        try:
            b64, raw_jpeg = _encode_frame(result.annotated_frame)
            with _latest_jpeg_lock:
                global _latest_annotated_jpeg
                _latest_annotated_jpeg = raw_jpeg
            persons_data = _persons_to_json(result.persons)

            elapsed = time.time() - t0
            fps_window.append(elapsed)
            _fps_counter = 1.0 / (sum(fps_window) / len(fps_window)) if fps_window else 0.0

            # ---------------------------------------------------------------
            # Real-time Two-Tier Gate counts
            # ---------------------------------------------------------------
            id_card_count = sum(1 for p in result.persons if p.status.name == "ID_CARD_VERIFIED")
            known_no_badge_count = sum(1 for p in result.persons if p.status.name == "KNOWN_NO_BADGE")
            unknown_intruder_count = sum(1 for p in result.persons if p.status.name in ("CONFIRMED_UNKNOWN", "EVASIVE_ALERT"))

            frame_msg = {
                "type":                   "frame",
                "frame":                  b64,
                "persons":                persons_data,
                "fps":                    round(_fps_counter, 1),
                "frame_idx":              result.frame_idx,
                "enrolled_count":         _enrolled_count(),
                "source":                 active_source,
                "id_card_count":          id_card_count,
                "known_no_badge_count":   known_no_badge_count,
                "unknown_intruder_count": unknown_intruder_count,
            }
            manager.broadcast_from_thread(frame_msg)

            # ---------------------------------------------------------------
            # Detect and broadcast alert transitions
            # ---------------------------------------------------------------
            for alert in _detect_new_alerts(result.persons):
                if alert.get("is_intruder", False):
                    log.warning(f"🚨 INTRUDER ALERT -> Track {alert['track_id']} ({alert['status']})")
                else:
                    log.info(f"ℹ️ BADGE WARNING -> Track {alert['track_id']} ({alert.get('person_name', '')}) verified by face without ID card")
                manager.broadcast_from_thread(alert)
        except Exception as exc:
            log.error(f"Post-processing error: {exc}")
            traceback.print_exc()
        finally:
            try:
                del result
            except (UnboundLocalError, NameError):
                pass
            try:
                del raw_frame
            except (UnboundLocalError, NameError):
                pass

        # ---------------------------------------------------------------
        # Throttle to target FPS
        # ---------------------------------------------------------------
        sleep_t = interval - (time.time() - t0)
        if sleep_t > 0:
            time.sleep(sleep_t)

    if cap is not None:
        cap.release()
    log.info("Processing thread stopped.")


# ===========================================================================
# FastAPI lifespan
# ===========================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _event_loop
    _event_loop = asyncio.get_event_loop()

    # Load engine if index exists
    if FAISS_PATH.exists() and DB_PATH.exists():
        engine.load(FAISS_PATH, DB_PATH)
        log.info(f"Engine loaded: {_enrolled_count()} enrolled person(s)")
    else:
        log.warning("No FAISS index found — run setup_demo_data.py first")

    # Pre-create the alerts table so /api/alerts never errors on first call
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
            log.warning(f"Could not pre-create alerts table: {e}")


    # Start background processing thread
    proc_thread = threading.Thread(
        target=processing_loop, daemon=True, name="video-proc"
    )
    proc_thread.start()

    log.info("🚁 Campus Guardian server is live at http://127.0.0.1:8000")
    yield

    # Shutdown
    shutdown_event.set()
    proc_thread.join(timeout=6)
    log.info("Server shut down cleanly.")


# ===========================================================================
# FastAPI App
# ===========================================================================
app = FastAPI(title="Campus Guardian", version="1.0", lifespan=lifespan)

# Serve static files (CSS, JS, assets if any)
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# Root — serve the Command Center dashboard
# ---------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
async def root():
    return FileResponse(str(STATIC_DIR / "index.html"))


# ---------------------------------------------------------------------------
# Snapshot serving
# ---------------------------------------------------------------------------
@app.get("/snapshots/{filename}", include_in_schema=False)
async def get_snapshot(filename: str):
    path = SNAPSHOT_DIR / filename
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Snapshot not found")
    return FileResponse(str(path), media_type="image/jpeg")


# ---------------------------------------------------------------------------
# MJPEG Live Stream Fallback (for browsers/clients where WebSockets stall)
# ---------------------------------------------------------------------------
@app.get("/video_feed", include_in_schema=False)
def video_feed():
    """
    MJPEG streaming endpoint fallback. Continuously streams multipart JPEG frames.
    """
    def iter_frames():
        while not shutdown_event.is_set():
            with _latest_jpeg_lock:
                frame_data = _latest_annotated_jpeg
            if frame_data is not None:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame_data + b"\r\n"
                )
            time.sleep(0.04)  # ~25 fps max
    return StreamingResponse(
        iter_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ---------------------------------------------------------------------------
# System status
# ---------------------------------------------------------------------------
@app.get("/api/status")
async def api_status():
    with _source_lock:
        src = _current_source
    return {
        "fps":            round(_fps_counter, 1),
        "enrolled_count": _enrolled_count(),
        "faiss_loaded":   engine._matcher.is_ready,
        "video_exists":   VIDEO_PATH.exists(),
        "video_path":     str(VIDEO_PATH),
        "uptime_s":       round(time.time()),
        "active_source":  src,
    }


# ---------------------------------------------------------------------------
# Video source toggle
# ---------------------------------------------------------------------------
@app.post("/api/source")
async def api_set_source(source: str):
    """
    Switch the live video source.
    source = 'webcam'  → laptop webcam  (cv2.VideoCapture(0, CAP_DSHOW))
    source = 'demo'    → demo drone video file
    """
    global _current_source
    if source not in (SOURCE_WEBCAM, SOURCE_DEMO):
        raise HTTPException(400, f"Unknown source '{source}'. Use 'webcam' or 'demo'.")
    with _source_lock:
        _current_source = source
    _source_changed.set()     # wake the processing loop
    log.info(f"Video source switched to: {source}")
    return {"success": True, "active_source": source}


# ---------------------------------------------------------------------------
# Historical alerts
# ---------------------------------------------------------------------------
@app.get("/api/alerts")
async def api_alerts(limit: int = 50):
    if not DB_PATH.exists():
        return []
    try:
        conn = sqlite3.connect(str(DB_PATH))
        rows = conn.execute(
            "SELECT track_id, alert_type, top1_person, s1, z_score, "
            "       snapshot_path, created_at "
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
                "snapshot_url":  f"/snapshots/{Path(r[5]).name}" if r[5] else None,
                "timestamp":     r[6],
                "timestamp_str": time.strftime("%H:%M:%S", time.localtime(r[6])),
            }
            for r in rows
        ]
    except Exception as exc:
        log.error(f"api_alerts DB error: {exc}")
        return []


# ---------------------------------------------------------------------------
# Enrollment — hot-reload FAISS after adding a new ID card photo
# ---------------------------------------------------------------------------
@app.post("/api/enroll")
async def api_enroll(
    person_id:   str        = Form(...),
    person_name: str        = Form(...),
    file:        UploadFile = File(...),
):
    """
    Save uploaded ID card photo and re-run enroll_ids.py.
    Hot-reloads the FAISS index in the running engine.
    """
    pid  = person_id.strip().upper()
    name = person_name.strip()
    if not pid or not name:
        raise HTTPException(400, "person_id and person_name are required")

    # Save photo
    safe_name = name.replace(" ", "_")
    dest = ID_CARDS_DIR / f"{pid}_{safe_name}.jpg"
    content = await file.read()
    if not content:
        raise HTTPException(400, "Uploaded file is empty")
    dest.write_bytes(content)
    log.info(f"Saved ID card: {dest}")

    # Run enrollment subprocess (non-blocking via thread)
    def run_enrollment():
        return subprocess.run(
            [sys.executable, str(ROOT / "enroll_ids.py"),
             "--id_cards_dir", str(ID_CARDS_DIR),
             "--storage_dir",  str(ROOT / "storage"),
             "--det_thresh",   "0.25"],
            capture_output=True,
            text=True,
        )

    proc = await asyncio.to_thread(run_enrollment)
    if proc.returncode not in (0, 1):   # 1 = stderr-only warnings (acceptable)
        raise HTTPException(
            500,
            f"Enrollment subprocess failed (code {proc.returncode}): "
            f"{proc.stderr[-400:] if proc.stderr else 'no output'}",
        )

    # Hot-reload FAISS + DB into running engine (in a thread to avoid blocking event loop)
    def do_reload():
        if FAISS_PATH.exists() and DB_PATH.exists():
            with reload_lock:
                engine.load(FAISS_PATH, DB_PATH)
            log.info(f"FAISS hot-reloaded after enrolling {pid}")

    await asyncio.to_thread(do_reload)

    enrolled = _enrolled_count()
    return {
        "success":        True,
        "person_id":      pid,
        "person_name":    name,
        "enrolled_total": enrolled,
        "message":        f"{pid} ({name}) enrolled. Index: {enrolled} person(s).",
    }


# ---------------------------------------------------------------------------
# WebSocket — live stream endpoint
# ---------------------------------------------------------------------------
@app.websocket("/ws/stream")
async def ws_stream(websocket: WebSocket):
    q = await manager.connect(websocket)

    async def sender():
        """Push messages from the processing thread to the WebSocket client."""
        while True:
            try:
                msg = await asyncio.wait_for(q.get(), timeout=3.0)
                await websocket.send_text(msg)
            except asyncio.TimeoutError:
                # Keepalive ping so the browser doesn't close the connection
                await websocket.send_text('{"type":"ping"}')

    async def receiver():
        """Drain incoming client messages (keepalives / pings) so the buffer clears."""
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
    except Exception as exc:
        traceback.print_exc()
        log.warning(f"WS error: {exc}")
    finally:
        await manager.disconnect(websocket)


# ===========================================================================
# Entry point
# ===========================================================================
if __name__ == "__main__":
    import uvicorn
    try:
        uvicorn.run(
            app,
            host="127.0.0.1",
            port=8000,
            log_level="info",
            access_log=False,
        )
    except Exception as e:
        log.error(f"Uvicorn server crashed: {e}")
        traceback.print_exc()
    except BaseException as be:
        log.warning(f"Uvicorn stopped with BaseException: {be} ({type(be).__name__})")
