# 🚁 Autonomous Drone Campus Surveillance System
## Architecture Document

> **100% Standalone · Zero-Cloud · Real-Time Intruder Detection**

---

## 1. System Overview

A fully offline, edge-deployable surveillance system that:
1. Ingests live drone video (RTSP/USB) or pre-recorded footage.
2. Detects and tracks every person in each frame using a YOLO-based detector.
3. Extracts a 512-d face embedding for each detected person via InsightFace (ArcFace backbone, ONNX runtime).
4. Matches each embedding against a pre-built FAISS index of authorised ID-card photos.
5. Flags any person whose best-match cosine similarity falls below a configurable threshold as an **intruder**.
6. Streams real-time alerts (bounding box, snapshot, match score) over WebSocket to a browser dashboard.

No data ever leaves the host machine. The system runs entirely on local CPU/GPU resources.

---

## 2. High-Level Architecture

```
+------------------------------------------------------------------+
|                        DRONE / VIDEO SOURCE                      |
|               (RTSP stream · USB camera · MP4 file)              |
+----------------------------+-------------------------------------+
                             |  raw frames (BGR)
                             v
+------------------------------------------------------------------+
|                   INGESTION & PRE-PROCESSING                     |
|   VideoCapture (OpenCV)  .  Frame queue  .  Resolution scaling   |
+----------------------------+-------------------------------------+
                             |  pre-processed frames
                             v
+------------------------------------------------------------------+
|               PERSON DETECTION & TRACKING MODULE                 |
|   YOLOv8n / YOLOv8s  (Ultralytics)                              |
|   ByteTrack via Supervision  ->  stable person IDs per frame     |
+----------------------------+-------------------------------------+
                             |  person crops + track IDs
                             v
+------------------------------------------------------------------+
|                  FACE ANALYSIS MODULE                            |
|   InsightFace buffalo_sc model (ONNX Runtime)                    |
|   . Face detection within person crop                            |
|   . ArcFace embedding extraction  ->  512-d float32 vector       |
+----------------------------+-------------------------------------+
                             |  face embeddings
                             v
+------------------------------------------------------------------+
|                  IDENTITY MATCHING MODULE                        |
|   FAISS IndexFlatIP (inner-product = cosine on unit vectors)     |
|   . Threshold: similarity < 0.35  ->  INTRUDER                  |
|   . Known person  ->  identity label returned                    |
+----------------------------+-------------------------------------+
                             |  match result + label
                             v
+------------------------------------------------------------------+
|                    ALERT & SNAPSHOT ENGINE                       |
|   . Annotates frame with bounding boxes + labels (Supervision)   |
|   . Saves JPEG snapshot to  storage/snapshots/                   |
|   . Publishes AlertEvent JSON over WebSocket                     |
+----------------------------+-------------------------------------+
                             |  AlertEvent JSON
                             v
+------------------------------------------------------------------+
|                    FastAPI BACKEND  (uvicorn)                    |
|   REST endpoints:                                                |
|   . POST /api/enroll          - add ID card photo                |
|   . POST /api/stream/start    - start video processing           |
|   . POST /api/stream/stop     - stop video processing            |
|   . GET  /api/alerts          - paginated alert history          |
|   . GET  /api/snapshots/{id}  - retrieve snapshot JPEG           |
|   WebSocket:                                                     |
|   . WS /ws/alerts             - live alert push to dashboard     |
+----------------------------+-------------------------------------+
                             |  HTTP / WS
                             v
+------------------------------------------------------------------+
|                   BROWSER DASHBOARD  (static/)                   |
|   Single-page app (vanilla HTML + CSS + JS)                      |
|   . Live annotated video feed (MJPEG or canvas)                  |
|   . Real-time intruder alert cards with snapshot thumbnails      |
|   . Enrollment form: drag-and-drop ID card photo                 |
|   . System status panel (FPS, queue depth, FAISS index size)     |
+------------------------------------------------------------------+
```

---

## 3. Component Breakdown

| Component | Technology | Purpose |
|---|---|---|
| Person Detection | YOLOv8n (Ultralytics) | Detect people class in each frame |
| Multi-Object Tracking | ByteTrack (Supervision) | Assign stable track IDs across frames |
| Face Detection | InsightFace RetinaFace | Locate face within person crop |
| Face Embedding | ArcFace (ONNX Runtime) | Extract 512-d identity embedding |
| Identity Index | FAISS IndexFlatIP | Sub-millisecond NN search over embeddings |
| Backend API | FastAPI + Uvicorn | REST + WebSocket server |
| Video I/O | OpenCV (headless) | Frame decode/encode, MJPEG streaming |
| Alert Delivery | WebSockets | Push alerts to all connected dashboards |
| Dashboard | Vanilla HTML/CSS/JS | Zero-dependency browser UI |

---

## 4. Data Flow — Enrollment

```
Admin uploads ID card JPEG
        |
        v
InsightFace detects & embeds face  ->  512-d vector (L2-normalised)
        |
        v
FAISS index.add(vector)  +  metadata store (name, photo path, enroll time)
        |
        v
Index persisted to  data/faiss_index.bin  +  data/metadata.json
```

---

## 5. Data Flow — Intruder Detection

```
Frame arrives from drone
   |
   v  YOLO detect
Person crops (N persons)
   |
   v  InsightFace per crop
Embeddings (M faces, M <= N)
   |
   v  FAISS search(k=1)
similarity score per face
   |
   +-- score >= 0.35  ->  KNOWN  (label, score)
   |
   +-- score < 0.35   ->  INTRUDER  ->  AlertEvent
                              |
                        snapshot saved
                        WebSocket push
                        alert logged
```

---

## 6. Directory Structure

```
drone_surveillance/
+-- ARCHITECTURE.md            <- this file
+-- requirements.txt           <- pinned Python dependencies
+-- .venv/                     <- isolated virtual environment
|
+-- app/                       <- application source (Phase 2+)
|   +-- main.py                <- FastAPI app entry point
|   +-- config.py              <- thresholds, paths, constants
|   +-- pipeline/
|   |   +-- detector.py        <- YOLOv8 person detection
|   |   +-- tracker.py         <- ByteTrack wrapper
|   |   +-- embedder.py        <- InsightFace face embedding
|   |   +-- matcher.py         <- FAISS identity matching
|   +-- api/
|   |   +-- enroll.py          <- /api/enroll endpoint
|   |   +-- stream.py          <- /api/stream/* endpoints
|   |   +-- alerts.py          <- /api/alerts endpoint
|   +-- ws/
|       +-- manager.py         <- WebSocket connection manager
|
+-- data/
|   +-- id_cards/              <- enrolled ID card photos (JPEG/PNG)
|   +-- videos/                <- test/recorded drone footage
|   +-- faiss_index.bin        <- persisted FAISS index (generated)
|   +-- metadata.json          <- identity metadata (generated)
|
+-- storage/
|   +-- snapshots/             <- intruder alert snapshot JPEGs
|
+-- static/                    <- browser dashboard assets
    +-- index.html
    +-- style.css
    +-- app.js
```

---

## 7. Performance Targets

| Metric | Target |
|---|---|
| Detection throughput | >= 15 FPS on CPU (YOLOv8n, 640px) |
| Face embedding latency | < 30 ms per crop (ONNX Runtime, CPU) |
| FAISS search latency | < 1 ms for 10,000 enrolled identities |
| Alert-to-dashboard latency | < 500 ms end-to-end |
| Snapshot write time | < 100 ms |

---

## 8. Security & Privacy

- All processing is **on-device** — no frames, embeddings, or alerts are sent to any external server.
- ID card photos are stored locally under `data/id_cards/` with restricted filesystem permissions.
- FAISS index and metadata are stored in binary format — not human-readable.
- Dashboard is served on `localhost` only by default (`--host 127.0.0.1`).

---

## 9. Dependency Rationale

| Package | Why |
|---|---|
| `onnxruntime` | Fast CPU ONNX model inference for InsightFace models |
| `insightface` | State-of-the-art face detection + ArcFace embedding (ONNX-backed) |
| `ultralytics` | YOLOv11 Nano person detection with simple Python API |
| `supervision` | ByteTrack multi-object tracking + annotation utilities |
| `faiss-cpu` | Facebook AI Similarity Search — sub-ms k-NN over float32 vectors |
| `fastapi` | High-performance async REST + WebSocket backend |
| `uvicorn` | ASGI server for FastAPI |
| `websockets` | WebSocket protocol support |
| `python-multipart` | File upload parsing for FastAPI |
| `opencv-python-headless` | Frame I/O, JPEG encode, MJPEG streaming (no GUI deps) |
| `scipy` | Cosine distance utilities, signal processing helpers |

---

*Last updated: 2026-10-14 | Phase 1 — Environment Setup*
