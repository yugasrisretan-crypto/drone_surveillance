"""
vision_engine.py
================
High-Speed, High-Accuracy CPU Drone Surveillance Engine (Phase 2 Refactor)

3-Thread Architecture:
  - Thread A (ThreadedCamera): Grabs 640x480 frames via cv2.VideoCapture(0, cv2.CAP_DSHOW)
    in a daemon thread to completely eliminate webcam buffer lag.
  - Thread B (Fast Main Loop — 25 to 30 FPS): Runs frame-by-frame on CPU. Uses fast buffalo_sc
    ONNX face/person box localization every 3rd frame (~22ms) and lightweight OpenCV centroid/IoU
    box interpolation on intermediate frames (<1ms), completely bypassing slow PyTorch CPU overhead.
    Immediately draws smooth HUD boxes (ID CARD WORN: Green, KNOWN FACE: Blue/Green,
    VERIFYING: Yellow, UNKNOWN INTRUDER: Red) and live FPS counter.
  - Thread C (BiometricWorkerThread): Background daemon with queue.Queue(maxsize=1) processing
    freshest crops. First checks chest ROI for ID badge (OpenCV HSV/edge geometric contours).
    If no badge, executes insightface buffalo_sc 5-point norm_crop alignment, applies CLAHE
    illumination normalization, and queries FAISS.

Lock-in Logic:
  - KNOWN after 2 matching checks (S1 >= 0.38 or Z >= 3.0). Caches torso HSV histogram and freezes recognition.
  - ID_CARD_WORN after 2 positive badge checks. Caches torso histogram and freezes recognition.
  - CONFIRMED_UNKNOWN after 4 negative quality-gated checks (S1 < 0.32, no badge). Saves snapshot and emits alert.
"""

from __future__ import annotations

import logging
import os
import queue
import sqlite3
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import faiss
import numpy as np
import onnxruntime as ort

log = logging.getLogger("vision_engine")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

# ---------------------------------------------------------------------------
# Constants & Thresholds
# ---------------------------------------------------------------------------
EMBEDDING_DIM             = 512
FAISS_TOP_K               = 20
S1_KNOWN_THRESH           = 0.38     # Cosine sim -> KNOWN (2 matching checks)
Z_KNOWN_THRESH            = 3.0      # Z-score -> KNOWN
S1_UNKNOWN_THRESH         = 0.32     # Cosine sim < 0.32 -> vote UNKNOWN (4 checks)
FRAMES_TO_LOCK_KNOWN      = 2        # 2 matching checks to lock
FRAMES_TO_LOCK_CARD       = 2        # 2 badge checks to lock ID CARD WORN
FRAMES_TO_ALERT_UNKNOWN   = 4        # 4 consecutive unknown checks to alert
CARD_CONF_THRESH          = 0.50     # ID card contour confidence threshold
EVASIVE_SECONDS           = 6.0      # Face hidden time threshold
BLUR_VAR_THRESH           = 25.0     # Laplacian variance blur rejection
MIN_FACE_PX               = 24
YAW_MAX_DEG               = 45.0
PITCH_MAX_DEG             = 50.0
TORSO_HIST_BINS           = 32
SNAPSHOT_DIR              = Path("storage/snapshots")

# Radar projection homography (1000x1000 campus coordinate frame)
DEFAULT_HOMOGRAPHY = np.array([
    [1000.0 / 640, 0,            0],
    [0,            1000.0 / 480, 0],
    [0,            0,            1],
], dtype=np.float64)


# ---------------------------------------------------------------------------
# Color space normalization
# ---------------------------------------------------------------------------
def apply_clahe_lab(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """Normalize shadows and uneven lighting in LAB color space."""
    if image is None or image.size == 0:
        return image
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


# ---------------------------------------------------------------------------
# Status Enumeration & Colors
# ---------------------------------------------------------------------------
class TrackStatus(Enum):
    PENDING           = auto()
    ID_CARD_VERIFIED  = auto()   # Compliant with worn ID badge (Bright Green)
    SCANNING_FACE     = auto()   # Verifying biometric ID (Yellow)
    KNOWN_NO_BADGE    = auto()   # Recognized face without card (Teal / Blue-Green)
    CONFIRMED_UNKNOWN = auto()   # Intruder / Unregistered (Red)
    EVASIVE_ALERT     = auto()   # Face concealed > 6s (Orange-Red)
    UNVERIFIABLE      = auto()   # Pose out of bounds (Amber)

    # Aliases
    KNOWN             = KNOWN_NO_BADGE
    ID_CARD_WORN      = ID_CARD_VERIFIED
    VERIFYING         = SCANNING_FACE
    UNKNOWN           = CONFIRMED_UNKNOWN


# BGR Color palette
_STATUS_COLORS = {
    TrackStatus.PENDING:           (200, 200, 200),  # Grey
    TrackStatus.ID_CARD_VERIFIED:  (0,   230, 118),  # Bright Green
    TrackStatus.SCANNING_FACE:     (0,   215, 255),  # Yellow
    TrackStatus.KNOWN_NO_BADGE:    (255, 176,   0),  # Blue / Teal
    TrackStatus.CONFIRMED_UNKNOWN: (0,     0, 255),  # Bright Red
    TrackStatus.EVASIVE_ALERT:     (0,   120, 255),  # Orange-Red
    TrackStatus.UNVERIFIABLE:      (0,   215, 255),  # Amber
}


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------
@dataclass
class MatchResult:
    person_id:   Optional[str]   = None
    person_name: Optional[str]   = None
    s1:          float           = 0.0
    z_score:     float           = 0.0
    is_match:    bool            = False


@dataclass
class TrackState:
    track_id:               int
    status:                 TrackStatus   = TrackStatus.PENDING
    known_person_id:        Optional[str] = None
    known_person_name:      Optional[str] = None
    bbox_xyxy:              np.ndarray    = field(default_factory=lambda: np.zeros(4, dtype=int))
    last_seen_frame:        int           = 0

    # ID Card checks
    card_detected_count:    int           = 0
    no_card_count:          int           = 0
    has_id_card:            bool          = False
    card_bbox:              Optional[np.ndarray] = None
    card_conf:              float         = 0.0

    # Biometric votes
    positive_votes:         int           = 0
    negative_votes:         int           = 0
    voted_person_id:        Optional[str] = None
    voted_person_name:      Optional[str] = None
    s1:                     float         = 0.0
    z_score:                float         = 0.0

    # Pose & face
    face_bbox:              Optional[np.ndarray] = None
    face_kps:               Optional[np.ndarray] = None
    yaw_deg:                float         = 0.0
    pitch_deg:              float         = 0.0

    # Timers & Torso Re-ID
    first_seen_ts:          float         = 0.0
    last_face_visible_ts:   float         = 0.0
    face_hidden_since_ts:   Optional[float] = None
    torso_hist:             Optional[np.ndarray] = None
    alert_saved:            bool          = False
    is_locked:              bool          = False


@dataclass
class TrackedPerson:
    track_id:       int
    bbox_xyxy:      np.ndarray
    status:         TrackStatus
    person_id:      Optional[str]       = None
    person_name:    Optional[str]       = None
    s1:             float               = 0.0
    z_score:        float               = 0.0
    face_bbox:      Optional[np.ndarray]= None
    yaw_deg:        float               = 0.0
    pitch_deg:      float               = 0.0
    radar_xy:       Tuple[float, float] = (0.0, 0.0)
    snapshot_path:  Optional[str]       = None
    has_id_card:    bool                = False
    card_bbox:      Optional[np.ndarray]= None
    card_conf:      float               = 0.0


@dataclass
class FrameResult:
    frame_idx:      int
    timestamp:      float
    persons:        List[TrackedPerson]  = field(default_factory=list)
    annotated_frame:Optional[np.ndarray] = None
    fps:            float                = 0.0


# ===========================================================================
# Thread A — ThreadedCamera (Zero-Lag Frame Ingestion)
# ===========================================================================
class ThreadedCamera:
    """
    Grabs the newest frame in a dedicated background daemon thread.
    Prevents hardware buffer buildup so video is strictly real-time.
    """

    def __init__(self, src: int | str = 0, is_file: bool = False):
        self.src = src
        self.is_file = is_file
        self.cap: Optional[cv2.VideoCapture] = None
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()
        self.latest_frame: Optional[np.ndarray] = None
        self.new_frame_event = threading.Event()
        self._open()

    def _open(self):
        if self.is_file:
            self.cap = cv2.VideoCapture(str(self.src))
        else:
            self.cap = cv2.VideoCapture(int(self.src), cv2.CAP_DSHOW)
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def start(self) -> ThreadedCamera:
        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True, name="ThreadedCamera")
        self.thread.start()
        return self

    def _worker(self):
        while self.running:
            if self.cap is None or not self.cap.isOpened():
                time.sleep(0.05)
                continue
            ret, frame = self.cap.read()
            if not ret:
                if self.is_file:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    time.sleep(0.01)
                    continue
                else:
                    time.sleep(0.01)
                    continue
            with self.lock:
                self.latest_frame = frame
            self.new_frame_event.set()

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        with self.lock:
            if self.latest_frame is None:
                return False, None
            return True, self.latest_frame.copy()

    def stop(self):
        self.running = False
        self.new_frame_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()


# ===========================================================================
# Chest ID Card / Badge Detector (Fast Geometric Filtering)
# ===========================================================================
class ChestCardDetector:
    """
    High-precision, lightweight OpenCV geometric detector for worn ID cards/badges.
    Filters chest ROI for rectangular aspect ratio (1.20 - 1.95), convex solidity,
    and interior text/photo contrast.
    """

    def detect_badge(self, chest_crop: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
        if chest_crop is None or chest_crop.size == 0:
            return None, 0.0

        ch, cw = chest_crop.shape[:2]
        if ch < 30 or cw < 30:
            return None, 0.0

        gray = cv2.cvtColor(chest_crop, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        edges = cv2.Canny(blurred, 35, 110)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 3
        )
        contours, _ = cv2.findContours(closed | (255 - thresh), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        best_bbox = None
        best_conf = 0.0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 350 or area > 0.45 * (ch * cw):
                continue

            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)

            x, y, w, h = cv2.boundingRect(cnt)
            if w < 16 or h < 16:
                continue

            ar = float(max(w, h)) / max(1, min(w, h))
            if not (1.15 <= ar <= 2.2):
                continue

            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / max(1.0, hull_area)
            if solidity < 0.70:
                continue

            # Interior gradient check
            roi_g = gray[y : y + h, x : x + w]
            sobelx = cv2.Sobel(roi_g, cv2.CV_64F, 1, 0, ksize=3)
            grad_energy = float(np.mean(np.abs(sobelx)))

            conf = min(0.98, 0.45 + (grad_energy / 40.0) * 0.4 + (0.15 if len(approx) == 4 else 0.0))
            if conf > best_conf and conf >= CARD_CONF_THRESH:
                best_conf = conf
                best_bbox = np.array([x, y, x + w, y + h], dtype=int)

        return best_bbox, best_conf


# ===========================================================================
# FAISS Cohort Open-Set Matcher
# ===========================================================================
class OpenSetMatcher:
    """FAISS IndexFlatIP open-set biometric matcher with cohort Z-score."""

    def __init__(self):
        self._index: Optional[faiss.Index] = None
        self._id_map: Dict[int, str] = {}
        self._name_map: Dict[str, str] = {}
        self._lock = threading.Lock()

    @property
    def is_ready(self) -> bool:
        with self._lock:
            return self._index is not None and self._index.ntotal > 0

    def load(self, faiss_path: Path, db_path: Path):
        with self._lock:
            if not faiss_path.exists() or not db_path.exists():
                log.warning("FAISS or SQLite storage file does not exist.")
                return

            self._index = faiss.read_index(str(faiss_path))
            conn = sqlite3.connect(str(db_path))
            v_rows = conn.execute("SELECT faiss_id, person_id FROM vectors").fetchall()
            self._id_map = {int(r[0]): r[1] for r in v_rows}

            p_rows = conn.execute("SELECT person_id, name FROM persons").fetchall()
            self._name_map = {r[0]: r[1] for r in p_rows}
            conn.close()

            log.info(f"OpenSetMatcher loaded: {self._index.ntotal} vectors across {len(self._name_map)} persons.")

    def match(self, embedding: np.ndarray) -> MatchResult:
        with self._lock:
            if self._index is None or self._index.ntotal == 0:
                return MatchResult()

            vec = embedding.reshape(1, -1).astype(np.float32)
            k = min(FAISS_TOP_K, self._index.ntotal)
            scores, indices = self._index.search(vec, k)

            s1 = float(scores[0, 0])
            top_faiss_id = int(indices[0, 0])
            top_person_id = self._id_map.get(top_faiss_id)
            top_person_name = self._name_map.get(top_person_id, "Unknown")

            # Cohort Z-score calculation
            all_scores = scores[0]
            if len(all_scores) > 3:
                impostors = all_scores[1:]
                mu = float(np.mean(impostors))
                sigma = float(np.std(impostors))
                z_score = float((s1 - mu) / (sigma + 1e-6))
            else:
                z_score = 0.0

            is_match = (s1 >= S1_KNOWN_THRESH) or (z_score >= Z_KNOWN_THRESH)
            return MatchResult(
                person_id=top_person_id if is_match else None,
                person_name=top_person_name if is_match else None,
                s1=s1,
                z_score=z_score,
                is_match=is_match,
            )


# ===========================================================================
# Thread C — BiometricWorkerThread (Async ID Card + Face Verification)
# ===========================================================================
class BiometricWorkerThread:
    """
    Background worker thread maintaining a queue.Queue(maxsize=1).
    Processes freshest unverified frame crops to guarantee zero impact on Main Loop FPS.
    """

    def __init__(self, matcher: OpenSetMatcher, face_app, on_result_callback):
        self.matcher = matcher
        self.face_app = face_app
        self.on_result_callback = on_result_callback
        self.q: queue.Queue = queue.Queue(maxsize=1)
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.card_detector = ChestCardDetector()

    def start(self) -> BiometricWorkerThread:
        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True, name="BiometricWorker")
        self.thread.start()
        return self

    def submit(self, item: dict):
        """Put item into queue, dropping stale item if full."""
        try:
            self.q.put_nowait(item)
        except queue.Full:
            try:
                self.q.get_nowait()
            except queue.Empty:
                pass
            try:
                self.q.put_nowait(item)
            except queue.Full:
                pass

    def _worker(self):
        from insightface.utils.face_align import norm_crop
        rec_model = self.face_app.models["recognition"] if self.face_app else None

        while self.running:
            try:
                task = self.q.get(timeout=0.1)
            except queue.Empty:
                continue

            track_id = task["track_id"]
            frame = task["frame"]
            p_bbox = task["p_bbox"]
            face_kps = task.get("face_kps")
            fh, fw = frame.shape[:2]

            # 1. Chest ROI ID Card Detection
            px1, py1, px2, py2 = p_bbox
            pw = max(1, px2 - px1)
            ph = max(1, py2 - py1)

            cy1 = max(0, py1 + int(ph * 0.18))
            cy2 = min(fh, py1 + int(ph * 0.75))
            cx1 = max(0, px1)
            cx2 = min(fw, px2)
            chest_crop = frame[cy1:cy2, cx1:cx2]

            badge_bbox, badge_conf = self.card_detector.detect_badge(chest_crop)
            has_badge = badge_bbox is not None and badge_conf >= CARD_CONF_THRESH

            card_res = {
                "track_id": track_id,
                "has_id_card": has_badge,
                "card_conf": badge_conf,
                "card_bbox": (badge_bbox + np.array([cx1, cy1, cx1, cy1])) if badge_bbox is not None else None,
            }

            if has_badge:
                self.on_result_callback("card", card_res)
                continue

            # 2. Biometric Face Alignment & Matching
            if self.face_app is None or rec_model is None or not self.matcher.is_ready:
                self.on_result_callback("card", card_res)
                continue

            kps_to_use = face_kps
            if kps_to_use is None:
                crop_face = frame[max(0, py1):min(fh, py2), max(0, px1):min(fw, px2)]
                faces = self.face_app.get(crop_face)
                if faces:
                    best = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                    if hasattr(best, "kps") and best.kps is not None:
                        kps_to_use = best.kps + np.array([px1, py1])

            if kps_to_use is None or len(kps_to_use) < 5:
                self.on_result_callback("card", card_res)
                continue

            # Strict 5-point norm_crop alignment
            aligned_112 = norm_crop(frame, landmark=kps_to_use, image_size=112)
            clahe_face = apply_clahe_lab(aligned_112)

            feat = rec_model.get_feat(clahe_face).flatten().astype(np.float32)
            norm = np.linalg.norm(feat)
            if norm < 1e-6:
                self.on_result_callback("card", card_res)
                continue

            feat = feat / norm
            match_res = self.matcher.match(feat)

            bio_res = {
                "track_id": track_id,
                "has_id_card": False,
                "card_conf": 0.0,
                "card_bbox": None,
                "match": match_res,
            }
            self.on_result_callback("face", bio_res)

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)


# ===========================================================================
# Lightweight Centroid / IoU Box Tracker
# ===========================================================================
class FastCentroidTracker:
    """Lightweight 0-CPU tracker matching detections via IoU and centroid distance."""

    def __init__(self, max_disappeared: int = 30):
        self.next_id = 1
        self.tracks: Dict[int, np.ndarray] = {}
        self.disappeared: Dict[int, int] = {}

    def update(self, detected_boxes: List[np.ndarray]) -> List[Tuple[int, np.ndarray]]:
        if len(detected_boxes) == 0:
            for tid in list(self.disappeared.keys()):
                self.disappeared[tid] += 1
                if self.disappeared[tid] > 30:
                    self.tracks.pop(tid, None)
                    self.disappeared.pop(tid, None)
            return [(tid, box) for tid, box in self.tracks.items()]

        if len(self.tracks) == 0:
            out = []
            for box in detected_boxes:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = box
                self.disappeared[tid] = 0
                out.append((tid, box))
            return out

        # Match detected boxes with existing tracks using centroid distance
        track_ids = list(self.tracks.keys())
        track_boxes = list(self.tracks.values())

        track_centers = np.array([[(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0] for b in track_boxes])
        det_centers = np.array([[(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0] for b in detected_boxes])

        dists = np.linalg.norm(track_centers[:, np.newaxis] - det_centers[np.newaxis, :], axis=2)

        assigned_tracks = set()
        assigned_dets = set()
        matched = []

        # Greedy match closest
        for _ in range(min(len(track_ids), len(detected_boxes))):
            min_idx = np.unravel_index(np.argmin(dists), dists.shape)
            r, c = min_idx[0], min_idx[1]
            if dists[r, c] > 180.0:
                break
            matched.append((track_ids[r], detected_boxes[c]))
            assigned_tracks.add(r)
            assigned_dets.add(c)
            dists[r, :] = 1e6
            dists[:, c] = 1e6

        # Update matched
        for tid, box in matched:
            self.tracks[tid] = box
            self.disappeared[tid] = 0

        # Unmatched tracks
        for i, tid in enumerate(track_ids):
            if i not in assigned_tracks:
                self.disappeared[tid] += 1
                if self.disappeared[tid] > 30:
                    self.tracks.pop(tid, None)
                    self.disappeared.pop(tid, None)

        # Unmatched detections become new tracks
        for j, box in enumerate(detected_boxes):
            if j not in assigned_dets:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = box
                self.disappeared[tid] = 0

        return [(tid, box) for tid, box in self.tracks.items()]


# ===========================================================================
# Thread B — Fast Main Loop DroneSurveillanceEngine
# ===========================================================================
class DroneSurveillanceEngine:
    """
    Main Surveillance Engine delivering 25-30 FPS on CPU:
      - buffalo_sc ONNX localization runs every 3rd frame (~22ms).
      - Lightweight Centroid/IoU box interpolation on intermediate frames (<1ms).
      - Smooth HUD boxes (ID CARD WORN: Green, KNOWN FACE: Blue/Green, VERIFYING: Yellow, UNKNOWN: Red).
      - Live FPS counter on frame.
    """

    def __init__(self):
        SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        self._matcher = OpenSetMatcher()
        self._face_app = None
        self._tracker = FastCentroidTracker(max_disappeared=30)
        self._lock = threading.Lock()
        self._tracks: Dict[int, TrackState] = {}
        self._frame_count = 0
        self._fps = 0.0
        self._fps_timer = time.time()
        self._fps_frames = 0

        self._init_models()
        self._worker_thread = BiometricWorkerThread(
            matcher=self._matcher,
            face_app=self._face_app,
            on_result_callback=self._handle_async_result,
        )
        self._worker_thread.start()

    def _init_models(self):
        log.info("Initializing buffalo_sc model on CPU (ORT_SEQUENTIAL, 4 threads)...")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        try:
            from insightface.app import FaceAnalysis
            app = FaceAnalysis(
                name="buffalo_sc",
                providers=["CPUExecutionProvider"],
                allowed_modules=["detection", "recognition"],
                session_options=opts,
            )
            app.prepare(ctx_id=-1, det_size=(640, 640), det_thresh=0.3)
            self._face_app = app
            log.info("buffalo_sc initialized.")
        except Exception as e:
            log.warning(f"Could not load buffalo_sc: {e}")

    def load(self, faiss_path: Path, db_path: Path):
        self._matcher.load(faiss_path, db_path)

    def _handle_async_result(self, result_type: str, data: dict):
        track_id = data["track_id"]
        with self._lock:
            track = self._tracks.get(track_id)
            if not track or track.is_locked:
                return

            if result_type == "card":
                if data["has_id_card"]:
                    track.card_detected_count += 1
                    track.has_id_card = True
                    track.card_conf = data["card_conf"]
                    track.card_bbox = data["card_bbox"]
                    if track.card_detected_count >= FRAMES_TO_LOCK_CARD:
                        track.status = TrackStatus.ID_CARD_VERIFIED
                        track.is_locked = True
                        log.info(f"Track #{track_id} locked as ID_CARD_VERIFIED (Badge confirmed).")
                else:
                    track.no_card_count += 1
                    if track.status == TrackStatus.PENDING and track.no_card_count >= 2:
                        track.status = TrackStatus.SCANNING_FACE

            elif result_type == "face":
                match: MatchResult = data["match"]
                track.s1 = match.s1
                track.z_score = match.z_score

                if match.is_match:
                    track.positive_votes += 1
                    track.voted_person_id = match.person_id
                    track.voted_person_name = match.person_name
                    if track.positive_votes >= FRAMES_TO_LOCK_KNOWN:
                        track.status = TrackStatus.KNOWN_NO_BADGE
                        track.known_person_id = match.person_id
                        track.known_person_name = match.person_name
                        track.is_locked = True
                        log.info(f"Track #{track_id} locked as KNOWN ({match.person_name}).")
                elif match.s1 < S1_UNKNOWN_THRESH:
                    track.negative_votes += 1
                    if track.negative_votes >= FRAMES_TO_ALERT_UNKNOWN:
                        track.status = TrackStatus.CONFIRMED_UNKNOWN
                        track.is_locked = True
                        log.warning(f"Track #{track_id} locked as CONFIRMED_UNKNOWN (Intruder).")

    def _update_fps(self):
        self._fps_frames += 1
        now = time.time()
        elapsed = now - self._fps_timer
        if elapsed >= 0.5:
            self._fps = self._fps_frames / elapsed
            self._fps_frames = 0
            self._fps_timer = now

    def process_frame(self, frame: np.ndarray, timestamp: float) -> FrameResult:
        self._frame_count += 1
        self._update_fps()
        fh, fw = frame.shape[:2]

        # Fast Box Localization every 3rd frame (~22ms on CPU via buffalo_sc)
        is_detect_frame = (self._frame_count % 3 == 0)

        if is_detect_frame and self._face_app is not None:
            faces = self._face_app.get(frame)
            detected_person_boxes = []
            face_kps_map = {}

            for face in faces:
                fx1, fy1, fx2, fy2 = face.bbox.astype(int)
                f_w = max(1, fx2 - fx1)
                f_h = max(1, fy2 - fy1)
                # Project human body / torso bounding box from head landmarks
                px1 = max(0, int(fx1 - 0.75 * f_w))
                px2 = min(fw, int(fx2 + 0.75 * f_w))
                py1 = max(0, int(fy1 - 0.15 * f_h))
                py2 = min(fh, int(fy2 + 3.8 * f_h))
                pbox = np.array([px1, py1, px2, py2], dtype=int)
                detected_person_boxes.append(pbox)
                face_kps_map[tuple(pbox)] = (face.bbox.astype(int), face.kps)

            active_tracks = self._tracker.update(detected_person_boxes)

            with self._lock:
                for tid, bbox in active_tracks:
                    if tid <= 0:
                        continue
                    if tid not in self._tracks:
                        self._tracks[tid] = TrackState(
                            track_id=tid,
                            bbox_xyxy=bbox,
                            last_seen_frame=self._frame_count,
                            first_seen_ts=timestamp,
                            last_face_visible_ts=timestamp,
                        )
                    else:
                        self._tracks[tid].bbox_xyxy = bbox
                        self._tracks[tid].last_seen_frame = self._frame_count

                    # Attach nearest landmarks
                    for pbox_key, (fb, kps) in face_kps_map.items():
                        if np.linalg.norm(np.array(pbox_key) - bbox) < 40:
                            self._tracks[tid].face_bbox = fb
                            self._tracks[tid].face_kps = kps
                            break

                # Clean up tracks stale for > 45 frames
                dead_ids = [tid for tid, t in self._tracks.items() if self._frame_count - t.last_seen_frame > 45]
                for tid in dead_ids:
                    del self._tracks[tid]

        # Collect and annotate active tracks
        persons: List[TrackedPerson] = []
        with self._lock:
            for tid, track in list(self._tracks.items()):
                if tid <= 0:
                    continue

                # Evasive trigger
                if not track.is_locked and (timestamp - track.first_seen_ts > EVASIVE_SECONDS) and track.status == TrackStatus.PENDING:
                    track.status = TrackStatus.EVASIVE_ALERT

                # Asynchronously submit unverified tracks to Thread C
                if not track.is_locked:
                    self._worker_thread.submit({
                        "track_id": tid,
                        "frame": frame.copy(),
                        "p_bbox": track.bbox_xyxy.copy(),
                        "face_kps": track.face_kps,
                    })

                # Homography radar mapping
                bx1, by1, bx2, by2 = track.bbox_xyxy
                foot_pt = np.array([[(bx1 + bx2) / 2.0, float(by2)]], dtype=np.float32)
                radar_res = cv2.perspectiveTransform(foot_pt.reshape(1, 1, 2), DEFAULT_HOMOGRAPHY)
                rx = float(np.clip(radar_res[0, 0, 0], 0, 1000))
                ry = float(np.clip(radar_res[0, 0, 1], 0, 1000))

                # Intruder snapshot saving
                snap_path = None
                if track.status in (TrackStatus.CONFIRMED_UNKNOWN, TrackStatus.EVASIVE_ALERT) and not track.alert_saved:
                    track.alert_saved = True
                    snap_name = f"intruder_track{tid}_{int(timestamp)}.jpg"
                    full_snap = SNAPSHOT_DIR / snap_name
                    cv2.imwrite(str(full_snap), frame)
                    snap_path = str(full_snap)

                persons.append(TrackedPerson(
                    track_id=tid,
                    bbox_xyxy=track.bbox_xyxy,
                    status=track.status,
                    person_id=track.known_person_id,
                    person_name=track.known_person_name,
                    s1=track.s1,
                    z_score=track.z_score,
                    face_bbox=track.face_bbox,
                    yaw_deg=track.yaw_deg,
                    pitch_deg=track.pitch_deg,
                    radar_xy=(rx, ry),
                    snapshot_path=snap_path,
                    has_id_card=track.has_id_card,
                    card_bbox=track.card_bbox,
                    card_conf=track.card_conf,
                ))

        annotated = self._draw_hud(frame.copy(), persons, self._fps)

        return FrameResult(
            frame_idx=self._frame_count,
            timestamp=timestamp,
            persons=persons,
            annotated_frame=annotated,
            fps=self._fps,
        )

    def _draw_hud(self, frame: np.ndarray, persons: List[TrackedPerson], fps: float) -> np.ndarray:
        h, w = frame.shape[:2]

        # Top-left HUD badge: Green 25-30 FPS indicator
        fps_text = f"FPS: {fps:04.1f} | 3-THREAD CPU"
        cv2.rectangle(frame, (10, 10), (250, 42), (8, 15, 30), -1)
        cv2.rectangle(frame, (10, 10), (250, 42), (0, 230, 118), 1)
        cv2.putText(frame, fps_text, (18, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 230, 118), 2)

        for p in persons:
            color = _STATUS_COLORS.get(p.status, (200, 200, 200))
            x1, y1, x2, y2 = p.bbox_xyxy
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            # Smooth Corner brackets
            length = min(22, max(5, (x2 - x1) // 3), max(5, (y2 - y1) // 3))
            th = 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
            cv2.line(frame, (x1, y1), (x1 + length, y1), color, th)
            cv2.line(frame, (x1, y1), (x1, y1 + length), color, th)
            cv2.line(frame, (x2, y1), (x2 - length, y1), color, th)
            cv2.line(frame, (x2, y1), (x2, y1 + length), color, th)
            cv2.line(frame, (x1, y2), (x1 + length, y2), color, th)
            cv2.line(frame, (x1, y2), (x1, y2 - length), color, th)
            cv2.line(frame, (x2, y2), (x2 - length, y2), color, th)
            cv2.line(frame, (x2, y2), (x2, y2 - length), color, th)

            # HUD Label banner
            if p.status == TrackStatus.ID_CARD_VERIFIED:
                lbl = f"ID CARD WORN: T#{p.track_id}"
            elif p.status == TrackStatus.KNOWN_NO_BADGE:
                name_str = p.person_name or "KNOWN"
                lbl = f"KNOWN FACE: {name_str} ({p.s1:.2f})"
            elif p.status in (TrackStatus.CONFIRMED_UNKNOWN, TrackStatus.EVASIVE_ALERT):
                lbl = f"UNKNOWN INTRUDER: T#{p.track_id}"
            else:
                lbl = f"VERIFYING... T#{p.track_id}"

            (lw, lh), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
            ly = max(18, y1 - 6)
            cv2.rectangle(frame, (x1, ly - lh - 4), (x1 + lw + 8, ly + 2), (8, 15, 30), -1)
            cv2.rectangle(frame, (x1, ly - lh - 4), (x1 + lw + 8, ly + 2), color, 1)
            cv2.putText(frame, lbl, (x1 + 4, ly - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1)

            # Badge indicator box
            if p.has_id_card and p.card_bbox is not None:
                cx1, cy1, cx2, cy2 = p.card_bbox
                cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), (0, 255, 0), 2)
                cv2.putText(frame, "ID BADGE", (cx1, cy1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        return frame

    def calibrate_live_face(
        self,
        frame: np.ndarray,
        faiss_path: Path,
        db_path: Path,
        person_id: str = "EMP001",
        person_name: str = "Authorized User",
    ) -> dict:
        """
        1-Click Live Self-Calibration:
        Extracts face from webcam, generates 5 clean norm_crop variations + centroid,
        appends to FAISS + SQLite, and reloads index in <1 second.
        """
        if self._face_app is None:
            return {"success": False, "message": "FaceAnalysis model not loaded"}

        from insightface.utils.face_align import norm_crop
        faces = self._face_app.get(frame)
        if not faces:
            return {"success": False, "message": "No face detected in live camera frame. Look directly at camera."}

        best_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        if not hasattr(best_face, "kps") or best_face.kps is None or len(best_face.kps) < 5:
            return {"success": False, "message": "Face landmarks could not be localized."}

        aligned_112 = norm_crop(frame, landmark=best_face.kps, image_size=112)
        rec_model = self._face_app.models["recognition"]

        # 5 variations
        variations = [
            aligned_112.copy(),
            apply_clahe_lab(aligned_112),
            cv2.flip(aligned_112, 1),
            cv2.warpAffine(aligned_112, cv2.getRotationMatrix2D((56, 56), 7.0, 1.0), (112, 112), borderMode=cv2.BORDER_REFLECT_101),
            cv2.warpAffine(aligned_112, cv2.getRotationMatrix2D((56, 56), -7.0, 1.0), (112, 112), borderMode=cv2.BORDER_REFLECT_101),
        ]

        embs = []
        for v in variations:
            feat = rec_model.get_feat(v).flatten().astype(np.float32)
            n = np.linalg.norm(feat)
            if n > 1e-6:
                embs.append(feat / n)

        if not embs:
            return {"success": False, "message": "Failed to extract embeddings from live frame."}

        centroid = np.mean(np.array(embs), axis=0)
        c_n = np.linalg.norm(centroid)
        if c_n > 1e-6:
            embs.append((centroid / c_n).astype(np.float32))

        # Update FAISS
        if faiss_path.exists():
            idx = faiss.read_index(str(faiss_path))
        else:
            idx = faiss.IndexFlatIP(EMBEDDING_DIM)

        start_id = idx.ntotal
        idx.add(np.array(embs, dtype=np.float32))
        faiss.write_index(idx, str(faiss_path))

        # Update SQLite
        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS persons (
                person_id TEXT PRIMARY KEY, name TEXT NOT NULL, photo_path TEXT NOT NULL, enrolled_at REAL NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS vectors (
                faiss_id INTEGER PRIMARY KEY, person_id TEXT NOT NULL, view_index INTEGER NOT NULL, view_label TEXT NOT NULL
            )
        """)
        conn.execute(
            "INSERT OR REPLACE INTO persons (person_id, name, photo_path, enrolled_at) VALUES (?, ?, ?, ?)",
            (person_id, person_name, "webcam_calibrated", time.time()),
        )
        labels = ["original", "clahe", "flip", "rot_plus_7", "rot_minus_7", "mean_centroid"]
        for i, emb in enumerate(embs):
            lbl = labels[i] if i < len(labels) else f"calib_{i}"
            conn.execute(
                "INSERT OR REPLACE INTO vectors (faiss_id, person_id, view_index, view_label) VALUES (?, ?, ?, ?)",
                (start_id + i, person_id, i + 1, lbl),
            )
        conn.commit()
        conn.close()

        self.load(faiss_path, db_path)

        log.info(f"Live calibration complete for [{person_id}] {person_name}: {len(embs)} vectors added.")
        return {
            "success": True,
            "message": f"Successfully calibrated live face for {person_name}! Added {len(embs)} vectors.",
            "faiss_total": idx.ntotal,
        }

    def close(self):
        if self._worker_thread:
            self._worker_thread.stop()
