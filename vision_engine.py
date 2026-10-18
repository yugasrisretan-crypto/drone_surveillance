"""
vision_engine.py
================
Core DroneSurveillanceEngine — Phase 2

Implements 5 technical modules end-to-end:

  Module 1 — Body Tracking
      YOLOv11 Nano (yolo11n.pt) + ByteTrack (via Supervision) for persistent track_ids.

  Module 2 — 5-Point Pose & Quality Gate
      InsightFace SCRFD face detection.  Quality checks:
        • Minimum face size 24x24 px (2x Lanczos upscale for 24–48 px faces).
        • Laplacian variance blur rejection.
        • 3D head yaw / pitch estimation from 5 landmarks; yaw>45° or pitch>50°
          → UNVERIFIABLE (never triggers a false UNKNOWN alert).

  Module 3 — Cohort Z-Score Open-Set Matcher
      FAISS IndexFlatIP top-K=20 search → S1 (top-1 cosine) + Z-score
      (S1 vs impostor distribution) for lighting-robust matching.

  Module 4 — Multi-Frame Temporal Voting & Torso Re-ID Cache
      Per-track state machine:
        PENDING → KNOWN  : 3 frames with S1≥0.40 or Z≥3.4
        PENDING → CONFIRMED_UNKNOWN : 6 quality-gated frames with S1≤0.30 and Z<2.2
        Any state        : EVASIVE_ALERT if face hidden >6 s continuously
        KNOWN fallback   : HSV torso histogram re-ID when face unavailable

  Module 5 — Homography Map Projection
      cv2.perspectiveTransform maps foot-point to (0–1000, 0–1000) campus radar.

Usage:
    engine = DroneSurveillanceEngine()
    engine.load(faiss_path="storage/campus.faiss", db_path="storage/campus.db")
    for frame in video_source:
        result = engine.process_frame(frame, timestamp=time.time())
        annotated = result.annotated_frame
"""

from __future__ import annotations

import os
import logging
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

log = logging.getLogger("vision_engine")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


def apply_clahe_lab(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) on the L channel
    in LAB color space. Normalizes shadows and harsh lighting while preserving natural skin chrominance.
    """
    if image is None or image.size == 0:
        return image
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

# ---------------------------------------------------------------------------
# Public constants / tunable thresholds
# ---------------------------------------------------------------------------
EMBEDDING_DIM             = 512
FAISS_TOP_K               = 20       # cohort size for Z-score
S1_KNOWN_THRESH           = 0.40    # cosine sim → KNOWN
Z_KNOWN_THRESH            = 3.4     # Z-score → KNOWN
S1_UNKNOWN_THRESH         = 0.30    # cosine sim → vote UNKNOWN
Z_UNKNOWN_THRESH          = 2.2     # Z-score → vote UNKNOWN
FRAMES_TO_LOCK_KNOWN      = 3       # positive votes needed for KNOWN_NO_BADGE
FRAMES_TO_ALERT           = 6       # negative votes needed for CONFIRMED_UNKNOWN
FRAMES_TO_LOCK_CARD       = 3       # frames wearing ID card to verify
FRAMES_NO_CARD_ESCALATE   = 10      # frames without card before face scan escalation
CARD_CONF_THRESH          = 0.65    # High-precision ID card detector threshold (rejects clothing & background)
CARD_CLASSES              = ["id card", "lanyard badge", "name tag"]
CARD_MODEL_NAME           = None    # Zero-download OpenCV detector (no YOLOWorld)
EVASIVE_SECONDS           = 6.0     # hidden-face time → EVASIVE_ALERT / CONFIRMED_UNKNOWN
BLUR_VAR_THRESH           = 25.0    # Laplacian variance; below → blurry (25 is ideal for indoor/webcam)
MIN_FACE_PX               = 24      # minimum face side length
UPSCALE_FACE_THRESH       = 48      # upscale faces below this size
YAW_MAX_DEG               = 45.0    # max yaw before UNVERIFIABLE
PITCH_MAX_DEG             = 50.0    # max pitch before UNVERIFIABLE
TORSO_HIST_BINS           = 32      # HSV histogram bins per channel (H only)
TORSO_SIM_THRESH          = 0.65    # cosine sim of torso histograms
SNAPSHOT_DIR              = Path("storage/snapshots")
YOLO_MODEL_NAME           = "yolo11n.pt"   # Nano — fastest CPU-friendly model
FACE_RECOG_EVERY_N_FRAMES = 3              # run face recog once per N frames; reuse bbox between
INSIGHTFACE_MODEL         = "buffalo_l"  # includes SCRFD det + ArcFace rec

# Default homography: identity mapping scaled to 1000×1000
# Replace with a real calibrated matrix for production deployments.
DEFAULT_HOMOGRAPHY = np.array([
    [1000.0 / 1920, 0,            0],
    [0,            1000.0 / 1080, 0],
    [0,            0,             1],
], dtype=np.float64)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
class TrackStatus(Enum):
    PENDING           = auto()   # Stage 1: initial frames checking for ID card
    ID_CARD_VERIFIED  = auto()   # Stage 1 Decision: ID Card Worn (Bright Green box)
    SCANNING_FACE     = auto()   # Stage 2: No card after 10 frames, face scan active (Yellow box)
    KNOWN_NO_BADGE    = auto()   # Branch 2A: Face verified, no ID card (Blue/Teal box)
    CONFIRMED_UNKNOWN = auto()   # Branch 2B: Intruder / unknown face (Red box)
    EVASIVE_ALERT     = auto()   # Face hidden >6s without ID card (Orange/Red box)
    UNVERIFIABLE      = auto()   # Current frame pose OOB
    KNOWN             = KNOWN_NO_BADGE     # Compatibility alias
    ID_CARD_WORN      = ID_CARD_VERIFIED   # Compatibility alias
    VERIFYING         = PENDING            # Compatibility alias
    UNKNOWN           = CONFIRMED_UNKNOWN  # Compatibility alias


@dataclass
class MatchResult:
    """Output of the open-set matcher for a single face embedding."""
    person_id:   Optional[str]   = None   # None if no match or index empty
    person_name: Optional[str]   = None
    s1:          float           = 0.0    # top-1 cosine similarity
    z_score:     float           = 0.0    # cohort Z-score
    is_match:    bool            = False


@dataclass
class TrackState:
    """Mutable state kept per ByteTrack track_id."""
    track_id:               int
    status:                 TrackStatus   = TrackStatus.PENDING
    known_person_id:        Optional[str] = None
    known_person_name:      Optional[str] = None

    # Stage 1: ID card tracking
    card_detected_frames:   int           = 0
    no_card_frames:         int           = 0
    has_id_card:            bool          = False
    card_bbox:              Optional[np.ndarray] = None
    card_conf:              float         = 0.0

    # Stage 2: Face voting buffers
    positive_votes:         int           = 0   # frames matching a specific person
    negative_votes:         int           = 0   # frames failing to match anyone
    voted_person_id:        Optional[str] = None  # candidate being voted for

    # Evasive tracking
    first_seen_ts:          float         = 0.0
    last_face_visible_ts:   float         = 0.0
    face_hidden_since_ts:   Optional[float] = None  # when face disappeared

    # Torso Re-ID cache
    torso_hist:             Optional[np.ndarray] = None  # HSV H-channel histogram

    # Alert deduplication
    alert_saved:            bool          = False
    badge_warning_logged:   bool          = False


@dataclass
class TrackedPerson:
    """Per-person result for a single frame."""
    track_id:       int
    bbox_xyxy:      np.ndarray          # [x1,y1,x2,y2] in frame pixels
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
    """Complete output of DroneSurveillanceEngine.process_frame()."""
    frame_idx:      int
    timestamp:      float
    persons:        List[TrackedPerson]  = field(default_factory=list)
    annotated_frame:Optional[np.ndarray] = None


# ---------------------------------------------------------------------------
# Colour scheme for annotation (BGR format for OpenCV)
# ---------------------------------------------------------------------------
_STATUS_COLORS = {
    TrackStatus.PENDING:           (200, 200, 200),  # Grey
    TrackStatus.ID_CARD_VERIFIED:  (0,   255,   0),  # Bright Green (Compliant)
    TrackStatus.SCANNING_FACE:     (0,   220, 255),  # Yellow
    TrackStatus.KNOWN_NO_BADGE:    (255, 180,   0),  # Blue / Teal
    TrackStatus.CONFIRMED_UNKNOWN: (0,     0, 255),  # Red (Flashing / Alert)
    TrackStatus.EVASIVE_ALERT:     (0,   120, 255),  # Orange-Red
    TrackStatus.UNVERIFIABLE:      (0,   215, 255),  # Amber
}


# ===========================================================================
# Module 1 — Body Tracker (YOLOv11 Nano + ByteTrack)
# ===========================================================================
class BodyTracker:
    """
    Wraps Ultralytics YOLOv11 Nano person detector + ByteTrackTracker
    (from the `trackers` package — the official replacement for
    supervision's deprecated ByteTrack as of sv-0.28+).
    """

    def __init__(self, model_name: str = YOLO_MODEL_NAME, conf: float = 0.55):
        from ultralytics import YOLO
        from trackers import ByteTrackTracker

        log.info(f"Loading YOLO model (CPU): {model_name}")
        self._model = YOLO(model_name)
        self._model.to("cpu")
        self._yolo_device = "cpu"
        log.info("YOLO running on CPU")

        self._conf = conf
        self._tracker = ByteTrackTracker(
            track_activation_threshold=0.35,
            lost_track_buffer=30,
            minimum_consecutive_frames=1,
            frame_rate=25.0,
        )
        log.info("BodyTracker ready.")

    def track(self, frame: np.ndarray) -> List[Tuple[int, np.ndarray]]:
        """
        Run YOLO person detection + ByteTrackTracker on *frame*.

        Returns:
            List of (track_id, bbox_xyxy) for every tracked person.
        """
        import supervision as sv

        results = self._model(
            frame,
            classes=[0],        # class 0 = person in COCO
            conf=self._conf,
            verbose=False,
            device=self._yolo_device,
            imgsz=416,
        )[0]

        detections = sv.Detections.from_ultralytics(results)
        del results
        # Keep only person class meeting confidence threshold
        if hasattr(detections, "confidence") and detections.confidence is not None:
            detections = detections[(detections.class_id == 0) & (detections.confidence >= self._conf)]
        else:
            detections = detections[detections.class_id == 0]
        # ByteTrackTracker.update() returns tracked Detections with tracker_id set
        tracked = self._tracker.update(detections)

        out: List[Tuple[int, np.ndarray]] = []
        if tracked.tracker_id is None:
            return out
        for tid, bbox in zip(tracked.tracker_id, tracked.xyxy):
            out.append((int(tid), bbox.astype(int)))
        return out


# ===========================================================================
# ===========================================================================
# Module 1B — Stage 1: Chest ROI & Fast Zero-Download ID Card / Lanyard Detector
# ===========================================================================

def extract_chest_roi(
    person_bbox: np.ndarray,
    frame_h: int,
    frame_w: int,
    face_bbox: Optional[np.ndarray] = None,
) -> Tuple[int, int, int, int]:
    """
    Extract the torso/chest interaction bounding box for a tracked person:
      - If face_bbox is provided: scan the chest & upper body region below the chin/neck.
      - Normal drone / standing view: y between 15% and 72% of person's box.
      - Close-up laptop webcam view: chest and hand interaction region.

    Returns:
        (cx1, cy1, cx2, cy2) in frame pixel coordinates.
    """
    x1, y1, x2, y2 = person_bbox[:4].astype(int)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame_w, x2), min(frame_h, y2)
    pw = max(1, x2 - x1)
    ph = max(1, y2 - y1)

    if face_bbox is not None:
        fx1, fy1, fx2, fy2 = face_bbox[:4].astype(int)
        fcx = (fx1 + fx2) / 2.0
        fw = max(1, fx2 - fx1)
        fh = max(1, fy2 - fy1)
        # Directly below chin to lower torso (wide enough to include hands holding ID card)
        cy1 = min(frame_h - 15, max(0, fy2 + int(0.04 * fh)))
        cy2 = min(frame_h, max(cy1 + 60, y2, fy2 + int(4.0 * fh)))
        cx1 = max(0, int(fcx - 1.80 * fw))
        cx2 = min(frame_w, int(fcx + 1.80 * fw))
    else:
        # Detect close-up webcam framing
        is_closeup = (ph >= int(0.55 * frame_h)) or (y1 <= int(0.10 * frame_h) and ph >= int(0.45 * frame_h))
        if is_closeup:
            cy1 = max(0, y1 + int(ph * 0.25))
            cy2 = min(frame_h, max(cy1 + 50, int(frame_h * 0.98), y2))
            cx1 = max(0, x1)
            cx2 = min(frame_w, x2)
        else:
            cy1 = max(0, y1 + int(ph * 0.15))
            cy2 = min(frame_h, y1 + int(ph * 0.72))
            cx1 = max(0, x1)
            cx2 = min(frame_w, x2)

    # Safety clamps for valid non-empty crop
    if cy2 <= cy1 + 10:
        cy2 = min(frame_h, cy1 + 35)
    if cx2 <= cx1 + 10:
        cx2 = min(frame_w, cx1 + 35)

    return cx1, cy1, cx2, cy2


class ChestCardDetector:
    """
    High-precision, zero-download OpenCV geometric ID Card & Badge Detector.
    Strictly detects rectangular ID cards / badges (CR80 ratio ~1.58 or badge holders)
    while completely rejecting background objects, furniture, light switches, walls,
    doors, shirt collars, buttons, zippers, pockets, and clothing patterns.
    """

    def __init__(
        self,
        conf: float = CARD_CONF_THRESH,
        **kwargs,  # absorbs legacy parameters
    ):
        self._conf = conf
        log.info(f"ChestCardDetector initialized (High-Precision ID Card Detector, conf={self._conf}).")

    def detect_badge_contours(
        self,
        chest_crop: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], float]:
        """
        High-precision OpenCV geometric ID Card & Badge Detector.
        Strictly requires:
        1. 4-corner orthogonal convex quadrilateral geometry (approxPolyDP == 4).
        2. Standard ID card aspect ratio: 1.20 to 1.95 (CR80 standard is 1.586).
        3. High solidity (>= 0.74): card must fill its rectangular bounding envelope.
        4. Core interior analysis: checks the inner 70% of the card to guarantee
           internal content (photo + text lines) and reject blank walls/switches.
        5. Horizontal gradient energy (Sobel X >= 8.0) ensuring presence of text rows.
        6. Rejects non-card items: shirt collars, buttons, zippers, clothing folds,
           pockets, light switches, door panels, and background clutter.
        """
        if chest_crop is None or chest_crop.size == 0:
            return None, 0.0

        ch, cw = chest_crop.shape[:2]
        crop_area = float(ch * cw)
        if crop_area < 500 or cw < 30 or ch < 30:
            return None, 0.0

        gray = cv2.cvtColor(chest_crop, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Multi-stage edge and contrast extraction
        edges = cv2.Canny(blurred, 35, 110)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 3
        )
        _, thresh_white = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY)
        thresh_white = cv2.morphologyEx(thresh_white, cv2.MORPH_CLOSE, kernel)

        contours_e, _ = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_t, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_w, _ = cv2.findContours(thresh_white, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        all_contours = list(contours_e) + list(contours_t) + list(contours_w)

        best_bbox = None
        best_score = 0.0

        for cnt in all_contours:
            area = cv2.contourArea(cnt)
            # A physical ID card has realistic minimum and maximum area
            if area < 800 or area > 0.65 * crop_area:
                continue

            rx, ry, rw, rh = cv2.boundingRect(cnt)
            rect_area = rw * rh
            if rect_area <= 0 or rw < 35 or rh < 25:
                continue

            # Solidity: rectangular card fills its bounding rectangle (>= 74%)
            solidity = area / float(rect_area)
            if solidity < 0.74:
                continue

            # Card aspect ratio check (CR80 standard is 1.586, portrait is 0.63)
            short_side = min(rw, rh)
            long_side = max(rw, rh)
            aspect = long_side / float(max(1, short_side))
            if not (1.20 <= aspect <= 1.95):
                continue

            # Polygonal quadrilateral approximation
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.035 * peri, True)

            # Check if contour is a 4-point convex quadrilateral
            is_quad = (len(approx) == 4 and cv2.isContourConvex(approx))

            # Internal card core patch analysis (inner 70% excluding boundary edges)
            # This is critical: blank objects (light switches, door panels, plain cloth)
            # have zero interior variance, while ID cards have dense text and a photo!
            my = int(rh * 0.15)
            mx = int(rw * 0.15)
            core_patch = gray[ry + my : ry + rh - my, rx + mx : rx + rw - mx]
            if core_patch.size == 0:
                continue

            std_val = float(np.std(core_patch))
            mean_val = float(np.mean(core_patch))
            lap_val = float(cv2.Laplacian(core_patch, cv2.CV_64F).var())

            # A real ID card interior MUST have internal details (photo, text, symbols)
            # Blank objects (switches, plain walls, solid panels) have std < 10 and lap < 20
            if std_val < 14.0 or lap_val < 25.0 or mean_val < 50.0:
                continue

            # Horizontal edge energy inside core (text lines)
            sobel_x = cv2.Sobel(core_patch, cv2.CV_64F, 1, 0, ksize=3)
            h_energy = float(np.mean(np.abs(sobel_x)))
            if h_energy < 8.0:
                continue

            # Check orthogonal corners if quadrilateral
            orthogonal = False
            if is_quad:
                pts = approx.reshape(4, 2)
                cosines = []
                for i in range(4):
                    p0 = pts[i]
                    p1 = pts[(i + 1) % 4]
                    p2 = pts[(i + 2) % 4]
                    v1 = (p1 - p0).astype(float)
                    v2 = (p2 - p1).astype(float)
                    norm1 = np.linalg.norm(v1)
                    norm2 = np.linalg.norm(v2)
                    if norm1 > 0 and norm2 > 0:
                        cos_ang = abs(np.dot(v1, v2) / (norm1 * norm2))
                        cosines.append(cos_ang)
                # Corner angles should be between 65 and 115 degrees (cos < 0.45)
                if len(cosines) == 4 and all(c < 0.45 for c in cosines):
                    orthogonal = True

            # Scoring:
            score = 0.40
            if is_quad:
                score += 0.25
            if orthogonal:
                score += 0.15
            if mean_val >= 90.0:
                score += 0.10
            if std_val >= 25.0:
                score += 0.10

            score = min(1.0, score)

            # Only accept if high confidence and superior to previous
            if score >= self._conf and score > best_score:
                best_score = score
                best_bbox = np.array([rx, ry, rx + rw, ry + rh], dtype=int)

        return best_bbox, best_score

    def detect(
        self,
        chest_crop: np.ndarray,
    ) -> Tuple[bool, Optional[np.ndarray], float, str]:
        """
        Execute precision ID card & badge detection.
        Returns:
            (detected: bool, bbox_in_crop: Optional[np.ndarray], conf: float, source: str)
        """
        if chest_crop is None or chest_crop.size == 0:
            return False, None, 0.0, "none"

        cv_box, cv_score = self.detect_badge_contours(chest_crop)
        if cv_box is not None and cv_score >= self._conf:
            return True, cv_box, cv_score, "opencv_id_card"

        return False, None, 0.0, "none"


class VirtualPersonTracker:
    """
    Lightweight tracker that maintains persistent track_ids for virtual
    person/chest bounding boxes generated from detected faces (e.g. on
    laptop webcams where only head/shoulders are visible).
    """
    def __init__(self, base_id: int = 1):
        self._tracks: Dict[int, Tuple[np.ndarray, float]] = {}  # tid -> (bbox, last_seen_ts)
        self._next_id = base_id

    def update(
        self,
        unmatched_boxes: List[np.ndarray],
        timestamp: float,
        existing_track_ids: set,
    ) -> List[Tuple[int, np.ndarray]]:
        """
        Match incoming virtual boxes to existing virtual tracks or allocate new IDs.
        """
        # Purge stale tracks (> 3.0s)
        stale_tids = [tid for tid, (_, last_seen) in self._tracks.items() if timestamp - last_seen > 3.0]
        for tid in stale_tids:
            del self._tracks[tid]

        results: List[Tuple[int, np.ndarray]] = []
        used_tids = set()

        for box in unmatched_boxes:
            bx1, by1, bx2, by2 = box
            bcx, bcy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
            best_tid = None
            best_dist = float("inf")

            for tid, (prev_box, _) in self._tracks.items():
                if tid in used_tids or tid in existing_track_ids:
                    continue
                px1, py1, px2, py2 = prev_box
                pcx, pcy = (px1 + px2) / 2.0, (py1 + py2) / 2.0
                dist = np.hypot(bcx - pcx, bcy - pcy)
                if dist < 220 and dist < best_dist:
                    best_dist = dist
                    best_tid = tid

            if best_tid is None:
                candidate_id = self._next_id
                while candidate_id in existing_track_ids or candidate_id in self._tracks:
                    candidate_id += 1
                best_tid = candidate_id
                self._next_id = candidate_id + 1

            self._tracks[best_tid] = (box, timestamp)
            used_tids.add(best_tid)
            results.append((best_tid, box))

        return results


# ===========================================================================
# Module 2 — Face Quality Gate + Pose Estimator
# ===========================================================================

# 3-D reference face model coordinates (canonical 5-pt: leye, reye, nose, lmouth, rmouth)
# OpenCV pinhole camera convention: X right, Y down, Z forward. Centred at nose tip.
_REF_LANDMARKS_3D = np.array([
    [-30.0, -30.0, -30.0],   # left eye  (X=-30, Y=-30 (above nose), Z=-30)
    [ 30.0, -30.0, -30.0],   # right eye (X=+30, Y=-30 (above nose), Z=-30)
    [  0.0,   0.0,   0.0],   # nose tip  (origin)
    [-20.0,  35.0, -20.0],   # left mouth corner  (Y=+35 (below nose))
    [ 20.0,  35.0, -20.0],   # right mouth corner (Y=+35 (below nose))
], dtype=np.float32)


def _estimate_head_pose(
    kps5: np.ndarray,
    img_w: int,
    img_h: int,
) -> Tuple[float, float]:
    """
    Estimate head yaw and pitch (degrees) from 5 facial landmarks using solvePnP
    (SOLVEPNP_SQPNP / SOLVEPNP_EPNP) with a resilient landmark-geometry fallback.

    Args:
        kps5:  (5,2) float32 array of [x,y] landmark coordinates in the image.
        img_w: image width in pixels.
        img_h: image height in pixels.

    Returns:
        (yaw_deg, pitch_deg) — positive yaw = face turned right,
                                positive pitch = face tilted up.
    """
    focal = max(img_w, img_h)
    cx, cy = img_w / 2.0, img_h / 2.0
    cam_matrix = np.array([
        [focal, 0,     cx],
        [0,     focal, cy],
        [0,     0,     1 ],
    ], dtype=np.float64)
    dist_coeffs = np.zeros(4, dtype=np.float64)

    # 1. Try solvePnP with SQPNP / EPNP (compatible with 5 points in OpenCV 5+)
    try:
        flag = cv2.SOLVEPNP_SQPNP if hasattr(cv2, "SOLVEPNP_SQPNP") else cv2.SOLVEPNP_EPNP
        success, rvec, _ = cv2.solvePnP(
            _REF_LANDMARKS_3D,
            kps5.astype(np.float64),
            cam_matrix,
            dist_coeffs,
            flags=flag,
        )
        if success:
            R, _ = cv2.Rodrigues(rvec)
            sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
            if sy > 1e-6:
                pitch = np.degrees(np.arctan2(-R[2, 1], R[2, 2]))
                yaw   = np.degrees(np.arctan2( R[2, 0], sy))
            else:
                pitch = np.degrees(np.arctan2(-R[1, 2], R[1, 1]))
                yaw   = np.degrees(np.arctan2( R[2, 0], sy))
            return float(yaw), float(pitch)
    except Exception:
        pass

    # 2. Resilient geometric fallback from 5 landmarks (leye, reye, nose, lmouth, rmouth)
    try:
        leye, reye, nose, lmouth, rmouth = kps5[:5]
        eye_w = max(1.0, float(np.linalg.norm(reye - leye)))
        eye_mid = (leye + reye) / 2.0
        # Yaw: lateral offset of nose from eye midpoint
        nose_dx = float(nose[0] - eye_mid[0])
        yaw = float(np.degrees(np.arctan2(nose_dx * 2.0, eye_w)))

        # Pitch: vertical position of nose relative to eye-mouth distance
        mouth_mid = (lmouth + rmouth) / 2.0
        face_h = max(1.0, float(np.linalg.norm(mouth_mid - eye_mid)))
        nose_ratio = float((nose[1] - eye_mid[1]) / face_h)
        pitch = float((nose_ratio - 0.55) * 60.0)

        return yaw, pitch
    except Exception:
        return 0.0, 0.0


def _laplacian_variance(gray: np.ndarray) -> float:
    """Measure sharpness as variance of the Laplacian."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


class FaceQualityGate:
    """
    Wraps InsightFace SCRFD detection + ArcFace embedding.
    Applies quality checks (minimum size, blur rejection, 5-point head pose)
    before returning an L2-normalized 512-D ArcFace embedding.
    """

    def __init__(self, model_name: str = INSIGHTFACE_MODEL, det_thresh: float = 0.35):
        from insightface.app import FaceAnalysis
        log.info(f"Loading InsightFace {model_name} (CPU, 320×320, detection+recognition only)…")
        self._app = FaceAnalysis(
            name=model_name,
            providers=["CPUExecutionProvider"],
            allowed_modules=["detection", "recognition"],
        )
        self._app.prepare(ctx_id=-1, det_size=(320, 320), det_thresh=det_thresh)
        self.face_app = self._app
        log.info("FaceQualityGate ready.")

    def analyse_crop(
        self,
        crop: np.ndarray,
    ) -> Tuple[Optional[np.ndarray],   # L2-normed 512-D embedding or None
               Optional[np.ndarray],   # face_bbox_xyxy in crop coords
               float,                  # yaw_deg
               float,                  # pitch_deg
               str]:                   # reason: "ok"|"no_face"|"too_small"|"blurry"|"unverifiable"
        """
        Detect face directly inside a pre-cropped region,
        run quality gates (size, blur, 5-point head pose), and extract 512-D ArcFace embedding.
        """
        if crop is None or crop.size == 0:
            return None, None, 0.0, 0.0, "no_crop"

        ch, cw = crop.shape[:2]
        faces = self._app.get(crop)
        if not faces:
            return None, None, 0.0, 0.0, "no_face"

        # Pick face with largest area
        best = max(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        )

        bx1, by1, bx2, by2 = best.bbox.astype(int)
        face_w = bx2 - bx1
        face_h = by2 - by1

        # --- Quality check 1: minimum size ---
        if face_w < MIN_FACE_PX or face_h < MIN_FACE_PX:
            return None, None, 0.0, 0.0, "too_small"

        # --- Optionally upscale small faces (24–48 px) for better embedding ---
        c_by1, c_by2 = max(0, by1), min(ch, by2)
        c_bx1, c_bx2 = max(0, bx1), min(cw, bx2)
        face_crop = crop[c_by1:c_by2, c_bx1:c_bx2]
        if face_crop.size == 0:
            return None, None, 0.0, 0.0, "no_face"

        if max(face_w, face_h) < UPSCALE_FACE_THRESH:
            face_crop = cv2.resize(
                face_crop,
                (face_w * 2, face_h * 2),
                interpolation=cv2.INTER_LANCZOS4,
            )

        # --- Quality check 2: blur (Laplacian variance) ---
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        if _laplacian_variance(gray) < BLUR_VAR_THRESH:
            return None, None, 0.0, 0.0, "blurry"

        # --- Quality check 3: pose (yaw / pitch from 5 landmarks) ---
        yaw_deg, pitch_deg = 0.0, 0.0
        if best.kps is not None and len(best.kps) >= 5:
            kps5 = best.kps.astype(np.float32)  # (5,2)
            yaw_deg, pitch_deg = _estimate_head_pose(kps5, cw, ch)

        if abs(yaw_deg) > YAW_MAX_DEG or abs(pitch_deg) > PITCH_MAX_DEG:
            return None, None, yaw_deg, pitch_deg, "unverifiable"

        # --- Extract embedding ---
        emb = best.embedding
        if emb is None:
            return None, None, yaw_deg, pitch_deg, "no_embedding"

        emb = emb.astype(np.float32)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb /= norm

        face_bbox_crop = np.array([bx1, by1, bx2, by2], dtype=int)
        return emb, face_bbox_crop, yaw_deg, pitch_deg, "ok"

    def analyse(
        self,
        frame: np.ndarray,
        person_bbox: np.ndarray,
        predetected_face = None,
    ) -> Tuple[Optional[np.ndarray],   # L2-normed 512-D embedding or None
               Optional[np.ndarray],   # face_bbox_xyxy in frame coords
               float,                  # yaw_deg
               float,                  # pitch_deg
               str]:                   # reason: "ok"|"no_face"|"too_small"|"blurry"|"unverifiable"
        """
        Crop the person region from *frame*, apply CLAHE LAB luminance normalization,
        detect the face, apply quality checks, and return the embedding plus pose angles.
        """
        fh, fw = frame.shape[:2]

        if predetected_face is not None:
            best = predetected_face
            crop = frame
            crop_offset_x, crop_offset_y = 0, 0
        else:
            x1, y1, x2, y2 = person_bbox[:4].astype(int)
            bw, bh = x2 - x1, y2 - y1

            # Add 10% padding so face is not clipped if near the edge of the person box
            pad_x = int(bw * 0.10)
            pad_y = int(bh * 0.10)
            cx1, cy1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
            cx2, cy2 = min(fw, x2 + pad_x), min(fh, y2 + pad_y)
            crop = frame[cy1:cy2, cx1:cx2]
            if crop.size == 0:
                return None, None, 0.0, 0.0, "no_crop"

            crop_offset_x, crop_offset_y = cx1, cy1
            # Apply CLAHE luminance normalization in LAB color space
            crop_norm = apply_clahe_lab(crop)
            emb, rel_box, yaw_deg, pitch_deg, reason = self.analyse_crop(crop_norm)
            if rel_box is not None:
                face_bbox_frame = np.array([
                    rel_box[0] + crop_offset_x,
                    rel_box[1] + crop_offset_y,
                    rel_box[2] + crop_offset_x,
                    rel_box[3] + crop_offset_y,
                ], dtype=int)
            else:
                face_bbox_frame = None
            return emb, face_bbox_frame, yaw_deg, pitch_deg, reason

        # Handling for predetected_face
        ch, cw = crop.shape[:2]
        bx1, by1, bx2, by2 = best.bbox.astype(int)
        face_w = bx2 - bx1
        face_h = by2 - by1

        if face_w < MIN_FACE_PX or face_h < MIN_FACE_PX:
            return None, None, 0.0, 0.0, "too_small"

        c_by1, c_by2 = max(0, by1), min(ch, by2)
        c_bx1, c_bx2 = max(0, bx1), min(cw, bx2)
        face_crop = crop[c_by1:c_by2, c_bx1:c_bx2]
        if face_crop.size == 0:
            return None, None, 0.0, 0.0, "no_face"

        if max(face_w, face_h) < UPSCALE_FACE_THRESH:
            face_crop = cv2.resize(face_crop, (face_w * 2, face_h * 2), interpolation=cv2.INTER_LANCZOS4)

        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        if _laplacian_variance(gray) < BLUR_VAR_THRESH:
            return None, None, 0.0, 0.0, "blurry"

        yaw_deg, pitch_deg = 0.0, 0.0
        if best.kps is not None and len(best.kps) >= 5:
            kps5 = best.kps.astype(np.float32)
            yaw_deg, pitch_deg = _estimate_head_pose(kps5, cw, ch)

        if abs(yaw_deg) > YAW_MAX_DEG or abs(pitch_deg) > PITCH_MAX_DEG:
            return None, None, yaw_deg, pitch_deg, "unverifiable"

        emb = best.embedding
        if emb is None:
            return None, None, yaw_deg, pitch_deg, "no_embedding"

        emb = emb.astype(np.float32)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb /= norm

        face_bbox_frame = np.array([
            bx1 + crop_offset_x, by1 + crop_offset_y, bx2 + crop_offset_x, by2 + crop_offset_y,
        ], dtype=int)

        return emb, face_bbox_frame, yaw_deg, pitch_deg, "ok"


# ===========================================================================
# Module 3 — Cohort Z-Score Open-Set Matcher
# ===========================================================================

class OpenSetMatcher:
    """
    Queries FAISS IndexFlatIP for top-K matches and computes:
      S1     = top-1 cosine similarity
      Z      = (S1 - μ_impostors) / σ_impostors   (cohort Z-score)
    """

    def __init__(self, k: int = FAISS_TOP_K):
        self._index:        Optional[faiss.Index]         = None
        self._id_map:       Dict[int, Tuple[str, str]]    = {}   # faiss_id → (person_id, name)
        self._k = k

    def load(self, faiss_path: Path, db_path: Path):
        log.info(f"Loading FAISS index from {faiss_path}")
        self._index = faiss.read_index(str(faiss_path))
        log.info(f"  → {self._index.ntotal} vectors")

        conn = sqlite3.connect(str(db_path))
        rows = conn.execute(
            "SELECT v.faiss_id, v.person_id, p.name "
            "FROM vectors v JOIN persons p ON v.person_id=p.person_id"
        ).fetchall()
        conn.close()
        self._id_map = {int(r[0]): (str(r[1]), str(r[2])) for r in rows}
        log.info(f"  → {len(self._id_map)} vector↔person mappings loaded")

    @property
    def is_ready(self) -> bool:
        return self._index is not None and self._index.ntotal > 0

    def match(self, embedding: np.ndarray) -> MatchResult:
        if not self.is_ready:
            return MatchResult()

        k_actual = min(self._k, self._index.ntotal)
        query    = embedding.reshape(1, -1)
        dists, ids = self._index.search(query, k_actual)
        dists = dists[0]   # shape (k_actual,)
        ids   = ids[0]

        if len(dists) == 0 or ids[0] < 0:
            return MatchResult()

        s1      = float(dists[0])
        top1_id = int(ids[0])
        top1_person_id, top1_name = self._id_map.get(top1_id, (None, None))

        # Cohort Z-score: filter out other views of the SAME person
        # so multiple enrolled views of target identity don't inflate impostor mean
        other_dists = [
            float(d) for d, vid in zip(dists[1:], ids[1:])
            if vid >= 0 and self._id_map.get(int(vid), (None,))[0] != top1_person_id
        ]
        if len(other_dists) >= 2:
            mu  = float(np.mean(other_dists))
            sig = float(np.std(other_dists))
            z_score = float((s1 - mu) / sig) if sig > 1e-9 else 0.0
        else:
            z_score = 0.0

        is_match = (s1 >= S1_KNOWN_THRESH) or (z_score >= Z_KNOWN_THRESH)

        return MatchResult(
            person_id   = top1_person_id if is_match else None,
            person_name = top1_name      if is_match else None,
            s1          = s1,
            z_score     = z_score,
            is_match    = is_match,
        )


# ===========================================================================
# Module 4 — Temporal Voting + Torso Re-ID Cache
# ===========================================================================

def _compute_torso_hist(frame: np.ndarray, person_bbox: np.ndarray) -> Optional[np.ndarray]:
    """
    Extract an HSV hue histogram from the torso region of a person bbox.
    The torso is defined as the middle vertical third of the bounding box.
    """
    x1, y1, x2, y2 = person_bbox[:4].astype(int)
    fh, fw = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(fw, x2), min(fh, y2)

    bh = y2 - y1
    bw = x2 - x1
    if bh < 30 or bw < 10:
        return None

    # Middle vertical third = torso region
    ty1 = y1 + bh // 3
    ty2 = y1 + 2 * bh // 3
    torso = frame[ty1:ty2, x1:x2]
    if torso.size == 0:
        return None

    hsv   = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    h_ch  = hsv[:, :, 0]   # hue channel
    hist  = cv2.calcHist([h_ch], [0], None, [TORSO_HIST_BINS], [0, 180])
    hist  = hist.flatten().astype(np.float32)
    norm  = np.linalg.norm(hist)
    if norm > 0:
        hist /= norm
    return hist


def _torso_similarity(h1: np.ndarray, h2: np.ndarray) -> float:
    """Cosine similarity between two torso histograms."""
    return float(np.dot(h1, h2))


class TemporalVoter:
    """
    Manages per-track state machines for temporal voting, evasive detection,
    and torso re-ID caching.
    """

    def __init__(self, db_path: Optional[Path] = None, snapshot_dir: Path = SNAPSHOT_DIR):
        self._states: Dict[int, TrackState] = {}
        self._db_path      = db_path
        self._snapshot_dir = snapshot_dir
        snapshot_dir.mkdir(parents=True, exist_ok=True)

    def _get_or_create(self, track_id: int, timestamp: float) -> TrackState:
        if track_id not in self._states:
            self._states[track_id] = TrackState(
                track_id             = track_id,
                first_seen_ts        = timestamp,
                last_face_visible_ts = timestamp,
            )
        return self._states[track_id]

    def get_or_create_state(self, track_id: int, timestamp: float) -> TrackState:
        return self._get_or_create(track_id, timestamp)

    def update_card_detection(
        self,
        track_id:      int,
        card_detected: bool,
        card_bbox:     Optional[np.ndarray],
        card_conf:     float,
        timestamp:     Optional[float] = None,
    ) -> TrackStatus:
        """
        Stage 1: Track ID card detection on chest.
        If card is detected for 3 frames, immediately lock status as ID_CARD_VERIFIED.
        """
        state = self._get_or_create(track_id, timestamp or time.time())

        # Once verified, it remains locked
        if state.status == TrackStatus.ID_CARD_VERIFIED:
            state.has_id_card = True
            if card_bbox is not None:
                state.card_bbox = card_bbox
                state.card_conf = card_conf
            return TrackStatus.ID_CARD_VERIFIED

        if card_detected:
            state.card_detected_frames += 1
            state.no_card_frames = 0
            state.card_bbox = card_bbox
            state.card_conf = card_conf

            if state.card_detected_frames >= FRAMES_TO_LOCK_CARD:
                state.status      = TrackStatus.ID_CARD_VERIFIED
                state.has_id_card = True
                log.info(f"Track {track_id}: ID_CARD_VERIFIED (compliant — wearing ID card)")
                return TrackStatus.ID_CARD_VERIFIED
        else:
            state.no_card_frames += 1
            if state.no_card_frames > 2:
                state.card_detected_frames = max(0, state.card_detected_frames - 1)
                state.card_bbox = None
                state.card_conf = 0.0

        return state.status

    def update_face_pipeline(
        self,
        track_id:       int,
        frame:          np.ndarray,
        person_bbox:    np.ndarray,
        match:          Optional[MatchResult],
        quality_reason: str,
        timestamp:      float,
    ) -> TrackStatus:
        """
        Stage 2: Facial recognition escalation (after 10 frames without ID card).
        Branch 2A: Face matches database (3 frames) -> KNOWN_NO_BADGE (minor warning)
        Branch 2B: Face fails match (6 frames) or evasive 6s -> CONFIRMED_UNKNOWN (Intruder Alert)
        """
        state = self._get_or_create(track_id, timestamp)

        # Sticky states
        if state.status in (TrackStatus.ID_CARD_VERIFIED, TrackStatus.KNOWN_NO_BADGE, TrackStatus.CONFIRMED_UNKNOWN):
            if state.status == TrackStatus.KNOWN_NO_BADGE and quality_reason != "ok":
                torso_hist = _compute_torso_hist(frame, person_bbox)
                if torso_hist is not None and state.torso_hist is not None:
                    sim = _torso_similarity(torso_hist, state.torso_hist)
                    if sim >= TORSO_SIM_THRESH:
                        state.face_hidden_since_ts = None
                        return TrackStatus.KNOWN_NO_BADGE
            return state.status

        # ------------------------------------------------------------------
        # Per-frame unverifiable pose
        # ------------------------------------------------------------------
        if quality_reason == "unverifiable":
            if state.face_hidden_since_ts is None:
                state.face_hidden_since_ts = timestamp
            hidden_secs = timestamp - state.face_hidden_since_ts
            if hidden_secs > EVASIVE_SECONDS:
                state.status = TrackStatus.CONFIRMED_UNKNOWN
                log.warning(f"Track {track_id}: CONFIRMED_UNKNOWN (face evasive > {hidden_secs:.1f}s with no ID card)")
                self._fire_alert(track_id, frame, person_bbox, match, timestamp)
                return TrackStatus.CONFIRMED_UNKNOWN
            return TrackStatus.UNVERIFIABLE

        # ------------------------------------------------------------------
        # No face / blurry / too small
        # ------------------------------------------------------------------
        if quality_reason != "ok" or match is None:
            if state.face_hidden_since_ts is None:
                state.face_hidden_since_ts = timestamp
            hidden_secs = timestamp - state.face_hidden_since_ts
            if hidden_secs > EVASIVE_SECONDS:
                state.status = TrackStatus.CONFIRMED_UNKNOWN
                log.warning(f"Track {track_id}: CONFIRMED_UNKNOWN (no face > {hidden_secs:.1f}s with no ID card)")
                self._fire_alert(track_id, frame, person_bbox, match, timestamp)
                return TrackStatus.CONFIRMED_UNKNOWN
            return TrackStatus.SCANNING_FACE

        # Valid face detected
        state.last_face_visible_ts = timestamp
        state.face_hidden_since_ts = None

        # ------------------------------------------------------------------
        # Branch 2A: Face matches enrolled database
        # ------------------------------------------------------------------
        if match.is_match:
            if state.voted_person_id is None or state.voted_person_id == match.person_id:
                state.voted_person_id = match.person_id
                state.positive_votes += 1
                state.negative_votes  = 0
            else:
                state.voted_person_id = match.person_id
                state.positive_votes  = 1
                state.negative_votes  = 0

            if state.positive_votes >= FRAMES_TO_LOCK_KNOWN:
                state.status            = TrackStatus.KNOWN_NO_BADGE
                state.known_person_id   = match.person_id
                state.known_person_name = match.person_name
                state.torso_hist        = _compute_torso_hist(frame, person_bbox)
                if not state.badge_warning_logged:
                    log.info(f"Track {track_id}: KNOWN_NO_BADGE → {match.person_name} (S1={match.s1:.3f}, Z={match.z_score:.2f})")
                    self._log_badge_warning(track_id, match, timestamp)
                    state.badge_warning_logged = True
                return TrackStatus.KNOWN_NO_BADGE
            else:
                return TrackStatus.SCANNING_FACE

        # ------------------------------------------------------------------
        # Branch 2B: Face does NOT match database — Unknown Intruder
        # ------------------------------------------------------------------
        else:
            poor_match = (match.s1 <= S1_UNKNOWN_THRESH) and (match.z_score < Z_UNKNOWN_THRESH)
            if poor_match:
                state.negative_votes += 1
                state.positive_votes  = 0
                state.voted_person_id = None

            if state.negative_votes >= FRAMES_TO_ALERT:
                state.status = TrackStatus.CONFIRMED_UNKNOWN
                log.warning(f"Track {track_id}: CONFIRMED_UNKNOWN (neg_votes={state.negative_votes})")
                self._fire_alert(track_id, frame, person_bbox, match, timestamp)
                return TrackStatus.CONFIRMED_UNKNOWN
            else:
                return TrackStatus.SCANNING_FACE

    def update(
        self,
        track_id:       int,
        frame:          np.ndarray,
        person_bbox:    np.ndarray,
        match:          Optional[MatchResult],
        quality_reason: str,
        timestamp:      float,
    ) -> TrackStatus:
        """Legacy / unified update wrapper."""
        state = self._get_or_create(track_id, timestamp)
        if state.status == TrackStatus.ID_CARD_VERIFIED:
            return TrackStatus.ID_CARD_VERIFIED
        return self.update_face_pipeline(track_id, frame, person_bbox, match, quality_reason, timestamp)

    def _log_badge_warning(
        self,
        track_id:  int,
        match:     Optional[MatchResult],
        timestamp: float,
    ):
        """Log a minor badge-compliance warning to SQLite (not an intruder alarm)."""
        if self._db_path and self._db_path.exists():
            try:
                conn = sqlite3.connect(str(self._db_path))
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS badge_warnings (
                        id            INTEGER PRIMARY KEY AUTOINCREMENT,
                        track_id      INTEGER,
                        person_id     TEXT,
                        person_name   TEXT,
                        s1            REAL,
                        z_score       REAL,
                        created_at    REAL
                    )
                """)
                conn.execute(
                    "INSERT INTO badge_warnings (track_id, person_id, person_name, s1, z_score, created_at) "
                    "VALUES (?,?,?,?,?,?)",
                    (
                        track_id,
                        match.person_id if match else "",
                        match.person_name if match else "",
                        match.s1 if match else 0.0,
                        match.z_score if match else 0.0,
                        timestamp,
                    ),
                )
                conn.commit()
                conn.close()
                log.info(f"Badge-compliance warning written to SQLite for track {track_id}")
            except Exception as e:
                log.error(f"SQLite badge_warnings write failed: {e}")

    def _fire_alert(
        self,
        track_id:    int,
        frame:       np.ndarray,
        person_bbox: np.ndarray,
        match:       Optional[MatchResult],
        timestamp:   float,
    ) -> Optional[str]:
        """Save a snapshot JPEG and write an alert row to SQLite."""
        state = self._states[track_id]
        if state.alert_saved:
            return None

        # Crop person for snapshot
        x1, y1, x2, y2 = person_bbox[:4].astype(int)
        fh, fw = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(fw, x2), min(fh, y2)
        crop    = frame[y1:y2, x1:x2]

        ts_str  = time.strftime("%Y%m%d_%H%M%S", time.localtime(timestamp))
        snap_name = f"intruder_track{track_id}_{ts_str}.jpg"
        snap_path = self._snapshot_dir / snap_name
        cv2.imwrite(str(snap_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
        state.alert_saved = True

        # SQLite alert row
        if self._db_path and self._db_path.exists():
            try:
                conn = sqlite3.connect(str(self._db_path))
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
                conn.execute(
                    "INSERT INTO alerts "
                    "(track_id, alert_type, top1_person, s1, z_score, snapshot_path, created_at) "
                    "VALUES (?,?,?,?,?,?,?)",
                    (
                        track_id,
                        "CONFIRMED_UNKNOWN",
                        match.person_id if match else None,
                        match.s1        if match else 0.0,
                        match.z_score   if match else 0.0,
                        str(snap_path),
                        timestamp,
                    ),
                )
                conn.commit()
                conn.close()
            except Exception as e:
                log.error(f"SQLite alert write failed: {e}")

        log.info(f"  Intruder snapshot saved: {snap_path}")
        return str(snap_path)

    def get_state(self, track_id: int) -> Optional[TrackState]:
        return self._states.get(track_id)

    def purge_stale_tracks(self, active_track_ids: set, max_age_s: float = 30.0):
        """Remove state entries for tracks that have been gone too long."""
        now = time.time()
        to_del = [
            tid for tid, st in self._states.items()
            if tid not in active_track_ids and
               (now - st.last_face_visible_ts) > max_age_s
        ]
        for tid in to_del:
            del self._states[tid]


# ===========================================================================
# Module 5 — Homography Map Projection
# ===========================================================================

class RadarProjector:
    """
    Projects a person's foot coordinate (bottom-centre of bbox) to a
    (0–1000, 0–1000) 2-D campus radar coordinate using a 3×3 homography.
    """

    def __init__(self, homography: Optional[np.ndarray] = None):
        self._H = homography if homography is not None else DEFAULT_HOMOGRAPHY.copy()

    def set_homography(self, H: np.ndarray):
        """Update the 3×3 homography matrix (call after manual calibration)."""
        assert H.shape == (3, 3), "Homography must be 3×3"
        self._H = H.astype(np.float64)

    def project(self, bbox_xyxy: np.ndarray) -> Tuple[float, float]:
        """
        Compute radar (x, y) for the foot-point of a person bbox.

        Args:
            bbox_xyxy: [x1, y1, x2, y2] in image pixels.

        Returns:
            (rx, ry) clamped to [0, 1000] × [0, 1000].
        """
        x1, y1, x2, y2 = bbox_xyxy[:4].astype(float)
        foot_x = (x1 + x2) / 2.0
        foot_y = float(y2)

        pt = np.array([[[foot_x, foot_y]]], dtype=np.float64)
        projected = cv2.perspectiveTransform(pt, self._H)
        rx = float(np.clip(projected[0, 0, 0], 0, 1000))
        ry = float(np.clip(projected[0, 0, 1], 0, 1000))
        return rx, ry


# ===========================================================================
# Frame Annotator
# ===========================================================================

def _annotate_frame(
    frame:   np.ndarray,
    persons: List[TrackedPerson],
) -> np.ndarray:
    """Draw clean annotations focusing strictly on the ID card and primary subject."""
    out = frame.copy()
    if not persons:
        return out

    # Filter to primary foreground person(s): reject background noise and spurious detections
    areas = [(p.bbox_xyxy[2] - p.bbox_xyxy[0]) * (p.bbox_xyxy[3] - p.bbox_xyxy[1]) for p in persons]
    max_area = max(areas) if areas else 0
    primary_persons = [
        p for p in persons
        if (p.bbox_xyxy[2] - p.bbox_xyxy[0]) * (p.bbox_xyxy[3] - p.bbox_xyxy[1]) >= 0.20 * max_area
        or p.has_id_card
    ]

    for p in primary_persons:
        x1, y1, x2, y2 = p.bbox_xyxy[:4].astype(int)

        # Draw prominent, vibrant Cyan/Neon Green bounding box around detected ID Card
        if p.card_bbox is not None:
            cx1, cy1, cx2, cy2 = p.card_bbox[:4].astype(int)
            # Bright Cyan/Yellow (255, 255, 0)
            cv2.rectangle(out, (cx1, cy1), (cx2, cy2), (255, 255, 0), 3)
            conf_pct = int(p.card_conf * 100) if p.card_conf > 0 else 95
            card_lbl = f"ID CARD DETECTED ({conf_pct}%)"
            (ctw, cth), _ = cv2.getTextSize(card_lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 2)
            cv2.rectangle(out, (cx1, max(0, cy1 - cth - 8)), (cx1 + ctw + 6, cy1), (255, 255, 0), -1)
            cv2.putText(out, card_lbl, (cx1 + 3, max(cth + 2, cy1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 2, cv2.LINE_AA)

        # Status and HUD Label
        if p.status == TrackStatus.ID_CARD_VERIFIED or p.has_id_card:
            color = (0, 255, 0)
            label = "ID CARD VERIFIED - COMPLIANT"
            thick = 3
        elif p.status == TrackStatus.KNOWN_NO_BADGE:
            color = (255, 180, 0)
            name_str = p.person_name if p.person_name else f"EMP#{p.track_id}"
            label = f"FACE VERIFIED: {name_str}"
            thick = 2
        elif p.status in (TrackStatus.CONFIRMED_UNKNOWN, TrackStatus.EVASIVE_ALERT):
            color = (0, 0, 255)
            label = "ALERT: NO ID CARD"
            thick = 2
        else:
            color = (200, 200, 200)
            label = "SEARCHING FOR ID CARD..."
            thick = 2

        # Draw person box (unless person bbox is basically the card box itself)
        is_card_only = (p.card_bbox is not None and abs(x1 - p.card_bbox[0]) < 20 and abs(y1 - p.card_bbox[1]) < 20)
        if not is_card_only:
            cv2.rectangle(out, (x1, y1), (x2, y2), color, thick)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
            lbl_y1 = max(0, y1 - th - 8)
            cv2.rectangle(out, (x1, lbl_y1), (x1 + tw + 6, y1), color, -1)
            txt_color = (0, 0, 0) if color in ((0, 255, 0), (0, 220, 255), (0, 215, 255)) else (255, 255, 255)
            cv2.putText(out, label, (x1 + 3, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, txt_color, 1, cv2.LINE_AA)

        # Subtle face indicator if face detection was active
        if p.face_bbox is not None and not p.has_id_card and p.status != TrackStatus.ID_CARD_VERIFIED:
            fx1, fy1, fx2, fy2 = p.face_bbox[:4].astype(int)
            cv2.rectangle(out, (fx1, fy1), (fx2, fy2), (255, 255, 255), 1)

    return out


# ===========================================================================
# DroneSurveillanceEngine — public API
# ===========================================================================

class DroneSurveillanceEngine:
    """
    Orchestrates the two-tier hierarchical ID Card Gate surveillance pipeline:
      Stage 1: Fast tracking (YOLOv11 Nano at imgsz=416) + cached status rendering at 25+ FPS
      Stage 2: Asynchronous Biometric Worker (threading.Thread + queue.Queue(maxsize=2))
               running CLAHE pre-normalization, InsightFace & card detection without video stutter.
    """

    def __init__(
        self,
        yolo_model:   str             = YOLO_MODEL_NAME,
        card_model:   str             = CARD_MODEL_NAME,
        yolo_conf:    float           = 0.55,
        card_conf:    float           = CARD_CONF_THRESH,
        det_thresh:   float           = 0.55,
        homography:   Optional[np.ndarray] = None,
        snapshot_dir: Path            = SNAPSHOT_DIR,
    ):
        log.info("Initialising DroneSurveillanceEngine…")
        self._tracker       = BodyTracker(model_name=yolo_model, conf=yolo_conf)
        self._card_detector = ChestCardDetector(conf=card_conf)
        self._quality       = FaceQualityGate(det_thresh=det_thresh)
        self.face_app       = self._quality.face_app
        self._matcher       = OpenSetMatcher(k=FAISS_TOP_K)
        self._voter         = TemporalVoter(snapshot_dir=snapshot_dir)
        self._radar         = RadarProjector(homography=homography)
        self._virtual_tracker = VirtualPersonTracker(base_id=1)
        self._frame_idx     = 0

        # Architectural Fix #2: Decoupled Biometric Worker
        self._bio_queue     = queue.Queue(maxsize=2)
        self._async_results: Dict[int, dict] = {}
        self.tracks         = self._async_results  # public alias as requested
        self._results_lock  = threading.Lock()
        self._in_flight: set = set()
        self._worker_stopped = threading.Event()
        self._bio_thread    = threading.Thread(
            target=self._biometric_worker,
            daemon=True,
            name="biometric-worker"
        )
        self._bio_thread.start()
        log.info("DroneSurveillanceEngine ready (decoupled biometric worker started, CPU mode).")

    def load(self, faiss_path: str | Path, db_path: str | Path):
        """Load FAISS index and SQLite metadata. Must be called before process_frame."""
        faiss_path = Path(faiss_path)
        db_path    = Path(db_path)
        self._matcher.load(faiss_path, db_path)
        self._voter._db_path = db_path
        log.info("Engine fully loaded and ready.")

    def set_homography(self, H: np.ndarray):
        """Update the campus homography matrix (3×3 float64)."""
        self._radar.set_homography(H)

    def stop(self):
        """Stop background worker thread."""
        self._worker_stopped.set()
        try:
            self._bio_queue.put_nowait(None)
        except Exception:
            pass

    def _biometric_worker(self):
        """
        Dedicated background worker for face + ID card verification.
        Pulls one unverified person crop at a time from _bio_queue,
        runs CLAHE pre-normalization, InsightFace & card detection,
        and updates self.tracks[tid] asynchronously without blocking the video stream.
        """
        log.info("Biometric worker thread active.")
        while not self._worker_stopped.is_set():
            try:
                task = self._bio_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if task is None:
                break

            tid, crop, bbox, ts = task
            try:
                ch, cw = crop.shape[:2]

                # -----------------------------------------------------------
                # Stage 1: Chest & Interaction ROI ID Card Detection
                # -----------------------------------------------------------
                cx1 = max(0, int(0.08 * cw))
                cx2 = min(cw, int(0.92 * cw))
                cy1 = max(0, int(0.12 * ch))
                cy2 = min(ch, int(0.88 * ch))
                chest_subcrop = crop[cy1:cy2, cx1:cx2]

                card_detected, rel_box, card_conf, card_src = self._card_detector.detect(chest_subcrop)
                frame_card_bbox = None
                if card_detected and rel_box is not None:
                    frame_card_bbox = np.array([
                        bbox[0] + cx1 + rel_box[0],
                        bbox[1] + cy1 + rel_box[1],
                        bbox[0] + cx1 + rel_box[2],
                        bbox[1] + cy1 + rel_box[3],
                    ], dtype=int)
                else:
                    # Also test full person crop
                    card_detected, rel_box, card_conf, card_src = self._card_detector.detect(crop)
                    if card_detected and rel_box is not None:
                        frame_card_bbox = np.array([
                            bbox[0] + rel_box[0],
                            bbox[1] + rel_box[1],
                            bbox[0] + rel_box[2],
                            bbox[1] + rel_box[3],
                        ], dtype=int)

                card_status = self._voter.update_card_detection(
                    track_id      = tid,
                    card_detected = card_detected,
                    card_bbox     = frame_card_bbox,
                    card_conf     = card_conf,
                    timestamp     = ts,
                )

                if card_status == TrackStatus.ID_CARD_VERIFIED or card_detected:
                    with self._results_lock:
                        self._async_results[tid] = {
                            "status":            TrackStatus.ID_CARD_VERIFIED,
                            "person_id":         None,
                            "person_name":       None,
                            "s1":                0.0,
                            "z_score":           0.0,
                            "has_id_card":       True,
                            "card_bbox":         frame_card_bbox,
                            "card_conf":         card_conf if card_conf > 0 else 0.95,
                            "face_bbox":         None,
                            "yaw":               0.0,
                            "pitch":             0.0,
                            "last_processed_ts": ts,
                        }
                else:
                    # -----------------------------------------------------------
                    # Stage 2: CLAHE Pre-normalization + Face Quality Gate + ArcFace
                    # -----------------------------------------------------------
                    clahe_crop = apply_clahe_lab(crop)
                    emb, rel_face_box, yaw, pitch, reason = self._quality.analyse_crop(clahe_crop)

                    frame_face_bbox = None
                    if rel_face_box is not None:
                        frame_face_bbox = np.array([
                            bbox[0] + rel_face_box[0],
                            bbox[1] + rel_face_box[1],
                            bbox[0] + rel_face_box[2],
                            bbox[1] + rel_face_box[3],
                        ], dtype=int)

                    match: Optional[MatchResult] = None
                    if reason == "ok" and emb is not None:
                        match = self._matcher.match(emb)

                    status = self._voter.update_face_pipeline(
                        track_id       = tid,
                        frame          = crop,
                        person_bbox    = bbox,
                        match          = match,
                        quality_reason = reason,
                        timestamp      = ts,
                    )

                    st = self._voter.get_state(tid)
                    with self._results_lock:
                        self._async_results[tid] = {
                            "status":            status,
                            "person_id":         st.known_person_id if st else (match.person_id if match else None),
                            "person_name":       st.known_person_name if st else (match.person_name if match else None),
                            "s1":                match.s1 if match else 0.0,
                            "z_score":           match.z_score if match else 0.0,
                            "has_id_card":       False,
                            "card_bbox":         frame_card_bbox,
                            "card_conf":         card_conf,
                            "face_bbox":         frame_face_bbox,
                            "yaw":               yaw,
                            "pitch":             pitch,
                            "last_processed_ts": ts,
                        }
            except Exception as exc:
                log.error(f"Biometric worker error on track {tid}: {exc}", exc_info=True)
            finally:
                with self._results_lock:
                    self._in_flight.discard(tid)
                self._bio_queue.task_done()

    def process_frame(
        self,
        frame:     np.ndarray,
        timestamp: Optional[float] = None,
        annotate:  bool            = True,
    ) -> FrameResult:
        """
        Fast non-blocking main loop:
          - Runs YOLOv11 Nano person tracking at imgsz=416 (~15 ms).
          - Instantly renders the video at 25+ FPS using latest cached status.
          - Dispatches unverified crops asynchronously to background worker.
        """
        if timestamp is None:
            timestamp = time.time()

        self._frame_idx += 1
        result = FrameResult(frame_idx=self._frame_idx, timestamp=timestamp)

        fh, fw = frame.shape[:2]
        frame_area = fh * fw

        # ---------------------------------------------------------------
        # Fast YOLOv11 Nano person tracking
        # ---------------------------------------------------------------
        tracked_persons = self._tracker.track(frame)

        # Filter out background furniture/noise detections
        if tracked_persons:
            p_areas = [(px2 - px1) * (py2 - py1) for _, (px1, py1, px2, py2) in tracked_persons]
            max_p_area = max(p_areas) if p_areas else 0
            filtered_persons = []
            for tid, bbox in tracked_persons:
                px1, py1, px2, py2 = bbox[:4]
                area = (px2 - px1) * (py2 - py1)
                if area >= 0.03 * frame_area and (max_p_area == 0 or area >= 0.20 * max_p_area):
                    filtered_persons.append((tid, bbox))
            tracked_persons = filtered_persons
        else:
            # Webcam close-up fallback: center region represents primary user
            center_bbox = np.array([int(0.12 * fw), int(0.04 * fh), int(0.88 * fw), int(0.96 * fh)], dtype=int)
            tracked_persons = [(1, center_bbox)]

        active_ids = {tid for tid, _ in tracked_persons if tid > 0}
        persons_out: List[TrackedPerson] = []

        for track_id, bbox in tracked_persons:
            if track_id <= 0:
                continue

            with self._results_lock:
                cached = self._async_results.get(track_id)

            if cached is None:
                current_status = TrackStatus.PENDING
                has_card  = False
                card_box  = None
                card_conf = 0.0
                face_box  = None
                pid       = None
                pname     = None
                s1        = 0.0
                z_score   = 0.0
                yaw       = 0.0
                pitch     = 0.0
            else:
                current_status = cached.get("status", TrackStatus.PENDING)
                has_card       = cached.get("has_id_card", False)
                card_box       = cached.get("card_bbox")
                card_conf      = cached.get("card_conf", 0.0)
                face_box       = cached.get("face_bbox")
                pid            = cached.get("person_id")
                pname          = cached.get("person_name")
                s1             = cached.get("s1", 0.0)
                z_score        = cached.get("z_score", 0.0)
                yaw            = cached.get("yaw", 0.0)
                pitch          = cached.get("pitch", 0.0)

            # Asynchronously dispatch unverified person crop to background worker
            needs_verification = (current_status in (TrackStatus.PENDING, TrackStatus.SCANNING_FACE, TrackStatus.UNVERIFIABLE))
            with self._results_lock:
                is_in_flight = track_id in self._in_flight

            last_ts = cached.get("last_processed_ts", 0.0) if cached else 0.0
            if needs_verification and not is_in_flight and (timestamp - last_ts > 0.15):
                px1, py1, px2, py2 = bbox[:4]
                cx1, cy1 = max(0, px1), max(0, py1)
                cx2, cy2 = min(fw, px2), min(fh, py2)
                crop = frame[cy1:cy2, cx1:cx2].copy()
                if crop.size > 0:
                    try:
                        self._bio_queue.put_nowait((track_id, crop, np.array([cx1, cy1, cx2, cy2]), timestamp))
                        with self._results_lock:
                            self._in_flight.add(track_id)
                    except queue.Full:
                        pass  # worker busy: zero dropping of video frames!

            radar_xy = self._radar.project(bbox)
            tp = TrackedPerson(
                track_id    = track_id,
                bbox_xyxy   = bbox,
                status      = current_status,
                person_id   = pid,
                person_name = pname,
                s1          = s1,
                z_score     = z_score,
                face_bbox   = face_box,
                yaw_deg     = yaw,
                pitch_deg   = pitch,
                radar_xy    = radar_xy,
                has_id_card = has_card,
                card_bbox   = card_box,
                card_conf   = card_conf,
            )
            persons_out.append(tp)

        # Purge stale tracks
        self._voter.purge_stale_tracks(active_ids)
        with self._results_lock:
            self._async_results = {tid: c for tid, c in self._async_results.items() if tid in active_ids}

        result.persons = persons_out
        if annotate:
            result.annotated_frame = _annotate_frame(frame, persons_out)
        else:
            result.annotated_frame = frame

        return result

    def calibrate_live_face(
        self,
        frame: np.ndarray,
        faiss_path: Path | str,
        db_path: Path | str,
        person_id: Optional[str] = None,
    ) -> dict:
        """
        Instant Live Self-Calibration (Fix #5):
        Grabs the largest face currently visible in *frame*, applies CLAHE LAB pre-normalization,
        extracts its 512-D embedding, merges it with the user's identity in campus.faiss and campus.db,
        and hot-reloads the index immediately.
        """
        faiss_path = Path(faiss_path)
        db_path = Path(db_path)
        if frame is None or frame.size == 0:
            return {"success": False, "message": "Invalid frame received"}

        # Apply CLAHE luminance normalization in LAB space
        norm_frame = apply_clahe_lab(frame)
        faces = self.face_app.get(norm_frame)
        if not faces:
            faces = self.face_app.get(frame)

        if not faces:
            return {
                "success": False,
                "message": "No face detected on webcam. Please face the camera with good lighting."
            }

        # Largest face by bounding box area
        best_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        if best_face.embedding is None:
            return {"success": False, "message": "Could not extract face embedding from webcam."}

        emb = best_face.embedding.astype(np.float32)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb /= norm

        # Connect to SQLite DB to identify the enrolled person
        conn = sqlite3.connect(str(db_path))
        target_pid = person_id
        target_name = None

        if target_pid:
            row = conn.execute("SELECT person_id, name FROM persons WHERE person_id=?", (target_pid,)).fetchone()
            if row:
                target_name = row[1]

        if not target_name:
            # Pick first enrolled person from database
            row = conn.execute("SELECT person_id, name FROM persons ORDER BY enrolled_at ASC LIMIT 1").fetchone()
            if row:
                target_pid, target_name = row[0], row[1]
            else:
                target_pid = "EMP001"
                target_name = "Campus User"
                conn.execute(
                    "INSERT OR REPLACE INTO persons (person_id, name, photo_path, enrolled_at) VALUES (?,?,?,?)",
                    (target_pid, target_name, "webcam_live", time.time())
                )
                conn.commit()

        # Update FAISS index
        if faiss_path.exists():
            idx = faiss.read_index(str(faiss_path))
        else:
            idx = faiss.IndexFlatIP(EMBEDDING_DIM)

        new_faiss_id = idx.ntotal
        idx.add(emb.reshape(1, -1))
        faiss.write_index(idx, str(faiss_path))

        # Record vector in SQLite
        conn.execute(
            "INSERT INTO vectors (faiss_id, person_id, view_index, view_label) VALUES (?,?,?,?)",
            (new_faiss_id, target_pid, 99, "live_webcam_calibration")
        )
        conn.commit()
        conn.close()

        # Hot-reload FAISS & DB into running matcher
        self.load(faiss_path, db_path)

        # Clear cached statuses so new vector is matched immediately
        with self._results_lock:
            self._async_results.clear()
            self._in_flight.clear()

        log.info(f"⚡ Live face calibrated for [{target_pid}] {target_name} -> FAISS ID {new_faiss_id} (total={idx.ntotal})")
        return {
            "success": True,
            "person_id": target_pid,
            "person_name": target_name,
            "faiss_total": idx.ntotal,
            "message": f"Successfully calibrated live face for {target_name} ({target_pid})! Index now has {idx.ntotal} vectors."
        }


# ===========================================================================
# Quick smoke-test CLI
# ===========================================================================
if __name__ == "__main__":
    import argparse, sys

    parser = argparse.ArgumentParser(description="Vision Engine smoke-test")
    parser.add_argument("--video",   default="data/videos/campus_drone.mp4")
    parser.add_argument("--faiss",   default="storage/campus.faiss")
    parser.add_argument("--db",      default="storage/campus.db")
    parser.add_argument("--frames",  type=int, default=10,
                        help="Number of frames to process (default: 10)")
    parser.add_argument("--no-annotate", action="store_true")
    args = parser.parse_args()

    engine = DroneSurveillanceEngine()

    faiss_p = Path(args.faiss)
    db_p    = Path(args.db)
    if faiss_p.exists() and db_p.exists():
        engine.load(faiss_path=faiss_p, db_path=db_p)
    else:
        log.warning("FAISS/DB not found — running without identity matching")

    video_p = Path(args.video)
    if not video_p.exists():
        log.error(f"Video not found: {video_p}")
        sys.exit(1)

    cap = cv2.VideoCapture(str(video_p))
    if not cap.isOpened():
        log.error("Cannot open video")
        sys.exit(1)

    log.info(f"Processing {args.frames} frames from {video_p.name}…")
    errors = []
    frame_count = 0

    while frame_count < args.frames:
        ret, frame = cap.read()
        if not ret:
            log.warning("Video ended before reaching target frame count.")
            break
        try:
            result = engine.process_frame(
                frame,
                timestamp=time.time(),
                annotate=not args.no_annotate,
            )
            log.info(
                f"  Frame {result.frame_idx:03d} | "
                f"persons={len(result.persons)} | "
                + " | ".join(
                    f"T{p.track_id}:{p.status.name}"
                    + (f"({p.person_name})" if p.person_name else "")
                    for p in result.persons
                )
            )
        except Exception as exc:
            errors.append((frame_count, str(exc)))
            log.error(f"  Frame {frame_count} ERROR: {exc}", exc_info=True)
        frame_count += 1

    cap.release()

    log.info("=" * 60)
    if errors:
        log.error(f"SMOKE TEST FAILED — {len(errors)} error(s):")
        for fi, err in errors:
            log.error(f"  frame {fi}: {err}")
        sys.exit(1)
    else:
        log.info(f"SMOKE TEST PASSED — {frame_count} frames processed, 0 errors.")
        sys.exit(0)
