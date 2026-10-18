"""
enroll_ids.py
=============
Phase 1 — ID Card Enrollment Pipeline

Reads ID card photos from data/id_cards/ (filename convention: EMP001_Name.jpg),
applies CLAHE contrast normalisation, generates 6 synthetic perspective-warped
views per photo (simulating drone downward/yaw angles), extracts 512-D ArcFace
embeddings via InsightFace, stores vectors in a FAISS IndexFlatIP index
(storage/campus.faiss), and records metadata + vector mappings in SQLite
(storage/campus.db).

Usage:
    python enroll_ids.py [--id_cards_dir data/id_cards]
                         [--storage_dir storage]
                         [--det_thresh 0.3]
                         [--verbose]
"""

# ---------------------------------------------------------------------------
# CPU setup
# ---------------------------------------------------------------------------
import argparse
import logging
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Auto-dispatch to project .venv if invoked with system python
# ---------------------------------------------------------------------------
_venv_python = Path(__file__).resolve().parent / ".venv" / "Scripts" / "python.exe"
if _venv_python.exists() and Path(sys.executable).resolve() != _venv_python.resolve():
    sys.exit(subprocess.call([str(_venv_python)] + sys.argv))

os.environ["OMP_NUM_THREADS"]       = "4"
os.environ["OPENBLAS_NUM_THREADS"]  = "4"

import cv2
import faiss
import numpy as np

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("enroll_ids")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EMBEDDING_DIM = 512          # ArcFace output dimensionality
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Synthetic view parameters: (pitch_deg, yaw_deg)
# pitch > 0  = downward tilt (drone looking down)
# yaw   != 0 = left/right rotation
SYNTHETIC_VIEWS: list[tuple[float, float]] = [
    (20.0,   0.0),   # mild downward
    (35.0,   0.0),   # moderate downward
    (45.0,   0.0),   # steep downward
    (20.0,  20.0),   # mild down + right yaw
    (20.0, -20.0),   # mild down + left yaw
    (35.0,  20.0),   # moderate down + right yaw
    # Note: original (0°,0°) is always included as view 0, total = 7
]


# ---------------------------------------------------------------------------
# CLAHE contrast normalisation
# ---------------------------------------------------------------------------
def apply_clahe(bgr: np.ndarray) -> np.ndarray:
    """Apply CLAHE to the L channel of LAB colour space."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l_ch)
    merged = cv2.merge([l_eq, a_ch, b_ch])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


# ---------------------------------------------------------------------------
# Perspective warp — simulate drone viewing angle
# ---------------------------------------------------------------------------
def perspective_warp(
    img: np.ndarray,
    pitch_deg: float,
    yaw_deg: float,
) -> np.ndarray:
    """
    Generate a synthetic perspective-warped view of *img* that approximates
    a drone looking at a person's face from above (pitch) and to one side (yaw).

    The transform is computed analytically from the camera projection model:
    a rotation matrix R = Ry(yaw) · Rx(pitch) is used to derive where the
    four image corners map to in the virtual camera plane, then cv2.getPerspectiveTransform
    is applied to warp the source image.

    Args:
        img:       Input BGR image (any size).
        pitch_deg: Downward tilt angle in degrees (0 = frontal, 90 = top-down).
        yaw_deg:   Left/right rotation angle in degrees (positive = right).

    Returns:
        Warped BGR image with the same spatial dimensions as *img*.
    """
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    # Focal length heuristic: use image diagonal / 2
    f = np.sqrt(w ** 2 + h ** 2) / 2.0

    # Convert angles to radians
    pitch = np.radians(pitch_deg)
    yaw   = np.radians(yaw_deg)

    # Rotation matrices
    Rx = np.array([
        [1,           0,            0],
        [0,  np.cos(pitch), -np.sin(pitch)],
        [0,  np.sin(pitch),  np.cos(pitch)],
    ])
    Ry = np.array([
        [ np.cos(yaw), 0, np.sin(yaw)],
        [           0, 1,           0],
        [-np.sin(yaw), 0, np.cos(yaw)],
    ])
    R = Ry @ Rx  # combined rotation

    # Camera intrinsics (principal point at image centre)
    K = np.array([
        [f, 0, cx],
        [0, f, cy],
        [0, 0,  1],
    ])

    # Source corners in image coords
    src_corners = np.float32([
        [0,     0    ],
        [w - 1, 0    ],
        [w - 1, h - 1],
        [0,     h - 1],
    ])

    # Lift src corners to normalised 3-D plane (Z=1)
    dst_pts = []
    for (px, py) in src_corners:
        # Unproject to camera ray
        p_cam = np.array([(px - cx) / f, (py - cy) / f, 1.0])
        p_rot = R @ p_cam          # rotate
        if p_rot[2] == 0:
            p_rot[2] = 1e-6
        # Re-project onto image plane
        p_img = K @ p_rot
        p_img /= p_img[2]
        dst_pts.append([p_img[0], p_img[1]])

    dst_corners = np.float32(dst_pts)

    # Clamp destination points so the transform is valid
    dst_corners[:, 0] = np.clip(dst_corners[:, 0], -w, 2 * w)
    dst_corners[:, 1] = np.clip(dst_corners[:, 1], -h, 2 * h)

    M = cv2.getPerspectiveTransform(src_corners, dst_corners)
    warped = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)
    return warped


# ---------------------------------------------------------------------------
# InsightFace model loader (lazy singleton)
# ---------------------------------------------------------------------------
_face_app = None


def get_face_app(det_thresh: float = 0.3) -> object:
    """Load and cache the InsightFace FaceAnalysis model."""
    global _face_app
    if _face_app is None:
        try:
            import insightface
            from insightface.app import FaceAnalysis
        except ImportError:
            log.error("insightface is not installed. Run: pip install insightface")
            sys.exit(1)

        log.info("Loading InsightFace buffalo_l model (CPU, 320×320)…")
        app = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"],
            allowed_modules=["detection", "recognition"],
        )
        app.prepare(ctx_id=-1, det_size=(320, 320), det_thresh=det_thresh)
        _face_app = app
        log.info("InsightFace model loaded.")
    return _face_app


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------
def extract_embedding(bgr: np.ndarray, det_thresh: float = 0.3) -> np.ndarray | None:
    """
    Detect the largest face in *bgr* and return its L2-normalised 512-D embedding.
    Returns None if no face is detected.
    """
    app = get_face_app(det_thresh)
    faces = app.get(bgr)
    if not faces:
        return None
    # Pick the face with the largest bounding-box area
    best = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    emb = best.embedding.astype(np.float32)
    # L2 normalise for cosine similarity via IndexFlatIP
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb /= norm
    return emb


# ---------------------------------------------------------------------------
# SQLite helpers
# ---------------------------------------------------------------------------
SCHEMA = """
CREATE TABLE IF NOT EXISTS persons (
    person_id   TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    photo_path  TEXT NOT NULL,
    enrolled_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS vectors (
    faiss_id    INTEGER PRIMARY KEY,
    person_id   TEXT NOT NULL,
    view_index  INTEGER NOT NULL,   -- 0=original, 1-6=synthetic views
    view_label  TEXT NOT NULL,
    FOREIGN KEY (person_id) REFERENCES persons(person_id)
);
"""


def init_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def person_exists(conn: sqlite3.Connection, person_id: str) -> bool:
    cur = conn.execute("SELECT 1 FROM persons WHERE person_id=?", (person_id,))
    return cur.fetchone() is not None


def insert_person(conn: sqlite3.Connection, person_id: str, name: str, photo_path: str):
    conn.execute(
        "INSERT OR REPLACE INTO persons (person_id, name, photo_path, enrolled_at) VALUES (?,?,?,?)",
        (person_id, name, photo_path, time.time()),
    )


def insert_vector_row(conn: sqlite3.Connection, faiss_id: int, person_id: str,
                      view_index: int, view_label: str):
    conn.execute(
        "INSERT INTO vectors (faiss_id, person_id, view_index, view_label) VALUES (?,?,?,?)",
        (faiss_id, person_id, view_index, view_label),
    )


# ---------------------------------------------------------------------------
# Main enrollment function
# ---------------------------------------------------------------------------
def enroll_all(
    id_cards_dir: Path,
    storage_dir: Path,
    det_thresh: float = 0.3,
    verbose: bool = False,
    rebuild: bool = False,
) -> dict:
    """
    Enrol all ID card photos found in *id_cards_dir*.

    Returns a summary dict with counts of enrolled persons, total vectors, and
    any photos that failed (no face detected).
    """
    if verbose:
        log.setLevel(logging.DEBUG)

    storage_dir.mkdir(parents=True, exist_ok=True)
    faiss_path = storage_dir / "campus.faiss"
    db_path    = storage_dir / "campus.db"

    if rebuild:
        log.info("Rebuilding index and database from scratch (--rebuild)")
        if faiss_path.exists():
            faiss_path.unlink()
        if db_path.exists():
            db_path.unlink()

    # ------------------------------------------------------------------
    # Load or create FAISS index
    # ------------------------------------------------------------------
    if faiss_path.exists():
        log.info(f"Loading existing FAISS index from {faiss_path}")
        index = faiss.read_index(str(faiss_path))
        log.info(f"  → {index.ntotal} vectors already in index")
    else:
        log.info("Creating new FAISS IndexFlatIP index")
        index = faiss.IndexFlatIP(EMBEDDING_DIM)

    # ------------------------------------------------------------------
    # Open / create SQLite database
    # ------------------------------------------------------------------
    conn = init_db(db_path)

    # ------------------------------------------------------------------
    # Discover ID card images
    # ------------------------------------------------------------------
    photos = sorted(
        p for p in id_cards_dir.iterdir()
        if p.suffix.lower() in SUPPORTED_EXTS
    )
    if not photos:
        log.warning(f"No ID card photos found in {id_cards_dir}")
        return {"enrolled": 0, "vectors_added": 0, "failed": []}

    log.info(f"Found {len(photos)} ID card photo(s) to process")

    enrolled_count  = 0
    vectors_added   = 0
    failed_photos   = []

    for photo_path in photos:
        # Check if this exact file was already enrolled
        row = conn.execute("SELECT person_id, name FROM persons WHERE photo_path=?", (str(photo_path),)).fetchone()
        if row is not None:
            log.info(f"  [SKIP] {photo_path.name} already enrolled as [{row[0]}] {row[1]}")
            continue

        stem = photo_path.stem  # e.g. "EMP001_John_Doe"
        parts = stem.split("_", 1)
        if len(parts) >= 2 and parts[0].isalnum() and len(parts[0]) <= 10:
            person_id   = parts[0].upper()          # e.g. "EMP001"
            person_name = parts[1].replace("_", " ")  # e.g. "John Doe"
        else:
            # Fallback for arbitrary file names (e.g. WhatsApp images)
            existing_count = conn.execute("SELECT COUNT(*) FROM persons").fetchone()[0]
            person_id   = f"EMP{existing_count + 1:03d}"
            person_name = stem.replace("_", " ").strip()
            log.info(f"Auto-assigned ID '{person_id}' and name '{person_name}' for {photo_path.name}")

        log.info(f"Processing {photo_path.name}  →  [{person_id}] {person_name}")

        if person_exists(conn, person_id):
            log.info(f"  [SKIP] {person_id} already enrolled — use --rebuild to re-enrol")
            continue

        # ---------------------------------------------------------------
        # Load and pre-process original image
        # ---------------------------------------------------------------
        bgr_orig = cv2.imread(str(photo_path))
        if bgr_orig is None:
            log.error(f"  [FAIL] Cannot read image: {photo_path}")
            failed_photos.append(str(photo_path))
            continue

        # Resize if very large (keep aspect ratio, max 1024 px on longest side)
        max_dim = 1024
        h, w = bgr_orig.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            bgr_orig = cv2.resize(bgr_orig, (int(w * scale), int(h * scale)),
                                  interpolation=cv2.INTER_AREA)

        bgr_clahe = apply_clahe(bgr_orig)

        # ---------------------------------------------------------------
        # Build all 7 views: original + 6 synthetic
        # ---------------------------------------------------------------
        views: list[tuple[np.ndarray, int, str]] = [
            (bgr_clahe, 0, "original_0deg_0deg")
        ]
        for idx, (pitch, yaw) in enumerate(SYNTHETIC_VIEWS, start=1):
            warped = perspective_warp(bgr_clahe, pitch, yaw)
            label  = f"synth_{int(pitch)}deg_pitch_{int(yaw):+d}deg_yaw"
            views.append((warped, idx, label))

        # ---------------------------------------------------------------
        # Extract embeddings for all views
        # ---------------------------------------------------------------
        person_embeddings: list[tuple[np.ndarray, int, str]] = []
        for view_img, view_idx, view_label in views:
            emb = extract_embedding(view_img, det_thresh=det_thresh)
            if emb is None:
                log.debug(f"  [WARN] No face in view {view_label} — skipping")
                continue
            person_embeddings.append((emb, view_idx, view_label))

        if not person_embeddings:
            log.error(f"  [FAIL] No face detected in ANY view of {photo_path.name}")
            failed_photos.append(str(photo_path))
            continue

        log.info(f"  Extracted {len(person_embeddings)}/7 embeddings")

        # ---------------------------------------------------------------
        # Add to FAISS and SQLite
        # ---------------------------------------------------------------
        insert_person(conn, person_id, person_name, str(photo_path))

        for emb, view_idx, view_label in person_embeddings:
            faiss_id = index.ntotal          # next available row ID
            vec_matrix = emb.reshape(1, -1)
            index.add(vec_matrix)
            insert_vector_row(conn, faiss_id, person_id, view_idx, view_label)
            vectors_added += 1

        conn.commit()
        enrolled_count += 1
        log.info(f"  ✓ Enrolled [{person_id}] {person_name} "
                 f"({len(person_embeddings)} vectors)")

    # ------------------------------------------------------------------
    # Persist FAISS index
    # ------------------------------------------------------------------
    faiss.write_index(index, str(faiss_path))
    log.info(f"\nFAISS index saved → {faiss_path}  ({index.ntotal} total vectors)")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    summary = {
        "enrolled":      enrolled_count,
        "vectors_added": vectors_added,
        "total_vectors": index.ntotal,
        "failed":        failed_photos,
        "faiss_path":    str(faiss_path),
        "db_path":       str(db_path),
    }

    log.info("=" * 60)
    log.info(f"Enrollment complete:")
    log.info(f"  Persons enrolled  : {enrolled_count}")
    log.info(f"  Vectors added     : {vectors_added}")
    log.info(f"  Total in index    : {index.ntotal}")
    if failed_photos:
        log.warning(f"  Failed photos     : {len(failed_photos)}")
        for fp in failed_photos:
            log.warning(f"    - {fp}")
    log.info("=" * 60)

    conn.close()
    return summary


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Enrol ID card photos into the FAISS + SQLite campus index."
    )
    parser.add_argument(
        "--id_cards_dir", default="data/id_cards",
        help="Directory containing ID card photos (default: data/id_cards)"
    )
    parser.add_argument(
        "--storage_dir", default="storage",
        help="Directory for FAISS index and SQLite DB (default: storage)"
    )
    parser.add_argument(
        "--det_thresh", type=float, default=0.3,
        help="InsightFace face detection threshold (default: 0.3)"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable DEBUG logging"
    )
    parser.add_argument(
        "--rebuild", action="store_true",
        help="Wipe existing database and FAISS index, and rebuild from scratch"
    )
    args = parser.parse_args()

    summary = enroll_all(
        id_cards_dir=Path(args.id_cards_dir),
        storage_dir=Path(args.storage_dir),
        det_thresh=args.det_thresh,
        verbose=args.verbose,
        rebuild=args.rebuild,
    )

    if summary["enrolled"] == 0 and not summary["failed"]:
        log.warning("No new persons were enrolled (all already in DB, or no photos found).")

    sys.exit(0 if not summary["failed"] else 1)


if __name__ == "__main__":
    main()
