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

def apply_clahe(bgr: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Apply CLAHE luminance normalization in LAB color space to equalize shadows
    and harsh lighting while preserving natural skin chrominance.
    """
    if bgr is None or bgr.size == 0:
        return bgr
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def generate_photometric_views(bgr: np.ndarray) -> list[tuple[np.ndarray, int, str]]:
    """
    Generate 7 clean photometric and mild-tilt variations:
      1. Original ID card crop
      2. CLAHE shadow-normalized (LAB color space)
      3. Gamma bright (gamma=0.74, expands shadow details)
      4. Gamma dark (gamma=1.35, enhances highlight definition)
      5. Horizontal flip (bilateral facial symmetry)
      6. +8 deg mild affine rotation
      7. -8 deg mild affine rotation
    """
    h, w = bgr.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    views: list[tuple[np.ndarray, int, str]] = []

    # 1. Original
    views.append((bgr.copy(), 1, "original"))

    # 2. CLAHE shadow-normalized
    views.append((apply_clahe(bgr), 2, "clahe_shadow_normalized"))

    # 3. Gamma bright (gamma ~ 0.74)
    inv_gamma_bright = 1.0 / 1.35
    lut_bright = np.array([((i / 255.0) ** inv_gamma_bright) * 255 for i in range(256)]).astype("uint8")
    views.append((cv2.LUT(bgr, lut_bright), 3, "gamma_bright"))

    # 4. Gamma dark (gamma = 1.35)
    inv_gamma_dark = 1.35
    lut_dark = np.array([((i / 255.0) ** inv_gamma_dark) * 255 for i in range(256)]).astype("uint8")
    views.append((cv2.LUT(bgr, lut_dark), 4, "gamma_dark"))

    # 5. Horizontal flip
    views.append((cv2.flip(bgr, 1), 5, "horizontal_flip"))

    # 6. +8 deg affine rotation
    M_p8 = cv2.getRotationMatrix2D((cx, cy), 8.0, 1.0)
    rot_p8 = cv2.warpAffine(bgr, M_p8, (w, h), borderMode=cv2.BORDER_REFLECT_101)
    views.append((rot_p8, 6, "rot_plus_8deg"))

    # 7. -8 deg affine rotation
    M_m8 = cv2.getRotationMatrix2D((cx, cy), -8.0, 1.0)
    rot_m8 = cv2.warpAffine(bgr, M_m8, (w, h), borderMode=cv2.BORDER_REFLECT_101)
    views.append((rot_m8, 7, "rot_minus_8deg"))

    return views


# ---------------------------------------------------------------------------
# InsightFace model loader (lazy singleton)
# ---------------------------------------------------------------------------
_face_app = None


def get_face_app(det_thresh: float = 0.3) -> object:
    """Load and cache the InsightFace FaceAnalysis model (CPU, 640x640, detection+recognition only)."""
    global _face_app
    if _face_app is None:
        try:
            import insightface
            from insightface.app import FaceAnalysis
        except ImportError:
            log.error("insightface is not installed. Run: pip install insightface")
            sys.exit(1)

        log.info("Loading InsightFace buffalo_l model (CPU, 640×640, detection+recognition only)…")
        app = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"],
            allowed_modules=["detection", "recognition"],
        )
        app.prepare(ctx_id=-1, det_size=(640, 640), det_thresh=det_thresh)
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

        # ---------------------------------------------------------------
        # Build 7 clean photometric & mild-tilt views
        # ---------------------------------------------------------------
        views = generate_photometric_views(bgr_orig)

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
        # Compute L2-normalized Centroid Template
        # ---------------------------------------------------------------
        embs_matrix = np.array([emb for emb, _, _ in person_embeddings], dtype=np.float32)
        centroid = np.mean(embs_matrix, axis=0)
        c_norm = np.linalg.norm(centroid)
        if c_norm > 0:
            centroid = (centroid / c_norm).astype(np.float32)

        # ---------------------------------------------------------------
        # Add Centroid + Views to FAISS and SQLite
        # ---------------------------------------------------------------
        insert_person(conn, person_id, person_name, str(photo_path))

        # Store Centroid Embedding first (view_index=0)
        faiss_id = index.ntotal
        index.add(centroid.reshape(1, -1))
        insert_vector_row(conn, faiss_id, person_id, 0, "centroid_template")
        vectors_added += 1

        # Store individual view embeddings
        for emb, view_idx, view_label in person_embeddings:
            faiss_id = index.ntotal
            index.add(emb.reshape(1, -1))
            insert_vector_row(conn, faiss_id, person_id, view_idx, view_label)
            vectors_added += 1

        conn.commit()
        enrolled_count += 1
        log.info(f"  ✓ Enrolled [{person_id}] {person_name} "
                 f"({len(person_embeddings)} views + Centroid Template)")

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
