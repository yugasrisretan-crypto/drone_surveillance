"""
enroll_ids.py
=============
Phase 1 — High-Speed, High-Accuracy CPU ID Card Enrollment Pipeline
Using InsightFace buffalo_sc + Strict 5-Point norm_crop Landmark Alignment.

Features:
  - buffalo_sc model with tuned ONNX Runtime session options (intra_op_num_threads=4,
    execution_mode=ORT_SEQUENTIAL) for 6x faster CPU execution.
  - Strict 5-point landmark norm_crop(img, landmark=face.kps, image_size=112) alignment.
  - 5 clean geometric variations per photo (original, CLAHE, flip, +7 deg tilt, -7 deg tilt).
  - L2-normalized Mean Centroid Vector computed and stored in FAISS.
  - Multi-photo prefix merging (e.g. EMP001_Name_1.jpg, EMP001_Name_2.jpg -> single person).
  - FAISS IndexFlatIP (storage/campus.faiss) + SQLite (storage/campus.db).

Usage:
    python enroll_ids.py [--id_cards_dir data/id_cards]
                         [--storage_dir storage]
                         [--det_thresh 0.3]
                         [--rebuild]
                         [--verbose]
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Auto-dispatch to project .venv if invoked with system python
# ---------------------------------------------------------------------------
_venv_python = Path(__file__).resolve().parent / ".venv" / "Scripts" / "python.exe"
if _venv_python.exists() and Path(sys.executable).resolve() != _venv_python.resolve():
    sys.exit(subprocess.call([str(_venv_python)] + sys.argv))

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import cv2
import faiss
import numpy as np
import onnxruntime as ort

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
EMBEDDING_DIM = 512
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ---------------------------------------------------------------------------
# Image processing helpers
# ---------------------------------------------------------------------------
def apply_clahe(bgr: np.ndarray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """Apply CLAHE luminance normalization in LAB color space."""
    if bgr is None or bgr.size == 0:
        return bgr
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def rotate_face(bgr: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate aligned 112x112 face around its center with reflection padding."""
    h, w = bgr.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    return cv2.warpAffine(bgr, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)


def generate_aligned_variations(aligned_face: np.ndarray) -> List[Tuple[np.ndarray, int, str]]:
    """
    Generate 5 clean variations from the strictly 5-point aligned 112x112 face:
      1. Original aligned face
      2. CLAHE illumination-normalized
      3. Horizontal flip
      4. +8 deg mild rotation
      5. -8 deg mild rotation
    """
    views: List[Tuple[np.ndarray, int, str]] = []
    # 1. Original
    views.append((aligned_face.copy(), 1, "original"))
    # 2. CLAHE normalized
    views.append((apply_clahe(aligned_face), 2, "clahe_normalized"))
    # 3. Horizontal flip
    views.append((cv2.flip(aligned_face, 1), 3, "horizontal_flip"))
    # 4. +8 deg tilt
    views.append((rotate_face(aligned_face, 8.0), 4, "rot_plus_8deg"))
    # 5. -8 deg tilt
    views.append((rotate_face(aligned_face, -8.0), 5, "rot_minus_8deg"))
    return views


# ---------------------------------------------------------------------------
# InsightFace buffalo_l Model Loader (Pure ResNet-50 w600k_r50.onnx)
# ---------------------------------------------------------------------------
_face_app = None

def get_face_app(det_thresh: float = 0.3):
    """Load and cache InsightFace buffalo_l model tuned for CPU."""
    global _face_app
    if _face_app is None:
        try:
            from insightface.app import FaceAnalysis
        except ImportError:
            log.error("insightface is not installed. Run: pip install insightface")
            sys.exit(1)

        log.info("Loading InsightFace buffalo_l (detection + recognition) with det_size=(320, 320)...")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        app = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"],
            allowed_modules=["detection", "recognition"],
            session_options=opts,
        )
        app.prepare(ctx_id=-1, det_size=(320, 320), det_thresh=det_thresh)
        _face_app = app
        log.info("buffalo_l model loaded successfully.")
    return _face_app


# ---------------------------------------------------------------------------
# Database Helpers
# ---------------------------------------------------------------------------
def init_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS persons (
            person_id   TEXT PRIMARY KEY,
            name        TEXT NOT NULL,
            photo_path  TEXT NOT NULL,
            enrolled_at REAL NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS vectors (
            faiss_id    INTEGER PRIMARY KEY,
            person_id   TEXT NOT NULL,
            view_index  INTEGER NOT NULL,
            view_label  TEXT NOT NULL,
            FOREIGN KEY (person_id) REFERENCES persons(person_id)
        )
    """)
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
    conn.execute("""
        CREATE VIEW IF NOT EXISTS vector_map AS
        SELECT faiss_id, person_id, view_index, view_label FROM vectors;
    """)
    conn.commit()
    return conn


def parse_filename(photo_path: Path, auto_index: int) -> Tuple[str, str]:
    """
    Parse filename to extract (person_id, person_name), supporting multi-photo prefixes:
      - EMP001_Authorized_Student.jpg   -> ("EMP001", "Authorized Student")
      - EMP001_Authorized_Student_1.jpg -> ("EMP001", "Authorized Student")
      - EMP001_Authorized_Student_2.jpg -> ("EMP001", "Authorized Student")
      - Arbitrary file names            -> ("EMP00X", "stem")
    """
    stem = photo_path.stem
    parts = stem.split("_")
    if len(parts) >= 2 and parts[0].isalnum() and len(parts[0]) <= 10:
        person_id = parts[0].upper()
        name_parts = parts[1:]
        if len(name_parts) > 1 and name_parts[-1].isdigit():
            name_parts = name_parts[:-1]
        person_name = " ".join(name_parts).replace("_", " ").strip()
        if not person_name:
            person_name = stem
    else:
        person_id = f"EMP{auto_index:03d}"
        person_name = stem.replace("_", " ").strip()
    return person_id, person_name


# ---------------------------------------------------------------------------
# Enrollment Execution
# ---------------------------------------------------------------------------
def enroll_all(
    id_cards_dir: Path,
    storage_dir: Path,
    det_thresh: float = 0.3,
    verbose: bool = False,
    rebuild: bool = False,
) -> dict:
    if verbose:
        log.setLevel(logging.DEBUG)

    storage_dir.mkdir(parents=True, exist_ok=True)
    faiss_path = storage_dir / "campus.faiss"
    db_path = storage_dir / "campus.db"

    if rebuild:
        log.info("Rebuilding index and database from scratch (--rebuild)")
        if faiss_path.exists():
            faiss_path.unlink()
        if db_path.exists():
            db_path.unlink()

    # Load or initialize FAISS index
    if faiss_path.exists():
        log.info(f"Loading existing FAISS index from {faiss_path}")
        index = faiss.read_index(str(faiss_path))
        log.info(f"  -> {index.ntotal} vectors already in index")
    else:
        log.info("Creating new FAISS IndexFlatIP index (dim=512)")
        index = faiss.IndexFlatIP(EMBEDDING_DIM)

    conn = init_db(db_path)

    # Discover ID card photos
    photos = sorted(p for p in id_cards_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTS)
    if not photos:
        log.warning(f"No ID card photos found in {id_cards_dir}")
        conn.close()
        return {"enrolled": 0, "vectors_added": 0, "failed": []}

    log.info(f"Found {len(photos)} ID card photo(s) in {id_cards_dir}")

    # Group photos by person_id (multi-photo prefix merging)
    person_photos: Dict[str, Dict] = {}
    auto_idx = 1
    for p in photos:
        pid, pname = parse_filename(p, auto_idx)
        if pid not in person_photos:
            person_photos[pid] = {
                "name": pname,
                "photos": [p],
            }
            auto_idx += 1
        else:
            person_photos[pid]["photos"].append(p)

    app = get_face_app(det_thresh)
    from insightface.utils.face_align import norm_crop
    rec_model = app.models["recognition"]

    enrolled_count = 0
    vectors_added = 0
    failed_photos = []

    for person_id, info in person_photos.items():
        person_name = info["name"]
        photo_list: List[Path] = info["photos"]

        # Check if already enrolled
        existing = conn.execute("SELECT name FROM persons WHERE person_id=?", (person_id,)).fetchone()
        if existing and not rebuild:
            log.info(f"  [SKIP] {person_id} ({existing[0]}) already enrolled.")
            continue

        log.info(f"Processing person [{person_id}] {person_name} with {len(photo_list)} photo(s)...")

        all_embs: List[np.ndarray] = []
        vector_metadata: List[Tuple[int, str]] = []  # (view_idx, view_label)
        primary_photo = str(photo_list[0])

        for photo_path in photo_list:
            bgr = cv2.imread(str(photo_path))
            if bgr is None:
                log.error(f"  [FAIL] Cannot read image: {photo_path}")
                failed_photos.append(str(photo_path))
                continue

            # Resize if excessive
            h, w = bgr.shape[:2]
            if max(h, w) > 1280:
                scale = 1280.0 / max(h, w)
                bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

            # Pad image with 100px border so close-up ID card faces do not touch image edges
            padded_bgr = cv2.copyMakeBorder(bgr, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=[128, 128, 128])

            # Detect face in padded image
            faces = app.get(padded_bgr)
            if not faces:
                log.warning(f"  [WARN] No face found in {photo_path.name}")
                failed_photos.append(str(photo_path))
                continue

            # Largest face with landmarks
            best_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            if not hasattr(best_face, "kps") or best_face.kps is None or len(best_face.kps) < 5:
                log.warning(f"  [WARN] Missing 5 landmarks for face in {photo_path.name}")
                continue

            # Strict 5-point norm_crop alignment (112x112) from padded image
            aligned_face = norm_crop(padded_bgr, landmark=best_face.kps, image_size=112)

            # Generate 5 clean variations
            variations = generate_aligned_variations(aligned_face)

            for var_img, v_idx, v_label in variations:
                # Extract 512-D embedding directly from aligned 112x112 chip
                feat = rec_model.get_feat(var_img).flatten().astype(np.float32)
                norm = np.linalg.norm(feat)
                if norm > 1e-6:
                    feat = feat / norm
                    all_embs.append(feat)
                    vector_metadata.append((v_idx, v_label))

        if not all_embs:
            log.error(f"  [FAIL] Could not extract embeddings for [{person_id}] {person_name}")
            continue

        # Compute L2-normalized Mean Centroid Vector
        embs_matrix = np.array(all_embs, dtype=np.float32)
        centroid = np.mean(embs_matrix, axis=0)
        c_norm = np.linalg.norm(centroid)
        if c_norm > 1e-6:
            centroid = (centroid / c_norm).astype(np.float32)
            all_embs.append(centroid)
            vector_metadata.append((0, "mean_centroid"))

        # Add to FAISS index
        final_matrix = np.array(all_embs, dtype=np.float32)
        start_id = index.ntotal
        index.add(final_matrix)
        count_added = len(all_embs)
        vectors_added += count_added

        # Store in SQLite
        conn.execute(
            "INSERT OR REPLACE INTO persons (person_id, name, photo_path, enrolled_at) VALUES (?, ?, ?, ?)",
            (person_id, person_name, primary_photo, time.time()),
        )
        for offset, (v_idx, v_lbl) in enumerate(vector_metadata):
            faiss_id = start_id + offset
            conn.execute(
                "INSERT OR REPLACE INTO vectors (faiss_id, person_id, view_index, view_label) VALUES (?, ?, ?, ?)",
                (faiss_id, person_id, v_idx, v_lbl),
            )

        conn.commit()
        enrolled_count += 1
        log.info(f"  -> Enrolled [{person_id}] {person_name}: {count_added} vectors added (including Mean Centroid).")

    # Persist FAISS index
    faiss.write_index(index, str(faiss_path))
    conn.close()

    log.info("------------------------------------------------------------")
    log.info(f"Enrollment Complete: {enrolled_count} person(s) enrolled, {vectors_added} total vector(s).")
    if failed_photos:
        log.warning(f"Failed photo(s): {failed_photos}")
    log.info(f"FAISS index saved -> {faiss_path}")
    log.info(f"SQLite database saved -> {db_path}")
    log.info("------------------------------------------------------------")

    return {
        "enrolled": enrolled_count,
        "vectors_added": vectors_added,
        "failed": failed_photos,
    }


def main():
    parser = argparse.ArgumentParser(description="Campus Guardian — ID Card Enrollment (buffalo_sc + norm_crop)")
    parser.add_argument("--id_cards_dir", default="data/id_cards", help="Folder containing ID photos")
    parser.add_argument("--storage_dir", default="storage", help="Output folder for campus.faiss & campus.db")
    parser.add_argument("--det_thresh", type=float, default=0.3, help="Face detection threshold")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild FAISS index from scratch")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose debug logging")
    args = parser.parse_args()

    enroll_all(
        id_cards_dir=Path(args.id_cards_dir),
        storage_dir=Path(args.storage_dir),
        det_thresh=args.det_thresh,
        verbose=args.verbose,
        rebuild=args.rebuild,
    )


if __name__ == "__main__":
    main()
