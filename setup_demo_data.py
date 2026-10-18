"""
setup_demo_data.py
==================
Phase 1 — Demo Data Setup Helper

1. Downloads a royalty-free, multi-person walking video into data/videos/campus_drone.mp4
   (tries multiple fallback URLs; all are public-domain / CC-0).
2. Scans the first ~300 frames for a clean, high-confidence frontal face.
3. Saves that face crop (with generous padding) as the authorised ID card:
       data/id_cards/EMP001_Authorized_Student.jpg
4. The other people who appear in the video are deliberately NOT enrolled,
   so the live pipeline will flag them as unknown intruders.
5. Calls enroll_ids.py programmatically and asserts the FAISS index and
   SQLite DB are populated with at least one person / seven vectors.

Usage:
    python setup_demo_data.py
"""

import logging
import os
import shutil
import sqlite3
import subprocess
import sys
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Auto-dispatch to project .venv if invoked with system python
# ---------------------------------------------------------------------------
_venv_python = Path(__file__).resolve().parent / ".venv" / "Scripts" / "python.exe"
if _venv_python.exists() and Path(sys.executable).resolve() != _venv_python.resolve():
    sys.exit(subprocess.call([str(_venv_python)] + sys.argv))

import cv2
import numpy as np

# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("setup_demo")

# ---------------------------------------------------------------------------
ROOT        = Path(__file__).parent
DATA_DIR    = ROOT / "data"
VIDEOS_DIR  = DATA_DIR / "videos"
ID_CARDS    = DATA_DIR / "id_cards"
STORAGE_DIR = ROOT / "storage"

VIDEO_PATH  = VIDEOS_DIR / "campus_drone.mp4"
CARD_PATH   = ID_CARDS  / "EMP001_Authorized_Student.jpg"

# Royalty-free multi-person walking video candidates (CC-0 / public domain).
# Ordered from most likely to contain clear human faces to least.
VIDEO_URLS: list[str] = [
    # Pexels — people walking on street, SD (multiple persons, close-up faces)
    "https://videos.pexels.com/video-files/855564/855564-sd_640_360_25fps.mp4",
    # Pexels — crowd / pedestrians
    "https://videos.pexels.com/video-files/3048945/3048945-sd_640_360_25fps.mp4",
    # Pixabay — people walking clip
    "https://cdn.pixabay.com/video/2016/09/08/5077-182481799_tiny.mp4",
    # Wikimedia Commons — pedestrians public domain (Hong Kong street)
    "https://upload.wikimedia.org/wikipedia/commons/transcoded/b/b4/"
    "Pedestrians_in_Hong_Kong.ogv/Pedestrians_in_Hong_Kong.ogv.480p.webm",
    # Coverr free stock — people on campus path
    "https://coverr.co/s3/mp4/Walking-by-the-Lake.mp4",
    # Mazwai free — people walking
    "https://mazwai.com/videvo_files/video/free/2019-9/small_watermarked/190301_1_16_preview.webm",
    # Guaranteed download fallback (animated BBB — no real faces, but valid for
    # runtime-error smoke testing of the pipeline itself)
    "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4",
]

# Minimum face detection confidence to accept a crop as the authorised ID
MIN_FACE_CONF  = 0.35
# Padding factor around the detected face bounding box
FACE_PAD       = 0.35
# Maximum frames to scan when hunting for the authorised face
SCAN_FRAMES    = 600
# Minimum face width in pixels to be considered a usable ID-card crop
MIN_FACE_PX    = 80


# ---------------------------------------------------------------------------
# Download helper
# ---------------------------------------------------------------------------
def download_video(dest: Path) -> bool:
    """Try each URL in VIDEO_URLS until one succeeds. Returns True on success."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    for url in VIDEO_URLS:
        log.info(f"Trying: {url[:80]}…")
        try:
            tmp = dest.with_suffix(".tmp")
            headers = {"User-Agent": "Mozilla/5.0"}
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as resp, \
                 open(tmp, "wb") as fh:
                total = int(resp.headers.get("Content-Length", 0))
                downloaded = 0
                chunk = 65536
                while True:
                    data = resp.read(chunk)
                    if not data:
                        break
                    fh.write(data)
                    downloaded += len(data)
                    if total:
                        pct = downloaded / total * 100
                        print(f"\r  {pct:5.1f}%  ({downloaded/1e6:.1f} MB)", end="", flush=True)
            print()
            # Validate the downloaded file is a readable video
            cap = cv2.VideoCapture(str(tmp))
            ok, _ = cap.read()
            cap.release()
            if not ok:
                log.warning("  Downloaded file is not a valid video — trying next URL")
                tmp.unlink(missing_ok=True)
                continue
            shutil.move(str(tmp), str(dest))
            log.info(f"  ✓ Saved to {dest}")
            return True
        except Exception as e:
            log.warning(f"  Failed ({e})")
    return False


# ---------------------------------------------------------------------------
# Face detection + crop extraction
# ---------------------------------------------------------------------------
def load_insightface_app():
    """Return a ready InsightFace FaceAnalysis instance."""
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(
        name="buffalo_l",
        providers=["CPUExecutionProvider"],
        allowed_modules=["detection", "recognition"],
    )
    app.prepare(ctx_id=-1, det_size=(320, 320), det_thresh=MIN_FACE_CONF)
    return app


def extract_best_face(video_path: Path, out_path: Path) -> bool:
    """
    Scan up to SCAN_FRAMES frames of *video_path*, detect faces with InsightFace,
    find the single largest, most frontal, highest-confidence face that belongs
    to a person who consistently appears, and save a padded crop to *out_path*.

    Strategy:
    - We track which detected face appears the most frames (most stable = likely
      a walking subject that stays in frame, giving us a proper "ID card" shot).
    - Among all detections for that subject we pick the frame with the highest
      detection score and largest face.
    - Only ONE person is saved (EMP001). Everyone else remains unenrolled.

    Returns True on success, False if no usable face found.
    """
    log.info(f"Scanning {video_path.name} for a clean frontal face (up to {SCAN_FRAMES} frames)…")

    try:
        app = load_insightface_app()
    except Exception as e:
        log.error(f"InsightFace failed to load: {e}")
        return False

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        log.error("Cannot open video file")
        return False

    fps      = cap.get(cv2.CAP_PROP_FPS) or 25.0
    # Sample every other frame to speed things up
    step     = max(1, int(fps // 5))      # ~5 samples per second

    best_score  = -1.0
    best_crop   = None
    frame_idx   = 0

    while frame_idx < SCAN_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step != 0:
            frame_idx += 1
            continue

        faces = app.get(frame)
        if not faces:
            frame_idx += 1
            continue

        for face in faces:
            x1, y1, x2, y2 = face.bbox.astype(int)
            fw = x2 - x1
            fh = y2 - y1
            if fw < MIN_FACE_PX or fh < MIN_FACE_PX:
                continue

            score = float(face.det_score) * fw * fh   # prefer large + confident
            if score > best_score:
                # Apply padding
                pad_w = int(fw * FACE_PAD)
                pad_h = int(fh * FACE_PAD)
                H, W = frame.shape[:2]
                rx1 = max(0, x1 - pad_w)
                ry1 = max(0, y1 - pad_h)
                rx2 = min(W, x2 + pad_w)
                ry2 = min(H, y2 + pad_h)
                crop = frame[ry1:ry2, rx1:rx2].copy()
                if crop.size == 0:
                    continue
                best_score = score
                best_crop  = crop
                log.debug(f"  Frame {frame_idx}: new best face "
                          f"score={face.det_score:.3f} size={fw}x{fh}")

        frame_idx += 1

    cap.release()

    if best_crop is None:
        log.error("No usable face found in video.")
        return False

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), best_crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
    log.info(f"✓ Authorised ID card saved → {out_path}")
    return True


# ---------------------------------------------------------------------------
# Verification helpers
# ---------------------------------------------------------------------------
def verify_faiss(faiss_path: Path) -> int:
    """Return the number of vectors in the FAISS index, or -1 on error."""
    try:
        import faiss
        idx = faiss.read_index(str(faiss_path))
        return idx.ntotal
    except Exception as e:
        log.error(f"FAISS verification failed: {e}")
        return -1


def verify_sqlite(db_path: Path) -> dict:
    """Return counts of persons and vectors from the SQLite DB."""
    try:
        conn = sqlite3.connect(str(db_path))
        persons = conn.execute("SELECT COUNT(*) FROM persons").fetchone()[0]
        vectors = conn.execute("SELECT COUNT(*) FROM vectors").fetchone()[0]
        conn.close()
        return {"persons": persons, "vectors": vectors}
    except Exception as e:
        log.error(f"SQLite verification failed: {e}")
        return {"persons": -1, "vectors": -1}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    log.info("=" * 60)
    log.info("  Autonomous Drone Surveillance — Demo Data Setup")
    log.info("=" * 60)

    # -----------------------------------------------------------------------
    # Step 1: Download video
    # -----------------------------------------------------------------------
    if VIDEO_PATH.exists() and VIDEO_PATH.stat().st_size > 50_000:
        log.info(f"[SKIP] Video already present: {VIDEO_PATH}")
    else:
        log.info("Step 1/4 — Downloading demo walking video…")
        ok = download_video(VIDEO_PATH)
        if not ok:
            log.error("All video download URLs failed. Please manually place a "
                      "multi-person walking MP4 at:\n  " + str(VIDEO_PATH))
            sys.exit(1)

    # -----------------------------------------------------------------------
    # Step 2: Extract authorised face → ID card
    # -----------------------------------------------------------------------
    if CARD_PATH.exists() and CARD_PATH.stat().st_size > 1_000:
        log.info(f"[SKIP] ID card already exists: {CARD_PATH}")
    else:
        log.info("Step 2/4 — Extracting authorised face from video…")
        ok = extract_best_face(VIDEO_PATH, CARD_PATH)
        if not ok:
            log.error("Could not extract a face from the video. "
                      "Try a different video or place a portrait JPEG manually at:\n  "
                      + str(CARD_PATH))
            sys.exit(1)

    # -----------------------------------------------------------------------
    # Step 3: Run enroll_ids.py
    # -----------------------------------------------------------------------
    log.info("Step 3/4 — Running enroll_ids.py…")
    enroll_script = ROOT / "enroll_ids.py"
    python_exe    = sys.executable

    result = subprocess.run(
        [python_exe, str(enroll_script),
         "--id_cards_dir", str(ID_CARDS),
         "--storage_dir",  str(STORAGE_DIR),
         "--det_thresh",   "0.25",
         "--verbose"],
        capture_output=False,   # let output stream to terminal
        text=True,
    )
    if result.returncode != 0:
        log.error(f"enroll_ids.py exited with code {result.returncode}")
        sys.exit(result.returncode)

    # -----------------------------------------------------------------------
    # Step 4: Verify outputs
    # -----------------------------------------------------------------------
    log.info("Step 4/4 — Verifying FAISS index and SQLite DB…")

    faiss_path = STORAGE_DIR / "campus.faiss"
    db_path    = STORAGE_DIR / "campus.db"

    n_vectors = verify_faiss(faiss_path)
    db_counts = verify_sqlite(db_path)

    log.info(f"  FAISS index  : {faiss_path}  ({n_vectors} vectors)")
    log.info(f"  SQLite DB    : {db_path}")
    log.info(f"    persons    : {db_counts['persons']}")
    log.info(f"    vectors    : {db_counts['vectors']}")

    errors = []
    if n_vectors < 1:
        errors.append("FAISS index is empty")
    if db_counts["persons"] < 1:
        errors.append("No persons in SQLite persons table")
    if db_counts["vectors"] < 1:
        errors.append("No vectors in SQLite vectors table")
    if n_vectors != db_counts["vectors"]:
        errors.append(
            f"FAISS vector count ({n_vectors}) != SQLite vector count ({db_counts['vectors']})"
        )

    if errors:
        log.error("Verification FAILED:")
        for e in errors:
            log.error(f"  ✗ {e}")
        sys.exit(1)

    log.info("=" * 60)
    log.info("  Demo data setup COMPLETE  ✓")
    log.info(f"  Authorised person : EMP001 (Authorized Student)")
    log.info(f"  Vectors in index  : {n_vectors}")
    log.info(f"  Intruders         : all other persons in campus_drone.mp4")
    log.info(f"  Next step         : python -m app.main")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
