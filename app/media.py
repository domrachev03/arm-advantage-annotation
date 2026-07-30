from __future__ import annotations

import hashlib
import os
import subprocess
import threading
from pathlib import Path

from fastapi import HTTPException

from .config import CACHE_ROOT

# Different frames are safe to decode concurrently. Striped locks still prevent
# two requests for the same cache path from writing the same temporary file,
# while the semaphore keeps a busy annotator from spawning unbounded ffmpeg
# processes.
_decode_locks = tuple(threading.Lock() for _ in range(64))
_decode_slots = threading.BoundedSemaphore(4)


def camera_cache_key(camera: str) -> str:
    return hashlib.sha256(camera.encode()).hexdigest()[:12]


def frame_cache_path(dataset_id: int, episode_index: int, camera: str, frame: int) -> Path:
    return (
        CACHE_ROOT
        / f"dataset-{dataset_id:06d}"
        / camera_cache_key(camera)
        / f"episode-{episode_index:06d}"
        / f"frame-{frame:08d}.jpg"
    )


def decode_frame(
    video_path: Path,
    *,
    timestamp: float,
    output_path: Path,
    max_width: int = 640,
) -> Path:
    if output_path.is_file() and output_path.stat().st_size > 0:
        return output_path
    if not video_path.is_file():
        raise HTTPException(404, f"video file is missing: {video_path.name}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".tmp")
    path_lock = _decode_locks[hash(output_path) % len(_decode_locks)]
    with path_lock, _decode_slots:
        if output_path.is_file() and output_path.stat().st_size > 0:
            return output_path
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{max(0.0, timestamp):.9f}",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-vf",
            f"scale='min({max_width},iw)':-2",
            "-q:v",
            "3",
            "-f",
            "image2",
            "-vcodec",
            "mjpeg",
            "-y",
            str(temporary),
        ]
        result = subprocess.run(command, capture_output=True, text=True, timeout=60, check=False)
        if result.returncode or not temporary.is_file() or temporary.stat().st_size == 0:
            temporary.unlink(missing_ok=True)
            detail = result.stderr.strip()[-500:]
            raise HTTPException(422, f"ffmpeg could not decode frame: {detail}")
        os.replace(temporary, output_path)
    return output_path
