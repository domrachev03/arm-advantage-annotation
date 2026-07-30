from __future__ import annotations

import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

from app.media import decode_frame


def test_distinct_frames_decode_concurrently(tmp_path: Path, monkeypatch) -> None:
    video = tmp_path / "episode.mp4"
    video.write_bytes(b"fixture")
    calls = 0
    calls_lock = Lock()

    def fake_run(command, **_kwargs):
        nonlocal calls
        with calls_lock:
            calls += 1
        time.sleep(0.12)
        Path(command[-1]).write_bytes(b"jpeg")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("app.media.subprocess.run", fake_run)
    outputs = [tmp_path / f"frame-{index}.jpg" for index in range(4)]
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=4) as pool:
        decoded = list(
            pool.map(
                lambda item: decode_frame(
                    video,
                    timestamp=float(item[0]),
                    output_path=item[1],
                ),
                enumerate(outputs),
            )
        )
    elapsed = time.monotonic() - started
    assert decoded == outputs
    assert calls == 4
    assert elapsed < 0.35


def test_same_frame_decode_is_coalesced(tmp_path: Path, monkeypatch) -> None:
    video = tmp_path / "episode.mp4"
    video.write_bytes(b"fixture")
    output = tmp_path / "frame.jpg"
    calls = 0

    def fake_run(command, **_kwargs):
        nonlocal calls
        calls += 1
        time.sleep(0.08)
        Path(command[-1]).write_bytes(b"jpeg")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("app.media.subprocess.run", fake_run)
    with ThreadPoolExecutor(max_workers=2) as pool:
        decoded = list(
            pool.map(
                lambda _index: decode_frame(video, timestamp=1.0, output_path=output),
                range(2),
            )
        )
    assert decoded == [output, output]
    assert calls == 1
