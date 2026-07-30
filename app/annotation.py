from __future__ import annotations

from typing import Any


def sample_targets(length: int, delta_frames: int, window_size: int = 5) -> list[int]:
    """Targets whose full causal five-frame context stays inside the episode."""
    if length <= 0 or delta_frames <= 0:
        return []
    first = (window_size - 1) * delta_frames
    return list(range(first, length, delta_frames))


def frame_window(target_frame: int, delta_frames: int, window_size: int = 5) -> list[int]:
    first = target_frame - (window_size - 1) * delta_frames
    if first < 0:
        raise ValueError("target does not have a full causal window")
    return [first + i * delta_frames for i in range(window_size)]


def required_targets(
    length: int,
    delta_frames: int,
    completion: dict[str, Any] | None,
    window_size: int = 5,
) -> list[int]:
    targets = sample_targets(length, delta_frames, window_size)
    if completion and completion["state"] == "marked":
        targets = [target for target in targets if target <= int(completion["frame"])]
    return targets
