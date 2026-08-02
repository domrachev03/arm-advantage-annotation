"""Non-destructive LeRobot v3 export for ARM direct-pair annotations.

The database is deliberately not imported here.  Callers provide immutable episode,
annotation, and completion records; this module validates the complete annotation grid,
reconstructs the scalar consumed by FluxVLA, and writes a new dataset root.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .annotation import sample_targets

PROGRESS_COLUMN = "progress"
DEFAULT_INTERVAL_EPS = 1e-3
DEFAULT_NO_COMPLETION_CEILING = 0.95
VideoMode = Literal["symlink", "copy", "none"]


class ExportError(ValueError):
    """The requested export would be incomplete, unsafe, or consumer-incompatible."""


@dataclass(frozen=True, slots=True)
class EpisodeMetadata:
    """Episode facts needed to prove annotation coverage and align parquet rows."""

    episode_index: int
    length: int
    delta_frames: int
    fps: float | None = None
    uid: str | None = None


@dataclass(frozen=True, slots=True)
class PairLabel:
    """One direct human label for ``target_frame - delta_frames -> target_frame``."""

    episode_index: int
    target_frame: int
    delta_frames: int
    label: int
    annotator: str
    start_frame: int | None = None
    annotation_id: int | None = None
    revision: int = 1
    created_at: str | None = None
    updated_at: str | None = None


@dataclass(frozen=True, slots=True)
class CompletionState:
    """The explicit success answer for an episode."""

    episode_index: int
    state: Literal["marked", "never"]
    frame: int | None
    annotator: str
    updated_at: str | None = None


@dataclass(frozen=True, slots=True)
class ExportResult:
    """Paths and counts for a completed atomic export."""

    root: Path
    manifest_path: Path
    pairs_path: Path
    episodes: int
    frames: int
    pairs: int


@dataclass(frozen=True, slots=True)
class _EpisodePass:
    metadata: EpisodeMetadata
    labels: tuple[PairLabel, ...]
    completion: CompletionState | None
    progress: np.ndarray


PAIR_SCHEMA = pa.schema(
    [
        ("episode_index", pa.int64()),
        ("episode_uid", pa.string()),
        ("frame_t", pa.int64()),
        ("frame_t_gap", pa.int64()),
        ("gap_frames", pa.int64()),
        ("label", pa.int64()),
        ("delta", pa.float64()),
        ("annotator", pa.string()),
        ("annotation_id", pa.int64()),
        ("revision", pa.int64()),
        ("created_at", pa.string()),
        ("updated_at", pa.string()),
        ("window_start_frame", pa.int64()),
        ("completion_state", pa.string()),
        ("completion_frame", pa.int64()),
        ("completion_annotator", pa.string()),
        ("completion_updated_at", pa.string()),
    ]
)


def _record_dict(value: object) -> dict[str, Any]:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(f"expected a dataclass or mapping, got {type(value).__name__}")


def _episode(value: EpisodeMetadata | Mapping[str, Any]) -> EpisodeMetadata:
    if isinstance(value, EpisodeMetadata):
        return value
    row = _record_dict(value)
    length = row.get("length", row.get("frames"))
    if length is None:
        raise ExportError("episode metadata is missing length")
    return EpisodeMetadata(
        episode_index=int(row["episode_index"]),
        length=int(length),
        delta_frames=int(row["delta_frames"]),
        fps=float(row["fps"]) if row.get("fps") is not None else None,
        uid=str(row["uid"]) if row.get("uid") is not None else None,
    )


def _pair(value: PairLabel | Mapping[str, Any]) -> PairLabel:
    if isinstance(value, PairLabel):
        return value
    row = _record_dict(value)
    return PairLabel(
        episode_index=int(row["episode_index"]),
        target_frame=int(row["target_frame"]),
        delta_frames=int(row["delta_frames"]),
        label=int(row["label"]),
        annotator=str(row.get("annotator", "")),
        start_frame=int(row["start_frame"]) if row.get("start_frame") is not None else None,
        annotation_id=(
            int(row["annotation_id"])
            if row.get("annotation_id") is not None
            else int(row["id"])
            if row.get("id") is not None
            else None
        ),
        revision=int(row.get("revision", 1)),
        created_at=str(row["created_at"]) if row.get("created_at") is not None else None,
        updated_at=str(row["updated_at"]) if row.get("updated_at") is not None else None,
    )


def _completion(value: CompletionState | Mapping[str, Any]) -> CompletionState:
    if isinstance(value, CompletionState):
        return value
    row = _record_dict(value)
    return CompletionState(
        episode_index=int(row["episode_index"]),
        state=str(row["state"]),  # type: ignore[arg-type]
        frame=int(row["frame"]) if row.get("frame") is not None else None,
        annotator=str(row.get("annotator", "")),
        updated_at=str(row["updated_at"]) if row.get("updated_at") is not None else None,
    )


def deadband_label(delta: float, interval_eps: float = DEFAULT_INTERVAL_EPS) -> int:
    """Apply the exact symmetric tri-state deadband used by ``ARMDataset``."""

    if delta > interval_eps:
        return 1
    if delta < -interval_eps:
        return -1
    return 0


def reconstruct_episode_progress(
    episode: EpisodeMetadata | Mapping[str, Any],
    labels: Iterable[PairLabel | Mapping[str, Any]],
    completion: CompletionState | Mapping[str, Any],
    *,
    interval_eps: float = DEFAULT_INTERVAL_EPS,
    no_completion_ceiling: float = DEFAULT_NO_COMPLETION_CEILING,
) -> np.ndarray:
    """Reconstruct and validate one episode's per-frame float32 progress.

    Direct labels are accumulated without a floor, then min-max scaled.  A marked
    completion is exactly ``1.0`` from its frame onward; an explicit non-completion
    remains below FluxVLA's success threshold.  The function refuses any curve whose
    float32 endpoint deltas do not reproduce every direct label.
    """

    metadata = _episode(episode)
    episode_labels = tuple(sorted((_pair(item) for item in labels), key=lambda item: item.target_frame))
    completion_state = _completion(completion)
    _validate_episode_inputs(metadata, episode_labels, completion_state)
    _validate_thresholds(interval_eps, no_completion_ceiling)

    raw = np.zeros(metadata.length, dtype=np.float64)
    level = 0.0
    cursor = 0
    for item in episode_labels:
        raw[cursor : item.target_frame] = level
        level += float(item.label)
        raw[item.target_frame] = level
        cursor = item.target_frame + 1
    raw[cursor:] = level

    progress = _normalize(raw, completion_state, no_completion_ceiling)
    progress = np.asarray(progress, dtype=np.float32)
    _validate_progress(metadata, episode_labels, completion_state, progress, interval_eps)
    return progress


def interpolate_progress_curve(
    length: int, points: Iterable[Mapping[str, Any]]
) -> np.ndarray:
    """Linearly interpolate explicit progress keypoints to one float32 value per frame."""
    rows = sorted((int(p["frame"]), float(p["value"])) for p in points)
    if length <= 0 or len(rows) < 2:
        raise ExportError("a progress curve needs at least two keypoints")
    frames = [frame for frame, _ in rows]
    if len(frames) != len(set(frames)) or frames[0] != 0 or frames[-1] != length - 1:
        raise ExportError("progress keypoints must be unique and include first and last frames")
    if any(not 0 <= value <= 1 for _, value in rows):
        raise ExportError("progress keypoint values must be within [0, 1]")
    return np.interp(np.arange(length), frames, [value for _, value in rows]).astype(np.float32)


def _validate_episode_inputs(
    episode: EpisodeMetadata,
    labels: tuple[PairLabel, ...],
    completion: CompletionState,
) -> None:
    prefix = f"episode {episode.episode_index}"
    if episode.episode_index < 0:
        raise ExportError(f"{prefix}: episode_index must be non-negative")
    if episode.length <= 0:
        raise ExportError(f"{prefix}: length must be positive")
    if episode.delta_frames <= 0:
        raise ExportError(f"{prefix}: delta_frames must be positive")
    if episode.fps is not None and episode.fps <= 0:
        raise ExportError(f"{prefix}: fps must be positive")
    if completion.episode_index != episode.episode_index:
        raise ExportError(f"{prefix}: completion belongs to episode {completion.episode_index}")
    if completion.state not in ("marked", "never"):
        raise ExportError(f"{prefix}: completion state must be 'marked' or 'never'")
    if not completion.annotator.strip():
        raise ExportError(f"{prefix}: completion has no annotator provenance")
    if completion.state == "marked":
        if completion.frame is None or not 0 <= completion.frame < episode.length:
            raise ExportError(f"{prefix}: marked completion frame is outside the episode")
    elif completion.frame is not None:
        raise ExportError(f"{prefix}: a 'never' completion answer cannot carry a frame")

    targets = sample_targets(episode.length, episode.delta_frames)
    if completion.state == "marked":
        assert completion.frame is not None
        targets = [target for target in targets if target <= completion.frame]
    expected = set(targets)
    actual: set[int] = set()
    for item in labels:
        if item.episode_index != episode.episode_index:
            raise ExportError(f"{prefix}: label belongs to episode {item.episode_index}")
        if item.delta_frames != episode.delta_frames:
            raise ExportError(
                f"{prefix}: frame {item.target_frame} uses delta {item.delta_frames}, "
                f"expected {episode.delta_frames}"
            )
        if item.label not in (-1, 0, 1):
            raise ExportError(f"{prefix}: frame {item.target_frame} has invalid label {item.label}")
        if not item.annotator.strip():
            raise ExportError(f"{prefix}: frame {item.target_frame} has no annotator provenance")
        if item.revision <= 0:
            raise ExportError(f"{prefix}: frame {item.target_frame} has invalid revision")
        expected_window_start = item.target_frame - 4 * episode.delta_frames
        if item.start_frame is not None and item.start_frame != expected_window_start:
            raise ExportError(
                f"{prefix}: frame {item.target_frame} window starts at {item.start_frame}, "
                f"expected {expected_window_start}"
            )
        if item.target_frame in actual:
            raise ExportError(f"{prefix}: duplicate label at frame {item.target_frame}")
        actual.add(item.target_frame)

    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        detail: list[str] = []
        if missing:
            detail.append(f"missing target frames {_short_frames(missing)}")
        if unexpected:
            detail.append(f"unexpected target frames {_short_frames(unexpected)}")
        raise ExportError(f"{prefix}: annotation grid is incomplete: {'; '.join(detail)}")


def _validate_thresholds(interval_eps: float, no_completion_ceiling: float) -> None:
    if not 0 < interval_eps < 1:
        raise ExportError("interval_eps must be between 0 and 1")
    if not 0 <= no_completion_ceiling < 1 - interval_eps:
        raise ExportError(
            "no_completion_ceiling must stay below FluxVLA's 1 - interval_eps success threshold"
        )


def _normalize(
    raw: np.ndarray,
    completion: CompletionState,
    no_completion_ceiling: float,
) -> np.ndarray:
    if completion.state == "never":
        floor = float(np.min(raw))
        scale = float(np.max(raw)) - floor
        if scale <= 0:
            return np.zeros_like(raw)
        return (raw - floor) / scale * no_completion_ceiling

    assert completion.frame is not None
    frame = completion.frame
    before = raw[: frame + 1]
    floor = float(np.min(before))
    scale = float(raw[frame]) - floor
    if scale <= 0:
        normalized = np.zeros_like(raw)
    else:
        normalized = np.clip((raw - floor) / scale, 0.0, 1.0)
    normalized[frame:] = 1.0
    return normalized


def _validate_progress(
    episode: EpisodeMetadata,
    labels: tuple[PairLabel, ...],
    completion: CompletionState,
    progress: np.ndarray,
    interval_eps: float,
) -> None:
    prefix = f"episode {episode.episode_index}"
    if progress.shape != (episode.length,) or progress.dtype != np.float32:
        raise ExportError(f"{prefix}: reconstruction did not produce one float32 value per frame")
    if not np.all(np.isfinite(progress)):
        raise ExportError(f"{prefix}: reconstruction produced non-finite progress")
    if np.any(progress < 0) or np.any(progress > 1):
        raise ExportError(f"{prefix}: reconstruction produced progress outside [0, 1]")
    if completion.state == "marked":
        assert completion.frame is not None
        if not np.all(progress[completion.frame :] == np.float32(1.0)):
            raise ExportError(f"{prefix}: marked completion is not held at 1.0")
    elif float(np.max(progress)) >= 1 - interval_eps:
        raise ExportError(f"{prefix}: explicit non-completion reaches FluxVLA's success threshold")

    for item in labels:
        frame_t = item.target_frame - item.delta_frames
        delta = float(progress[item.target_frame]) - float(progress[frame_t])
        derived = deadband_label(delta, interval_eps)
        if derived != item.label:
            raise ExportError(
                f"{prefix}: frame pair ({frame_t}, {item.target_frame}) label {item.label} "
                f"round-trips as {derived} after float32 export (delta={delta:.9g})"
            )


def export_dataset(
    dataset_root: Path | str,
    export_root: Path | str,
    episodes: Iterable[EpisodeMetadata | Mapping[str, Any]],
    labels: Iterable[PairLabel | Mapping[str, Any]],
    completions: Iterable[CompletionState | Mapping[str, Any]],
    *,
    video_mode: VideoMode = "symlink",
    interval_eps: float = DEFAULT_INTERVAL_EPS,
    no_completion_ceiling: float = DEFAULT_NO_COMPLETION_CEILING,
    progress_keypoints: Iterable[Mapping[str, Any]] | None = None,
) -> ExportResult:
    """Atomically export a complete ARM annotation set as a LeRobot v3 copy.

    ``dataset_root`` is read-only by contract.  ``export_root`` must not exist and
    must not overlap it.  ``meta`` and ``data`` are always copied; ``videos`` is
    linked, copied, or omitted according to ``video_mode``.
    """

    source = Path(dataset_root).expanduser().resolve()
    output = Path(export_root).expanduser().resolve()
    _validate_roots(source, output, video_mode)
    _validate_thresholds(interval_eps, no_completion_ceiling)
    version, previous_feature = _read_v3_info(source)

    episode_rows = tuple(_episode(item) for item in episodes)
    pair_rows = tuple(_pair(item) for item in labels)
    completion_rows = tuple(_completion(item) for item in completions)
    keypoint_rows = tuple(progress_keypoints or ())
    curve_mode = progress_keypoints is not None
    passes = (
        _build_curve_passes(
            episode_rows,
            keypoint_rows,
            completion_rows,
            no_completion_ceiling=no_completion_ceiling,
        )
        if curve_mode
        else _build_passes(
            episode_rows,
            pair_rows,
            completion_rows,
            interval_eps=interval_eps,
            no_completion_ceiling=no_completion_ceiling,
        )
    )
    frame_count = _validate_data_rows(source, passes)

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=str(output.parent))
    ).resolve()
    try:
        shutil.copytree(source / "meta", staging / "meta", symlinks=False)
        shutil.copytree(source / "data", staging / "data", symlinks=False)
        _write_videos(source, staging, video_mode)
        _patch_info(staging)
        written = _patch_data(staging, passes)
        if written != frame_count:
            raise ExportError(f"patched {written} data rows, expected {frame_count}")

        pair_records = _pair_records(passes)
        pairs_path = staging / "meta" / "arm_pairs.parquet"
        pq.write_table(_pairs_table(pair_records), pairs_path, compression="snappy")
        manifest = _manifest(
            source=source,
            output=output,
            version=version,
            previous_feature=previous_feature,
            video_mode=video_mode,
            interval_eps=interval_eps,
            no_completion_ceiling=no_completion_ceiling,
            passes=passes,
            frame_count=frame_count,
            pair_records=pair_records,
            curve_mode=curve_mode,
        )
        manifest_path = staging / "meta" / "arm_export_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

        if output.exists():
            raise ExportError(f"export root appeared while exporting: {output}")
        staging.rename(output)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise

    return ExportResult(
        root=output,
        manifest_path=output / "meta" / "arm_export_manifest.json",
        pairs_path=output / "meta" / "arm_pairs.parquet",
        episodes=len(passes),
        frames=frame_count,
        pairs=sum(len(item.labels) for item in passes.values()),
    )


def export_fluxvla_dataset(
    *,
    source_root: Path | str,
    output_root: Path | str,
    episodes: Iterable[EpisodeMetadata | Mapping[str, Any]],
    annotations: Iterable[PairLabel | Mapping[str, Any]],
    completions: Iterable[CompletionState | Mapping[str, Any]],
    delta_frames: int,
    fps: float,
    video_mode: VideoMode = "symlink",
    interval_eps: float = DEFAULT_INTERVAL_EPS,
    no_completion_ceiling: float = DEFAULT_NO_COMPLETION_CEILING,
    progress_keypoints: Iterable[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Service adapter for records whose dataset-wide Δ and FPS are separate.

    This remains independent of SQLite: it accepts ordinary mappings, enriches episode
    records with the persisted dataset settings, and returns the JSON-safe manifest.
    """

    prepared_episodes: list[EpisodeMetadata] = []
    for value in episodes:
        if isinstance(value, EpisodeMetadata):
            prepared_episodes.append(value)
            continue
        row = _record_dict(value)
        prepared_episodes.append(
            EpisodeMetadata(
                episode_index=int(row["episode_index"]),
                length=int(row.get("length", row.get("frames"))),
                delta_frames=int(delta_frames),
                fps=float(fps),
                uid=str(row["uid"]) if row.get("uid") is not None else None,
            )
        )
    result = export_dataset(
        dataset_root=source_root,
        export_root=output_root,
        episodes=prepared_episodes,
        labels=annotations,
        completions=completions,
        video_mode=video_mode,
        interval_eps=interval_eps,
        no_completion_ceiling=no_completion_ceiling,
        progress_keypoints=progress_keypoints,
    )
    return json.loads(result.manifest_path.read_text(encoding="utf-8"))


def _validate_roots(source: Path, output: Path, video_mode: str) -> None:
    if video_mode not in ("symlink", "copy", "none"):
        raise ExportError("video_mode must be 'symlink', 'copy', or 'none'")
    if not (source / "meta").is_dir() or not (source / "data").is_dir():
        raise ExportError(f"{source} is missing meta/ or data/")
    if output.exists():
        raise ExportError(f"export root already exists: {output}")
    if source == output or source in output.parents or output in source.parents:
        raise ExportError("source and export roots must not overlap")
    if video_mode != "none" and not (source / "videos").is_dir():
        raise ExportError(f"{source} has no videos/ for video_mode={video_mode!r}")


def _read_v3_info(source: Path) -> tuple[str, Any]:
    path = source / "meta" / "info.json"
    try:
        info = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExportError(f"cannot read {path}: {exc}") from None
    version = str(info.get("codebase_version", ""))
    match = re.search(r"\d+", version)
    if match is None or int(match.group()) != 3:
        raise ExportError(f"{path} is not LeRobot v3 (codebase_version={version!r})")
    features = info.get("features")
    if not isinstance(features, dict):
        raise ExportError(f"{path} has no features object")
    return version, features.get(PROGRESS_COLUMN)


def _build_passes(
    episodes: tuple[EpisodeMetadata, ...],
    labels: tuple[PairLabel, ...],
    completions: tuple[CompletionState, ...],
    *,
    interval_eps: float,
    no_completion_ceiling: float,
) -> dict[int, _EpisodePass]:
    episode_counts = Counter(item.episode_index for item in episodes)
    duplicates = sorted(index for index, count in episode_counts.items() if count > 1)
    if duplicates:
        raise ExportError(f"duplicate episode metadata for {_short_frames(duplicates)}")
    completion_counts = Counter(item.episode_index for item in completions)
    duplicates = sorted(index for index, count in completion_counts.items() if count > 1)
    if duplicates:
        raise ExportError(f"duplicate completion answers for {_short_frames(duplicates)}")
    if not episodes:
        raise ExportError("cannot export an empty episode set")

    by_episode = {item.episode_index: item for item in episodes}
    by_completion = {item.episode_index: item for item in completions}
    unknown_labels = sorted({item.episode_index for item in labels} - set(by_episode))
    unknown_completions = sorted(set(by_completion) - set(by_episode))
    if unknown_labels:
        raise ExportError(f"labels reference unknown episodes {_short_frames(unknown_labels)}")
    if unknown_completions:
        raise ExportError(
            f"completion answers reference unknown episodes {_short_frames(unknown_completions)}"
        )
    missing_completions = sorted(set(by_episode) - set(by_completion))
    if missing_completions:
        raise ExportError(
            f"missing completion answers for episodes {_short_frames(missing_completions)}"
        )

    labels_by_episode: dict[int, list[PairLabel]] = {index: [] for index in by_episode}
    for item in labels:
        labels_by_episode[item.episode_index].append(item)

    passes: dict[int, _EpisodePass] = {}
    for index in sorted(by_episode):
        metadata = by_episode[index]
        episode_labels = tuple(
            sorted(labels_by_episode[index], key=lambda item: item.target_frame)
        )
        completion = by_completion[index]
        progress = reconstruct_episode_progress(
            metadata,
            episode_labels,
            completion,
            interval_eps=interval_eps,
            no_completion_ceiling=no_completion_ceiling,
        )
        passes[index] = _EpisodePass(metadata, episode_labels, completion, progress)
    return passes


def _build_curve_passes(
    episodes: tuple[EpisodeMetadata, ...],
    points: tuple[Mapping[str, Any], ...],
    completions: tuple[CompletionState, ...],
    *,
    no_completion_ceiling: float,
) -> dict[int, _EpisodePass]:
    by_episode: dict[int, list[Mapping[str, Any]]] = {item.episode_index: [] for item in episodes}
    for point in points:
        index = int(point["episode_index"])
        if index not in by_episode:
            raise ExportError(f"progress keypoint references unknown episode {index}")
        by_episode[index].append(point)
    completion_by_episode = {item.episode_index: item for item in completions}
    passes: dict[int, _EpisodePass] = {}
    for episode in episodes:
        progress = interpolate_progress_curve(episode.length, by_episode[episode.episode_index])
        completion = completion_by_episode.get(episode.episode_index)
        if completion is not None:
            if completion.state == "marked":
                if completion.frame is None or not 0 <= completion.frame < episode.length:
                    raise ExportError(
                        f"episode {episode.episode_index}: completion frame is outside the episode"
                    )
                progress[completion.frame :] = np.float32(1.0)
            elif completion.state == "never":
                progress = np.minimum(progress, np.float32(no_completion_ceiling))
            else:
                raise ExportError(
                    f"episode {episode.episode_index}: invalid completion state"
                )
        passes[episode.episode_index] = _EpisodePass(episode, (), completion, progress)
    return passes


def _data_shards(root: Path) -> list[Path]:
    shards = sorted((root / "data").rglob("*.parquet"))
    if not shards:
        raise ExportError(f"{root / 'data'} contains no parquet shards")
    return shards


def _validate_data_rows(source: Path, passes: dict[int, _EpisodePass]) -> int:
    seen: dict[int, set[int]] = {index: set() for index in passes}
    rows = 0
    for path in _data_shards(source):
        table = pq.read_table(path, columns=["episode_index", "frame_index"])
        for episode_index, frame_index in zip(
            table["episode_index"].to_pylist(),
            table["frame_index"].to_pylist(),
            strict=True,
        ):
            index = int(episode_index)
            frame = int(frame_index)
            if index not in passes:
                raise ExportError(f"{path}: data row references unknown episode {index}")
            if frame in seen[index]:
                raise ExportError(f"{path}: duplicate frame {frame} in episode {index}")
            seen[index].add(frame)
            rows += 1

    for index, item in passes.items():
        expected = set(range(item.metadata.length))
        missing = sorted(expected - seen[index])
        unexpected = sorted(seen[index] - expected)
        if missing or unexpected:
            detail: list[str] = []
            if missing:
                detail.append(f"missing {_short_frames(missing)}")
            if unexpected:
                detail.append(f"unexpected {_short_frames(unexpected)}")
            raise ExportError(f"episode {index}: parquet frame indices {'; '.join(detail)}")
    return rows


def _write_videos(source: Path, staging: Path, mode: VideoMode) -> None:
    if mode == "none":
        return
    if mode == "copy":
        shutil.copytree(source / "videos", staging / "videos", symlinks=False)
        return
    relative_target = os.path.relpath(source / "videos", start=staging)
    (staging / "videos").symlink_to(relative_target, target_is_directory=True)


def _patch_info(staging: Path) -> None:
    path = staging / "meta" / "info.json"
    info = json.loads(path.read_text(encoding="utf-8"))
    info["features"][PROGRESS_COLUMN] = {
        "dtype": "float32",
        "shape": [1],
        "names": None,
    }
    path.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")


def _patch_data(staging: Path, passes: dict[int, _EpisodePass]) -> int:
    written = 0
    for path in _data_shards(staging):
        table = pq.read_table(path)
        values = [
            float(passes[int(episode_index)].progress[int(frame_index)])
            for episode_index, frame_index in zip(
                table["episode_index"].to_pylist(),
                table["frame_index"].to_pylist(),
                strict=True,
            )
        ]
        column = pa.array(values, type=pa.float32())
        if PROGRESS_COLUMN in table.column_names:
            position = table.column_names.index(PROGRESS_COLUMN)
            table = table.set_column(position, PROGRESS_COLUMN, column)
        else:
            table = table.append_column(PROGRESS_COLUMN, column)
        pq.write_table(table, path, compression="snappy")
        written += table.num_rows
    return written


def _pair_records(passes: dict[int, _EpisodePass]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index in sorted(passes):
        item = passes[index]
        if item.completion is None:
            continue
        for label in item.labels:
            frame_t = label.target_frame - label.delta_frames
            delta = float(item.progress[label.target_frame]) - float(item.progress[frame_t])
            records.append(
                {
                    "episode_index": index,
                    "episode_uid": item.metadata.uid or f"episode-{index}",
                    "frame_t": frame_t,
                    "frame_t_gap": label.target_frame,
                    "gap_frames": label.delta_frames,
                    "label": label.label,
                    "delta": delta,
                    "annotator": label.annotator,
                    "annotation_id": label.annotation_id,
                    "revision": label.revision,
                    "created_at": label.created_at,
                    "updated_at": label.updated_at,
                    "window_start_frame": (
                        label.start_frame
                        if label.start_frame is not None
                        else label.target_frame - 4 * label.delta_frames
                    ),
                    "completion_state": item.completion.state,
                    "completion_frame": item.completion.frame,
                    "completion_annotator": item.completion.annotator,
                    "completion_updated_at": item.completion.updated_at,
                }
            )
    return records


def _pairs_table(records: list[dict[str, Any]]) -> pa.Table:
    columns = {name: [record[name] for record in records] for name in PAIR_SCHEMA.names}
    return pa.table(columns, schema=PAIR_SCHEMA)


def _manifest(
    *,
    source: Path,
    output: Path,
    version: str,
    previous_feature: Any,
    video_mode: VideoMode,
    interval_eps: float,
    no_completion_ceiling: float,
    passes: dict[int, _EpisodePass],
    frame_count: int,
    pair_records: list[dict[str, Any]],
    curve_mode: bool,
) -> dict[str, Any]:
    label_counts = Counter(record["label"] for record in pair_records)
    return {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "tool": "annotation.app.exporter:export_dataset",
        "source_root": str(source),
        "out_root": str(output),
        "lerobot_codebase_version": version,
        "video_mode": video_mode,
        "progress_column": PROGRESS_COLUMN,
        "progress_feature": {"dtype": "float32", "shape": [1], "names": None},
        "progress_column_replaced": previous_feature,
        "pairs_file": "meta/arm_pairs.parquet",
        "pair_deadband": interval_eps,
        "completion_threshold": 1 - interval_eps,
        "no_completion_ceiling": no_completion_ceiling,
        "annotation_mode": "curve" if curve_mode else "direct",
        "reconstruction": {
            "input": (
                "progress keypoints"
                if curve_mode
                else "direct labels for (target_frame - delta_frames, target_frame)"
            ),
            "grid": "five-frame causal windows; first target is 4 * delta_frames",
            "accumulator": "unbounded cumulative sum; each label lands on target_frame",
            "interpolation": (
                "linear between progress keypoints"
                if curve_mode
                else "step/hold between target frames"
            ),
            "normalization": (
                "none; keypoint values are stored in [0, 1]"
                if curve_mode
                else "episode-relative min-max after accumulation"
            ),
            "marked_completion": "completion frame is exactly 1.0 and held afterward",
            "never_completion": "scaled maximum is no_completion_ceiling",
            "round_trip": (
                "float32 endpoint delta must reproduce each label with FluxVLA's "
                "symmetric pair_deadband"
            ),
        },
        "totals": {
            "episodes": len(passes),
            "frames": frame_count,
            "pairs": len(pair_records),
            "marked_completions": sum(
                item.completion is not None and item.completion.state == "marked" for item in passes.values()
            ),
            "never_completions": sum(
                item.completion is not None and item.completion.state == "never" for item in passes.values()
            ),
            "pair_labels": {
                "-1": label_counts[-1],
                "0": label_counts[0],
                "1": label_counts[1],
            },
        },
        "episode_detail": [
            {
                "episode_index": index,
                "uid": item.metadata.uid,
                "frames": item.metadata.length,
                "fps": item.metadata.fps,
                "delta_frames": item.metadata.delta_frames,
                "pairs": len(item.labels),
                "completion": {
                    "state": item.completion.state,
                    "frame": item.completion.frame,
                    "annotator": item.completion.annotator,
                    "updated_at": item.completion.updated_at,
                } if item.completion is not None else None,
                "min_progress": float(np.min(item.progress)),
                "max_progress": float(np.max(item.progress)),
            }
            for index, item in sorted(passes.items())
        ],
    }


def _short_frames(frames: list[int], limit: int = 8) -> str:
    shown = ", ".join(str(frame) for frame in frames[:limit])
    if len(frames) > limit:
        return f"[{shown}, ...] ({len(frames)} total)"
    return f"[{shown}]"
