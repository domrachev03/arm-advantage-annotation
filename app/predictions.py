"""Storage for ARM prediction runs.

One uploaded artifact (docs/model_predictions.md) becomes one `prediction_run`
row, one `prediction_episode` row per covered episode, one `prediction_frame`
row per frame, and one `prediction_interval` row per evaluated causal window.
The artifact is validated by the upload endpoint (`app/prediction_artifact.py`);
this module writes what it is given, reads it back, and thins one episode's
series down to what a chart needs.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Iterator
from typing import Any

from .db import connect, dumps, loads, now, transaction

# Per-frame rows are inserted in batches of this size so a 60 000-frame run
# stays a bounded number of parameter buffers instead of one enormous list.
INSERT_BATCH_ROWS = 10_000

# A compare chart is a few hundred pixels wide, so an episode's series is thinned
# to this many frames unless the caller asks for more.
DEFAULT_SERIES_POINTS = 500

# Frames kept per bucket by `downsample_series`: the bucket's first frame plus
# the extremes of both curves.
FRAMES_PER_BUCKET = 5

RUN_COLUMNS = (
    "dataset_id",
    "name",
    "schema_version",
    "artifact_kind",
    "tool",
    "generated_at",
    "checkpoint_path",
    "checkpoint_sha256",
    "config_id",
    "config_path",
    "git_sha",
    "git_dirty",
    "training_command",
    "seed",
    "trained_at",
    "split_file",
    "split_file_sha256",
    "notes",
    "delta_frames",
    "window_size",
    "interval_eps",
    "no_completion_ceiling",
    "gt_progress_source",
    "dataset_json",
    "aggregate_metrics_json",
    "artifact_sha256",
    "artifact_bytes",
    "uploaded_by",
    "created_at",
    "updated_at",
)

INSERT_RUN_SQL = (
    f"INSERT INTO prediction_run({','.join(RUN_COLUMNS)})"
    f" VALUES({','.join('?' * len(RUN_COLUMNS))})"
)

INSERT_EPISODE_SQL = (
    "INSERT INTO prediction_episode(run_id,dataset_id,episode_index,split,length,frames,intervals,"
    "spearman,pearson,mae,interval_accuracy,linear_ramp_baseline_json,"
    "success_predicted_probability,success_predicted,success_gt,success_gt_frame)"
    " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
)

INSERT_FRAME_SQL = (
    "INSERT INTO prediction_frame(run_id,dataset_id,episode_index,frame_index,"
    "predicted_progress,gt_progress) VALUES(?,?,?,?,?,?)"
)

SELECT_RUN_SQL = (
    "SELECT prediction_run.*,"
    " (SELECT COUNT(*) FROM prediction_episode WHERE run_id=prediction_run.id) AS episode_count"
    " FROM prediction_run"
)

INSERT_INTERVAL_SQL = (
    "INSERT INTO prediction_interval(run_id,dataset_id,episode_index,target_frame,start_frame,"
    "delta_frames,predicted_label,gt_label,predicted_probabilities_json) VALUES(?,?,?,?,?,?,?,?,?)"
)


def _optional_bool(value: Any) -> int | None:
    return None if value is None else int(bool(value))


def _insert_batched(
    conn: sqlite3.Connection,
    statement: str,
    rows: Iterable[tuple[Any, ...]],
    *,
    batch: int = INSERT_BATCH_ROWS,
) -> int:
    buffer: list[tuple[Any, ...]] = []
    written = 0
    for row in rows:
        buffer.append(row)
        if len(buffer) >= batch:
            conn.executemany(statement, buffer)
            written += len(buffer)
            buffer.clear()
    if buffer:
        conn.executemany(statement, buffer)
        written += len(buffer)
    return written


def _frame_rows(
    run_id: int, dataset_id: int, episodes: list[dict[str, Any]]
) -> Iterator[tuple[Any, ...]]:
    for episode in episodes:
        episode_index = int(episode["episode_index"])
        pairs = zip(episode["predicted_progress"], episode["gt_progress"], strict=True)
        for frame_index, (predicted, ground_truth) in enumerate(pairs):
            yield (
                run_id,
                dataset_id,
                episode_index,
                frame_index,
                float(predicted),
                float(ground_truth),
            )


def _interval_rows(
    run_id: int, dataset_id: int, delta_frames: int, episodes: list[dict[str, Any]]
) -> Iterator[tuple[Any, ...]]:
    for episode in episodes:
        episode_index = int(episode["episode_index"])
        for interval in episode["intervals"]:
            target_frame = int(interval["target_frame"])
            window = interval["window_frames"]
            probabilities = interval.get("predicted_probabilities")
            gt_label = interval.get("gt_label")
            yield (
                run_id,
                dataset_id,
                episode_index,
                target_frame,
                int(window[0]) if window else target_frame - 4 * delta_frames,
                delta_frames,
                int(interval["predicted_label"]),
                None if gt_label is None else int(gt_label),
                None if probabilities is None else dumps(list(probabilities)),
            )


def insert_prediction_run(
    artifact: dict[str, Any],
    *,
    dataset_id: int,
    uploaded_by: str,
    artifact_sha256: str,
    artifact_bytes: int,
) -> int:
    """Write one validated artifact and return the new run's identifier."""
    run = artifact["run"]
    grid = artifact["grid"]
    episodes = list(artifact["episodes"])
    delta_frames = int(grid["delta_frames"])
    stamp = now()
    with transaction() as conn:
        cursor = conn.execute(
            INSERT_RUN_SQL,
            (
                dataset_id,
                run["name"],
                int(artifact["schema_version"]),
                artifact["artifact_kind"],
                artifact["tool"],
                artifact["generated_at"],
                run["checkpoint_path"],
                run["checkpoint_sha256"],
                run["config_id"],
                run["config_path"],
                run["git_sha"],
                int(bool(run["git_dirty"])),
                run["training_command"],
                run["seed"],
                run["created_at"],
                run["split_file"],
                run["split_file_sha256"],
                run.get("notes"),
                delta_frames,
                int(grid["window_size"]),
                float(grid["interval_eps"]),
                float(grid["no_completion_ceiling"]),
                grid["gt_progress_source"],
                dumps(artifact["dataset"]),
                dumps(artifact["aggregate_metrics"]),
                artifact_sha256,
                artifact_bytes,
                uploaded_by,
                stamp,
                stamp,
            ),
        )
        run_id = int(cursor.lastrowid)
        for episode in episodes:
            metrics = episode["metrics"]
            success = episode.get("success") or {}
            conn.execute(
                INSERT_EPISODE_SQL,
                (
                    run_id,
                    dataset_id,
                    int(episode["episode_index"]),
                    episode["split"],
                    int(episode["length"]),
                    int(metrics["frames"]),
                    int(metrics["intervals"]),
                    metrics["spearman"],
                    metrics["pearson"],
                    float(metrics["mae"]),
                    metrics["interval_accuracy"],
                    dumps(metrics["linear_ramp_baseline"])
                    if metrics.get("linear_ramp_baseline")
                    else None,
                    success.get("predicted_probability"),
                    _optional_bool(success.get("predicted")),
                    _optional_bool(success.get("gt")),
                    success.get("gt_frame"),
                ),
            )
        _insert_batched(conn, INSERT_FRAME_SQL, _frame_rows(run_id, dataset_id, episodes))
        _insert_batched(
            conn, INSERT_INTERVAL_SQL, _interval_rows(run_id, dataset_id, delta_frames, episodes)
        )
    return run_id


def run_row(row: Any) -> dict[str, Any]:
    item = dict(row)
    item["git_dirty"] = bool(item["git_dirty"])
    item["dataset"] = loads(item.pop("dataset_json", None), {})
    item["aggregate_metrics"] = loads(item.pop("aggregate_metrics_json", None), {})
    return item


def episode_row(row: Any) -> dict[str, Any]:
    """Reassemble one stored episode into the artifact's own block structure."""
    item = dict(row)
    item.pop("id", None)
    item["metrics"] = {
        "frames": item.pop("frames"),
        "intervals": item.pop("intervals"),
        "spearman": item.pop("spearman"),
        "pearson": item.pop("pearson"),
        "mae": item.pop("mae"),
        "interval_accuracy": item.pop("interval_accuracy"),
        "linear_ramp_baseline": loads(item.pop("linear_ramp_baseline_json", None), None),
    }
    probability = item.pop("success_predicted_probability")
    predicted = item.pop("success_predicted")
    ground_truth = item.pop("success_gt")
    gt_frame = item.pop("success_gt_frame")
    item["success"] = (
        None
        if probability is None
        else {
            "predicted_probability": probability,
            "predicted": bool(predicted),
            "gt": None if ground_truth is None else bool(ground_truth),
            "gt_frame": gt_frame,
        }
    )
    return item


def get_prediction_run(run_id: int) -> dict[str, Any] | None:
    with connect(read_only=True) as conn:
        row = conn.execute(f"{SELECT_RUN_SQL} WHERE id=?", (run_id,)).fetchone()
    return run_row(row) if row else None


def list_prediction_runs(dataset_id: int) -> list[dict[str, Any]]:
    with connect(read_only=True) as conn:
        rows = conn.execute(
            f"{SELECT_RUN_SQL} WHERE dataset_id=? ORDER BY id DESC", (dataset_id,)
        ).fetchall()
    return [run_row(row) for row in rows]


def list_prediction_episodes(run_id: int) -> list[dict[str, Any]]:
    with connect(read_only=True) as conn:
        rows = conn.execute(
            "SELECT * FROM prediction_episode WHERE run_id=? ORDER BY episode_index", (run_id,)
        ).fetchall()
    return [episode_row(row) for row in rows]


def episode_series(run_id: int, episode_index: int) -> dict[str, Any] | None:
    """Return one episode's dense curves, interval labels, and metrics."""
    with connect(read_only=True) as conn:
        row = conn.execute(
            "SELECT * FROM prediction_episode WHERE run_id=? AND episode_index=?",
            (run_id, episode_index),
        ).fetchone()
        if row is None:
            return None
        frames = conn.execute(
            "SELECT predicted_progress,gt_progress FROM prediction_frame"
            " WHERE run_id=? AND episode_index=? ORDER BY frame_index",
            (run_id, episode_index),
        ).fetchall()
        intervals = conn.execute(
            "SELECT target_frame,start_frame,delta_frames,predicted_label,gt_label,"
            "predicted_probabilities_json FROM prediction_interval"
            " WHERE run_id=? AND episode_index=? ORDER BY target_frame",
            (run_id, episode_index),
        ).fetchall()
    item = episode_row(row)
    item["predicted_progress"] = [frame["predicted_progress"] for frame in frames]
    item["gt_progress"] = [frame["gt_progress"] for frame in frames]
    item["intervals"] = [
        {
            "target_frame": interval["target_frame"],
            "start_frame": interval["start_frame"],
            "delta_frames": interval["delta_frames"],
            "predicted_label": interval["predicted_label"],
            "gt_label": interval["gt_label"],
            "predicted_probabilities": loads(interval["predicted_probabilities_json"], None),
        }
        for interval in intervals
    ]
    return item


def _disagrees(interval: dict[str, Any]) -> bool:
    return interval["gt_label"] is not None and interval["predicted_label"] != interval["gt_label"]


def _frame_indices(
    predicted: list[float], ground_truth: list[float], max_points: int
) -> list[int]:
    """Pick the frames to keep: bucket boundaries plus both curves' extremes.

    Thinning by a plain stride flattens a peak that falls between two kept
    frames. Keeping each bucket's minimum and maximum of both curves instead
    preserves the envelope, which is what a reader compares.
    """
    total = len(predicted)
    if total <= max_points:
        return list(range(total))
    buckets = max(1, (max_points - 2) // FRAMES_PER_BUCKET)
    keep = {0, total - 1}
    for bucket in range(buckets):
        start = bucket * total // buckets
        stop = (bucket + 1) * total // buckets
        if start >= stop:
            continue
        span = range(start, stop)
        keep.add(start)
        for values in (predicted, ground_truth):
            keep.add(min(span, key=values.__getitem__))
            keep.add(max(span, key=values.__getitem__))
    return sorted(keep)


def _evenly_spaced(positions: list[int], budget: int) -> list[int]:
    if budget <= 0:
        return []
    if budget >= len(positions):
        return positions
    step = len(positions) / budget
    return [positions[int(slot * step)] for slot in range(budget)]


def _thin_intervals(
    intervals: list[dict[str, Any]], max_points: int
) -> tuple[list[dict[str, Any]], int]:
    """Thin the interval strip, never dropping a predicted/human disagreement."""
    if len(intervals) <= max_points:
        return list(intervals), 0
    disagreeing = [position for position, item in enumerate(intervals) if _disagrees(item)]
    agreeing = [position for position, item in enumerate(intervals) if not _disagrees(item)]
    kept_agreeing = _evenly_spaced(agreeing, max_points - len(disagreeing))
    keep = sorted(set(disagreeing) | set(kept_agreeing))
    return [intervals[position] for position in keep], len(agreeing) - len(kept_agreeing)


def downsample_series(
    series: dict[str, Any], max_points: int = DEFAULT_SERIES_POINTS
) -> dict[str, Any]:
    """Thin one episode's curves and interval strip down to chart size.

    The curves keep their first and last frame and every bucket extreme, so the
    visual shape survives. The strip keeps every window where the predicted and
    human labels differ, so a disagreement is never dropped silently; the
    `sampling` block reports exactly what was thinned.
    """
    predicted = series["predicted_progress"]
    ground_truth = series["gt_progress"]
    intervals = series["intervals"]
    indices = _frame_indices(predicted, ground_truth, max_points)
    kept, dropped = _thin_intervals(intervals, max_points)
    return {
        "frame_indices": indices,
        "predicted_progress": [predicted[index] for index in indices],
        "gt_progress": [ground_truth[index] for index in indices],
        "intervals": kept,
        "sampling": {
            "max_points": max_points,
            "frames": len(predicted),
            "frame_points": len(indices),
            "frames_downsampled": len(indices) < len(predicted),
            "intervals": len(intervals),
            "interval_points": len(kept),
            "intervals_downsampled": len(kept) < len(intervals),
            "interval_disagreements": sum(1 for item in intervals if _disagrees(item)),
            "agreeing_intervals_dropped": dropped,
        },
    }


def delete_prediction_run(run_id: int) -> bool:
    with transaction() as conn:
        deleted = conn.execute("DELETE FROM prediction_run WHERE id=?", (run_id,)).rowcount
    return bool(deleted)
