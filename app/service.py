from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from .annotation import required_targets, sample_targets
from .config import HF_TOKEN, NO_COMPLETION_CEILING
from .db import connect, dumps, loads, now, transaction


def dataset_row(row: Any) -> dict[str, Any]:
    item = dict(row)
    item["camera_keys"] = loads(item.pop("camera_keys_json", "[]"), [])
    item["info"] = loads(item.pop("info_json", None), {})
    return item


def get_dataset(dataset_id: int) -> dict[str, Any] | None:
    with connect(read_only=True) as conn:
        row = conn.execute("SELECT * FROM dataset WHERE id=?", (dataset_id,)).fetchone()
    return dataset_row(row) if row else None


def list_datasets() -> list[dict[str, Any]]:
    with connect(read_only=True) as conn:
        rows = conn.execute("SELECT * FROM dataset ORDER BY id DESC").fetchall()
    return [dataset_row(row) for row in rows]


def get_episodes(dataset_id: int) -> list[dict[str, Any]]:
    with connect(read_only=True) as conn:
        rows = conn.execute(
            "SELECT * FROM episode WHERE dataset_id=? ORDER BY episode_index", (dataset_id,)
        ).fetchall()
    out = []
    for row in rows:
        item = dict(row)
        item["cameras"] = loads(item.pop("cameras_json", "{}"), {})
        out.append(item)
    return out


def coverage(dataset_id: int) -> dict[str, Any]:
    dataset = get_dataset(dataset_id)
    if not dataset:
        return {
            "total_samples": 0,
            "labeled_samples": 0,
            "completed_episodes": 0,
            "total_episodes": 0,
            "export_ready": False,
        }
    episodes = get_episodes(dataset_id)
    with connect(read_only=True) as conn:
        annotations = conn.execute(
            "SELECT episode_index,target_frame FROM annotation"
            " WHERE dataset_id=? AND delta_frames=?",
            (dataset_id, dataset["delta_frames"] or -1),
        ).fetchall()
        completions = conn.execute(
            "SELECT * FROM completion WHERE dataset_id=?", (dataset_id,)
        ).fetchall()
        curve_rows = conn.execute(
            "SELECT episode_index,count(*) AS points FROM progress_keypoint"
            " WHERE dataset_id=? GROUP BY episode_index",
            (dataset_id,),
        ).fetchall()
    labels_by_episode: dict[int, set[int]] = {}
    for row in annotations:
        labels_by_episode.setdefault(int(row["episode_index"]), set()).add(int(row["target_frame"]))
    completion_by_episode = {int(row["episode_index"]): dict(row) for row in completions}

    total = labeled = completed = 0
    for episode in episodes:
        ep_idx = int(episode["episode_index"])
        completion = completion_by_episode.get(ep_idx)
        required = required_targets(
            int(episode["length"]), int(dataset["delta_frames"] or 1), completion
        )
        have = labels_by_episode.get(ep_idx, set())
        total += len(required)
        labeled += sum(target in have for target in required)
        if completion and all(target in have for target in required):
            completed += 1
    total_episodes = len(episodes)
    curve_completed = sum(int(row["points"]) >= 2 for row in curve_rows)
    direct_ready = dataset["status"] == "ready" and total_episodes > 0 and completed == total_episodes
    curve_ready = dataset["status"] == "ready" and total_episodes > 0 and curve_completed == total_episodes
    curve_percent = round(100 * curve_completed / total_episodes, 1) if total_episodes else 0.0
    return {
        "total_samples": total,
        "labeled_samples": labeled,
        "completed_episodes": completed,
        "total_episodes": total_episodes,
        "curve_completed_episodes": curve_completed,
        "direct_export_ready": direct_ready,
        "curve_export_ready": curve_ready,
        "curve_percent": curve_percent,
        "export_ready": curve_ready if dataset.get("annotation_mode") == "curve" else direct_ready,
        "percent": round(100 * labeled / total, 1) if total else (100.0 if completed else 0.0),
    }


def queue_current(dataset_id: int) -> dict[str, Any]:
    dataset = get_dataset(dataset_id)
    if not dataset:
        raise KeyError(dataset_id)
    if dataset["status"] != "ready":
        return {"kind": "waiting", "status": dataset["status"], "error": dataset["error"]}
    episodes = get_episodes(dataset_id)
    delta = int(dataset["delta_frames"])
    with connect(read_only=True) as conn:
        annotations = conn.execute(
            "SELECT episode_index,target_frame FROM annotation"
            " WHERE dataset_id=? AND delta_frames=?",
            (dataset_id, delta),
        ).fetchall()
        completions = conn.execute(
            "SELECT * FROM completion WHERE dataset_id=?", (dataset_id,)
        ).fetchall()
    have: dict[int, set[int]] = {}
    for row in annotations:
        have.setdefault(int(row["episode_index"]), set()).add(int(row["target_frame"]))
    completion_by_episode = {int(row["episode_index"]): dict(row) for row in completions}
    for episode in episodes:
        ep_idx = int(episode["episode_index"])
        completion = completion_by_episode.get(ep_idx)
        for target in required_targets(int(episode["length"]), delta, completion):
            if target not in have.get(ep_idx, set()):
                return {
                    "kind": "sample",
                    "episode_index": ep_idx,
                    "target_frame": target,
                }
        if completion is None:
            targets = sample_targets(int(episode["length"]), delta)
            suggested = targets[-1] if targets else max(0, int(episode["length"]) - 1)
            return {
                "kind": "completion",
                "episode_index": ep_idx,
                "suggested_frame": suggested,
            }
    return {"kind": "done"}


def _import_worker(dataset_id: int, requested_cameras: list[str]) -> None:
    try:
        from .hf_import import HFDatasetReference, download_lerobot_v3

        dataset = get_dataset(dataset_id)
        if dataset is None:
            return
        cache_dir = Path(dataset["root_path"]) / "hub"
        reference = HFDatasetReference(
            repo_id=dataset["repo_id"],
            revision=dataset["revision"] or None,
            subfolder=dataset["subpath"] or None,
        )
        result = download_lerobot_v3(
            reference,
            cache_dir=cache_dir,
            cameras=requested_cameras or None,
            token=HF_TOKEN,
        )
        _persist_import(dataset_id, dataset, result)
    # A background job must persist every operational failure for the polling UI.
    except Exception as exc:  # noqa: BLE001
        _fail_import(dataset_id, exc)


def _local_import_worker(dataset_id: int, requested_cameras: list[str], archive: Path) -> None:
    try:
        from .hf_import import validate_lerobot_v3
        from .local_import import extract_dataset_zip

        dataset = get_dataset(dataset_id)
        if dataset is None:
            return
        extracted_root = archive.with_suffix("")
        root = extract_dataset_zip(archive, extracted_root)
        result = validate_lerobot_v3(root, cameras=requested_cameras or None)
        _persist_import(dataset_id, dataset, result)
    except Exception as exc:  # noqa: BLE001
        _fail_import(dataset_id, exc)


def _persist_import(dataset_id: int, dataset: dict[str, Any], result: Any) -> None:
    episodes = result.episodes
    info = dict(result.info)
    fps = float(info["fps"])
    camera_keys = list(result.camera_keys)
    with transaction() as conn:
        conn.execute("DELETE FROM episode WHERE dataset_id=?", (dataset_id,))
        for episode in episodes:
            cameras = {}
            for key, video in episode.videos.items():
                cameras[key] = {
                    "path": str(video.path.relative_to(result.root)),
                    "chunk_index": video.chunk_index,
                    "file_index": video.file_index,
                    "from_timestamp": video.from_timestamp,
                    "to_timestamp": video.to_timestamp,
                }
            conn.execute(
                "INSERT INTO episode(dataset_id,episode_index,length,task,data_from_index,"
                "data_to_index,data_path,cameras_json) VALUES(?,?,?,?,?,?,?,?)",
                (
                    dataset_id,
                    episode.episode_index,
                    episode.length,
                    " · ".join(episode.task_texts),
                    episode.dataset_from_index,
                    episode.dataset_to_index,
                    str(episode.data_path.relative_to(result.root)),
                    dumps(cameras),
                ),
            )
        delta_frames = max(1, round(fps * float(dataset["delta_seconds"])))
        conn.execute(
            "UPDATE dataset SET root_path=?,status='ready',error=NULL,fps=?,delta_frames=?,"
            "camera_keys_json=?,info_json=?,total_episodes=?,total_frames=?,updated_at=?"
            " WHERE id=?",
            (
                str(result.root),
                fps,
                delta_frames,
                dumps(camera_keys),
                dumps(info),
                len(episodes),
                sum(episode.length for episode in episodes),
                now(),
                dataset_id,
            ),
        )


def _fail_import(dataset_id: int, exc: Exception) -> None:
    with transaction() as conn:
        conn.execute(
            "UPDATE dataset SET status='failed',error=?,updated_at=? WHERE id=?",
            (f"{type(exc).__name__}: {exc}", now(), dataset_id),
        )


def start_import(dataset_id: int, requested_cameras: list[str]) -> None:
    thread = threading.Thread(
        target=_import_worker,
        args=(dataset_id, requested_cameras),
        name=f"arm-import-{dataset_id}",
        daemon=True,
    )
    thread.start()


def start_local_import(dataset_id: int, requested_cameras: list[str], archive: Path) -> None:
    thread = threading.Thread(
        target=_local_import_worker,
        args=(dataset_id, requested_cameras, archive),
        name=f"arm-local-import-{dataset_id}",
        daemon=True,
    )
    thread.start()


def _export_worker(export_id: int) -> None:
    try:
        from .exporter import export_fluxvla_dataset

        with transaction() as conn:
            job = conn.execute("SELECT * FROM export_job WHERE id=?", (export_id,)).fetchone()
            if not job:
                return
            conn.execute(
                "UPDATE export_job SET status='running',updated_at=? WHERE id=?",
                (now(), export_id),
            )
        dataset = get_dataset(int(job["dataset_id"]))
        if not dataset:
            raise RuntimeError("dataset was deleted")
        episodes = get_episodes(dataset["id"])
        with connect(read_only=True) as conn:
            labels = []
            for row in conn.execute(
                    "SELECT episode_index,start_frame,target_frame,delta_frames,label,annotator,"
                    "revision,created_at,updated_at FROM annotation WHERE dataset_id=?"
                    " ORDER BY episode_index,target_frame",
                    (dataset["id"],),
                ):
                item = dict(row)
                # The database's start_frame is the labeled pair's first state (t-Δ).
                # Exporter's start_frame records the full causal window start (t-4Δ).
                item["start_frame"] = int(item["target_frame"]) - 4 * int(item["delta_frames"])
                labels.append(item)
            completions = [
                dict(row)
                for row in conn.execute(
                    "SELECT episode_index,state,frame,annotator,updated_at FROM completion"
                    " WHERE dataset_id=? ORDER BY episode_index",
                    (dataset["id"],),
                )
            ]
            progress_keypoints = [
                dict(row)
                for row in conn.execute(
                    "SELECT episode_index,frame,value,annotator,updated_at FROM progress_keypoint"
                    " WHERE dataset_id=? ORDER BY episode_index,frame",
                    (dataset["id"],),
                )
            ]
        result = export_fluxvla_dataset(
            source_root=Path(dataset["root_path"]),
            output_root=Path(job["output_path"]),
            episodes=episodes,
            annotations=labels,
            completions=completions,
            delta_frames=int(dataset["delta_frames"]),
            fps=float(dataset["fps"]),
            video_mode=str(job["video_mode"]),
            no_completion_ceiling=NO_COMPLETION_CEILING,
            progress_keypoints=(
                progress_keypoints if dataset.get("annotation_mode") == "curve" else None
            ),
        )
        manifest = result if isinstance(result, dict) else vars(result)
        with transaction() as conn:
            conn.execute(
                "UPDATE export_job SET status='ready',manifest_json=?,error=NULL,updated_at=?"
                " WHERE id=?",
                (dumps(manifest), now(), export_id),
            )
    # A background job must persist every operational failure for the polling UI.
    except Exception as exc:  # noqa: BLE001
        with transaction() as conn:
            conn.execute(
                "UPDATE export_job SET status='failed',error=?,updated_at=? WHERE id=?",
                (f"{type(exc).__name__}: {exc}", now(), export_id),
            )


def start_export(export_id: int) -> None:
    thread = threading.Thread(
        target=_export_worker,
        args=(export_id,),
        name=f"arm-export-{export_id}",
        daemon=True,
    )
    thread.start()


def clean_interrupted_jobs() -> None:
    with transaction() as conn:
        conn.execute(
            "UPDATE dataset SET status='failed',error='import interrupted; retry it',updated_at=?"
            " WHERE status='importing'",
            (now(),),
        )
        conn.execute(
            "UPDATE export_job SET status='failed',error='export interrupted; start it again',"
            "updated_at=? WHERE status IN ('queued','running')",
            (now(),),
        )
