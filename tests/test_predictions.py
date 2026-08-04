from __future__ import annotations

import copy
import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from app.db import connect, drop_prediction_tables, dumps, migrate, now, transaction
from app.predictions import (
    delete_prediction_run,
    episode_series,
    get_prediction_run,
    insert_prediction_run,
    list_prediction_episodes,
    list_prediction_runs,
)

FIXTURES = Path(__file__).parent / "fixtures"
LEGACY_TABLES = (
    "dataset",
    "episode",
    "annotation",
    "annotation_history",
    "completion",
    "export_job",
)
PREDICTION_TABLES = (
    "prediction_run",
    "prediction_episode",
    "prediction_frame",
    "prediction_interval",
)


def load_artifact() -> dict[str, Any]:
    text = (FIXTURES / "arm_predictions_example.json").read_text(encoding="utf-8")
    return json.loads(text)


def artifact_digest(artifact: dict[str, Any]) -> tuple[str, int]:
    payload = json.dumps(artifact).encode("utf-8")
    return hashlib.sha256(payload).hexdigest(), len(payload)


def store(artifact: dict[str, Any], dataset_id: int) -> int:
    digest, size = artifact_digest(artifact)
    return insert_prediction_run(
        artifact,
        dataset_id=dataset_id,
        uploaded_by="tester",
        artifact_sha256=digest,
        artifact_bytes=size,
    )


def object_sql(path: Path | str) -> dict[str, str]:
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
        rows = conn.execute("SELECT name,sql FROM sqlite_master WHERE sql IS NOT NULL").fetchall()
    return {name: sql for name, sql in rows}


def table_names(path: Path | str) -> set[str]:
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
        rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    return {name for (name,) in rows}


def seed_dataset(lengths: dict[int, int], *, fps: float = 30.0, delta: int = 2) -> int:
    """Register a dataset with the episodes the example artifact covers."""
    stamp = now()
    with transaction() as conn:
        cursor = conn.execute(
            "INSERT INTO dataset(source_url,repo_id,revision,subpath,title,root_path,status,fps,"
            "delta_seconds,delta_frames,camera_keys_json,info_json,total_episodes,total_frames,"
            "created_by,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                "https://huggingface.co/datasets/orel-lab/kuka_assemble",
                "orel-lab/kuka_assemble",
                "main",
                "",
                "kuka assemble",
                "/tmp/kuka",
                "ready",
                fps,
                delta / fps,
                delta,
                dumps(["observation.images.front"]),
                dumps({"codebase_version": "v3.0", "fps": fps}),
                len(lengths),
                sum(lengths.values()),
                "tester",
                stamp,
                stamp,
            ),
        )
        dataset_id = int(cursor.lastrowid)
        for episode_index, length in lengths.items():
            conn.execute(
                "INSERT INTO episode(dataset_id,episode_index,length,task,data_from_index,"
                "data_to_index,data_path,cameras_json) VALUES(?,?,?,?,?,?,?,?)",
                (
                    dataset_id,
                    episode_index,
                    length,
                    "assemble the part",
                    0,
                    length,
                    "data/chunk-000/file-000.parquet",
                    dumps({}),
                ),
            )
    return dataset_id


def write_legacy_database(path: Path) -> None:
    """Create a database with the schema deployed before prediction runs existed."""
    schema = (FIXTURES / "legacy_schema.sql").read_text(encoding="utf-8")
    stamp = "2026-07-30T10:00:00+00:00"
    with sqlite3.connect(path) as conn:
        conn.executescript(schema)
        conn.execute(
            "INSERT INTO dataset(id,source_url,repo_id,revision,subpath,title,root_path,status,"
            "fps,delta_seconds,delta_frames,camera_keys_json,info_json,total_episodes,"
            "total_frames,created_by,created_at,updated_at)"
            " VALUES(1,'owner/repo','owner/repo','main','','legacy','/tmp/legacy','ready',"
            "30.0,1.0,30,'[]',NULL,1,260,'annotator',?,?)",
            (stamp, stamp),
        )
        conn.execute(
            "INSERT INTO episode(id,dataset_id,episode_index,length,task,data_from_index,"
            "data_to_index,data_path,cameras_json)"
            " VALUES(1,1,0,260,'legacy task',0,260,'data/chunk-000/file-000.parquet','{}')",
        )
        conn.execute(
            "INSERT INTO annotation(id,dataset_id,episode_index,start_frame,target_frame,"
            "delta_frames,label,annotator,revision,created_at,updated_at)"
            " VALUES(1,1,0,90,120,30,1,'annotator',2,?,?)",
            (stamp, stamp),
        )
        conn.execute(
            "INSERT INTO annotation_history(id,annotation_id,dataset_id,episode_index,start_frame,"
            "target_frame,delta_frames,label,annotator,revision,recorded_at)"
            " VALUES(1,1,1,0,90,120,30,0,'annotator',1,?)",
            (stamp,),
        )
        conn.execute(
            "INSERT INTO completion(dataset_id,episode_index,state,frame,annotator,updated_at)"
            " VALUES(1,0,'marked',185,'annotator',?)",
            (stamp,),
        )


def test_migration_adds_prediction_tables_and_keeps_annotations(tmp_path: Path) -> None:
    path = tmp_path / "legacy.db"
    write_legacy_database(path)
    before = object_sql(path)
    assert not table_names(path) & set(PREDICTION_TABLES)

    migrate(path)
    # Running the migration twice must be a no-op, as it is on every restart.
    migrate(path)

    after = object_sql(path)
    assert {name: after[name] for name in before} == before
    assert set(PREDICTION_TABLES) <= table_names(path)

    with connect(path, read_only=True) as conn:
        annotation = conn.execute("SELECT * FROM annotation WHERE id=1").fetchone()
        history = conn.execute("SELECT COUNT(*) AS n FROM annotation_history").fetchone()
        completion = conn.execute("SELECT * FROM completion WHERE dataset_id=1").fetchone()
        joined = conn.execute(
            "SELECT dataset.title,episode.length,annotation.label FROM annotation"
            " JOIN episode ON episode.dataset_id=annotation.dataset_id"
            " AND episode.episode_index=annotation.episode_index"
            " JOIN dataset ON dataset.id=annotation.dataset_id"
        ).fetchone()
        empty = {
            table: conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"]
            for table in PREDICTION_TABLES
        }
    assert (annotation["target_frame"], annotation["label"], annotation["revision"]) == (120, 1, 2)
    assert history["n"] == 1
    assert (completion["state"], completion["frame"]) == ("marked", 185)
    assert tuple(joined) == ("legacy", 260, 1)
    assert empty == dict.fromkeys(PREDICTION_TABLES, 0)


def test_prediction_tables_have_a_reversible_down_path(tmp_path: Path) -> None:
    path = tmp_path / "legacy.db"
    write_legacy_database(path)
    migrate(path)
    with connect(path) as conn:
        conn.execute(
            "INSERT INTO prediction_run(id,dataset_id,name,schema_version,artifact_kind,tool,"
            "generated_at,checkpoint_path,checkpoint_sha256,config_id,config_path,git_sha,"
            "git_dirty,training_command,seed,trained_at,split_file,split_file_sha256,notes,"
            "delta_frames,window_size,interval_eps,no_completion_ceiling,gt_progress_source,"
            "dataset_json,aggregate_metrics_json,artifact_sha256,artifact_bytes,uploaded_by,"
            "created_at,updated_at)"
            " VALUES(1,1,'run',1,'arm_prediction_run','tool','t','/ckpt',NULL,'cfg',NULL,'sha',"
            "0,'cmd',NULL,'t','split',NULL,NULL,30,5,0.001,0.95,'manifest','{}','{}','sha',1,"
            "'tester','t','t')"
        )
        conn.execute(
            "INSERT INTO prediction_frame(run_id,dataset_id,episode_index,frame_index,"
            "predicted_progress,gt_progress) VALUES(1,1,0,0,0.0,0.0)"
        )

    drop_prediction_tables(path)

    assert not table_names(path) & set(PREDICTION_TABLES)
    with connect(path, read_only=True) as conn:
        assert conn.execute("SELECT COUNT(*) AS n FROM annotation").fetchone()["n"] == 1
        assert conn.execute("SELECT COUNT(*) AS n FROM completion").fetchone()["n"] == 1

    # The down path leaves a database the current code can migrate again.
    migrate(path)
    with connect(path, read_only=True) as conn:
        assert conn.execute("SELECT COUNT(*) AS n FROM prediction_run").fetchone()["n"] == 0


def test_example_artifact_round_trips_through_storage(database: None) -> None:
    artifact = load_artifact()
    dataset_id = seed_dataset({0: 13, 1: 11, 2: 9})

    run_id = store(artifact, dataset_id)

    run = get_prediction_run(run_id)
    assert run is not None
    assert run["name"] == artifact["run"]["name"]
    assert run["git_dirty"] is False
    assert run["seed"] == 17
    assert run["trained_at"] == artifact["run"]["created_at"]
    assert run["delta_frames"] == 2
    assert run["window_size"] == 5
    assert run["dataset"] == artifact["dataset"]
    assert run["aggregate_metrics"] == artifact["aggregate_metrics"]
    assert [item["id"] for item in list_prediction_runs(dataset_id)] == [run_id]

    episodes = list_prediction_episodes(run_id)
    assert [item["episode_index"] for item in episodes] == [0, 1, 2]
    assert [item["split"] for item in episodes] == ["train", "val", "test"]

    for expected in artifact["episodes"]:
        series = episode_series(run_id, expected["episode_index"])
        assert series is not None
        assert series["predicted_progress"] == expected["predicted_progress"]
        assert series["gt_progress"] == expected["gt_progress"]
        assert series["length"] == expected["length"]
        assert series["mae"] == expected["metrics"]["mae"]
        assert series["spearman"] == expected["metrics"]["spearman"]
        assert series["interval_accuracy"] == expected["metrics"]["interval_accuracy"]
        assert series["linear_ramp_baseline"] == expected["metrics"]["linear_ramp_baseline"]
        assert series["success"] == expected["success"]
        assert [item["target_frame"] for item in series["intervals"]] == [
            item["target_frame"] for item in expected["intervals"]
        ]
        for stored, source in zip(series["intervals"], expected["intervals"], strict=True):
            assert stored["predicted_label"] == source["predicted_label"]
            assert stored["gt_label"] == source["gt_label"]
            assert stored["start_frame"] == source["window_frames"][0]
            assert stored["delta_frames"] == 2


def test_nullable_artifact_fields_survive_storage(database: None) -> None:
    artifact = load_artifact()
    artifact["run"]["notes"] = None
    artifact["run"]["seed"] = None
    artifact["run"]["checkpoint_sha256"] = None
    episode = artifact["episodes"][0]
    episode["success"] = None
    episode["metrics"]["spearman"] = None
    episode["metrics"]["pearson"] = None
    episode["metrics"]["interval_accuracy"] = None
    episode["metrics"].pop("linear_ramp_baseline")
    episode["intervals"][0]["gt_label"] = None
    episode["intervals"][1]["predicted_probabilities"] = [0.05, 0.15, 0.8]
    dataset_id = seed_dataset({0: 13, 1: 11, 2: 9})

    run_id = store(artifact, dataset_id)

    run = get_prediction_run(run_id)
    assert run is not None
    assert (run["notes"], run["seed"], run["checkpoint_sha256"]) == (None, None, None)
    series = episode_series(run_id, 0)
    assert series is not None
    assert series["success"] is None
    assert series["spearman"] is None
    assert series["interval_accuracy"] is None
    assert series["linear_ramp_baseline"] is None
    assert series["intervals"][0]["gt_label"] is None
    assert series["intervals"][0]["predicted_probabilities"] is None
    assert series["intervals"][1]["predicted_probabilities"] == [0.05, 0.15, 0.8]


def test_deleting_a_run_removes_its_rows_only(database: None) -> None:
    artifact = load_artifact()
    dataset_id = seed_dataset({0: 13, 1: 11, 2: 9})
    kept = store(artifact, dataset_id)
    second = copy.deepcopy(artifact)
    second["run"]["name"] = "arm-kuka-assemble-front-r4"
    removed = store(second, dataset_id)

    assert delete_prediction_run(removed) is True
    assert delete_prediction_run(removed) is False

    with connect(read_only=True) as conn:
        counts = {
            table: conn.execute(
                f"SELECT COUNT(*) AS n FROM {table} WHERE run_id=?", (removed,)
            ).fetchone()["n"]
            for table in PREDICTION_TABLES[1:]
        }
        surviving = conn.execute(
            "SELECT COUNT(*) AS n FROM prediction_frame WHERE run_id=?", (kept,)
        ).fetchone()["n"]
    assert counts == dict.fromkeys(PREDICTION_TABLES[1:], 0)
    assert surviving == 33


def build_bulk_artifact(*, episodes: int, length: int, delta: int) -> dict[str, Any]:
    artifact = load_artifact()
    artifact["grid"]["delta_frames"] = delta
    artifact["episodes"] = []
    for episode_index in range(episodes):
        progress = [round(frame / (length - 1), 6) for frame in range(length)]
        intervals = [
            {
                "target_frame": target,
                "window_frames": [target - step * delta for step in (4, 3, 2, 1, 0)],
                "predicted_label": 1,
                "gt_label": 1,
                "predicted_probabilities": [0.01, 0.09, 0.9],
            }
            for target in range(4 * delta, length, delta)
        ]
        artifact["episodes"].append(
            {
                "episode_index": episode_index,
                "split": "train",
                "length": length,
                "predicted_progress": progress,
                "gt_progress": progress,
                "intervals": intervals,
                "success": {
                    "predicted_probability": 0.9,
                    "predicted": True,
                    "gt": True,
                    "gt_frame": length - 1,
                },
                "metrics": {
                    "frames": length,
                    "intervals": len(intervals),
                    "spearman": 1.0,
                    "pearson": 1.0,
                    "mae": 0.0,
                    "interval_accuracy": 1.0,
                },
            }
        )
    return artifact


def test_bulk_insert_of_a_full_run_is_fast_and_indexed(database: None) -> None:
    episodes, length, delta = 60, 1000, 2
    artifact = build_bulk_artifact(episodes=episodes, length=length, delta=delta)
    dataset_id = seed_dataset(dict.fromkeys(range(episodes), length))

    started = time.perf_counter()
    run_id = store(artifact, dataset_id)
    elapsed = time.perf_counter() - started

    with connect(read_only=True) as conn:
        frames = conn.execute(
            "SELECT COUNT(*) AS n FROM prediction_frame WHERE run_id=?", (run_id,)
        ).fetchone()["n"]
        intervals = conn.execute(
            "SELECT COUNT(*) AS n FROM prediction_interval WHERE run_id=?", (run_id,)
        ).fetchone()["n"]
        by_dataset = conn.execute(
            "EXPLAIN QUERY PLAN SELECT predicted_progress FROM prediction_frame"
            " WHERE dataset_id=? AND episode_index=? ORDER BY frame_index",
            (dataset_id, 0),
        ).fetchall()
        by_run = conn.execute(
            "EXPLAIN QUERY PLAN SELECT predicted_progress FROM prediction_frame"
            " WHERE run_id=? AND episode_index=? ORDER BY frame_index",
            (run_id, 0),
        ).fetchall()

    assert frames == episodes * length
    assert intervals == episodes * len(artifact["episodes"][0]["intervals"])
    assert any("prediction_frame_dataset_episode" in row["detail"] for row in by_dataset)
    assert all("SCAN" not in row["detail"] for row in by_run)
    # A full run takes well under a second locally; the generous bound only catches
    # a regression to per-row commits, which takes minutes.
    assert elapsed < 20.0, f"bulk insert of {frames} frames took {elapsed:.1f}s"
