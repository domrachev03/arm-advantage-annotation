from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import DB_PATH, ensure_state_dirs

SCHEMA = """
PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS dataset (
    id INTEGER PRIMARY KEY,
    source_url TEXT NOT NULL,
    repo_id TEXT NOT NULL,
    revision TEXT NOT NULL,
    subpath TEXT NOT NULL DEFAULT '',
    title TEXT NOT NULL,
    root_path TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('importing','ready','failed')),
    error TEXT,
    fps REAL,
    delta_seconds REAL NOT NULL,
    delta_frames INTEGER,
    window_size INTEGER NOT NULL DEFAULT 5 CHECK(window_size = 5),
    camera_keys_json TEXT NOT NULL DEFAULT '[]',
    info_json TEXT,
    total_episodes INTEGER NOT NULL DEFAULT 0,
    total_frames INTEGER NOT NULL DEFAULT 0,
    created_by TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS episode (
    id INTEGER PRIMARY KEY,
    dataset_id INTEGER NOT NULL REFERENCES dataset(id) ON DELETE CASCADE,
    episode_index INTEGER NOT NULL,
    length INTEGER NOT NULL CHECK(length >= 0),
    task TEXT NOT NULL DEFAULT '',
    data_from_index INTEGER,
    data_to_index INTEGER,
    data_path TEXT,
    cameras_json TEXT NOT NULL DEFAULT '{}',
    UNIQUE(dataset_id, episode_index)
);

CREATE TABLE IF NOT EXISTS annotation (
    id INTEGER PRIMARY KEY,
    dataset_id INTEGER NOT NULL REFERENCES dataset(id) ON DELETE CASCADE,
    episode_index INTEGER NOT NULL,
    start_frame INTEGER NOT NULL,
    target_frame INTEGER NOT NULL,
    delta_frames INTEGER NOT NULL,
    label INTEGER NOT NULL CHECK(label IN (-1,0,1)),
    annotator TEXT NOT NULL,
    revision INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(dataset_id, episode_index, target_frame, delta_frames)
);

CREATE TABLE IF NOT EXISTS annotation_history (
    id INTEGER PRIMARY KEY,
    annotation_id INTEGER NOT NULL,
    dataset_id INTEGER NOT NULL,
    episode_index INTEGER NOT NULL,
    start_frame INTEGER NOT NULL,
    target_frame INTEGER NOT NULL,
    delta_frames INTEGER NOT NULL,
    label INTEGER NOT NULL,
    annotator TEXT NOT NULL,
    revision INTEGER NOT NULL,
    recorded_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS completion (
    dataset_id INTEGER NOT NULL REFERENCES dataset(id) ON DELETE CASCADE,
    episode_index INTEGER NOT NULL,
    state TEXT NOT NULL CHECK(state IN ('marked','never')),
    frame INTEGER,
    annotator TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY(dataset_id, episode_index),
    CHECK((state = 'marked' AND frame IS NOT NULL) OR (state = 'never' AND frame IS NULL))
);

CREATE TABLE IF NOT EXISTS export_job (
    id INTEGER PRIMARY KEY,
    dataset_id INTEGER NOT NULL REFERENCES dataset(id) ON DELETE CASCADE,
    status TEXT NOT NULL CHECK(status IN ('queued','running','ready','failed')),
    output_path TEXT NOT NULL,
    video_mode TEXT NOT NULL CHECK(video_mode IN ('symlink','copy','none')),
    error TEXT,
    manifest_json TEXT,
    created_by TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS annotation_dataset_episode
ON annotation(dataset_id, episode_index, target_frame);
CREATE INDEX IF NOT EXISTS completion_dataset ON completion(dataset_id, episode_index);

CREATE TABLE IF NOT EXISTS prediction_run (
    id INTEGER PRIMARY KEY,
    dataset_id INTEGER NOT NULL REFERENCES dataset(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    schema_version INTEGER NOT NULL,
    artifact_kind TEXT NOT NULL,
    tool TEXT NOT NULL,
    generated_at TEXT NOT NULL,
    checkpoint_path TEXT NOT NULL,
    checkpoint_sha256 TEXT,
    config_id TEXT NOT NULL,
    config_path TEXT,
    git_sha TEXT NOT NULL,
    git_dirty INTEGER NOT NULL CHECK(git_dirty IN (0,1)),
    training_command TEXT NOT NULL,
    seed INTEGER,
    trained_at TEXT NOT NULL,
    split_file TEXT NOT NULL,
    split_file_sha256 TEXT,
    notes TEXT,
    delta_frames INTEGER NOT NULL CHECK(delta_frames >= 1),
    window_size INTEGER NOT NULL CHECK(window_size = 5),
    interval_eps REAL NOT NULL,
    no_completion_ceiling REAL NOT NULL,
    gt_progress_source TEXT NOT NULL,
    dataset_json TEXT NOT NULL,
    aggregate_metrics_json TEXT NOT NULL,
    artifact_sha256 TEXT NOT NULL,
    artifact_bytes INTEGER NOT NULL,
    uploaded_by TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(dataset_id, name)
);

CREATE TABLE IF NOT EXISTS prediction_episode (
    id INTEGER PRIMARY KEY,
    run_id INTEGER NOT NULL REFERENCES prediction_run(id) ON DELETE CASCADE,
    dataset_id INTEGER NOT NULL,
    episode_index INTEGER NOT NULL,
    split TEXT NOT NULL CHECK(split IN ('train','val','test')),
    length INTEGER NOT NULL CHECK(length >= 0),
    frames INTEGER NOT NULL,
    intervals INTEGER NOT NULL,
    spearman REAL,
    pearson REAL,
    mae REAL NOT NULL,
    interval_accuracy REAL,
    linear_ramp_baseline_json TEXT,
    success_predicted_probability REAL,
    success_predicted INTEGER CHECK(success_predicted IN (0,1)),
    success_gt INTEGER CHECK(success_gt IN (0,1)),
    success_gt_frame INTEGER,
    UNIQUE(run_id, episode_index)
);

CREATE TABLE IF NOT EXISTS prediction_frame (
    run_id INTEGER NOT NULL REFERENCES prediction_run(id) ON DELETE CASCADE,
    dataset_id INTEGER NOT NULL,
    episode_index INTEGER NOT NULL,
    frame_index INTEGER NOT NULL,
    predicted_progress REAL NOT NULL,
    gt_progress REAL NOT NULL,
    PRIMARY KEY(run_id, episode_index, frame_index)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS prediction_interval (
    run_id INTEGER NOT NULL REFERENCES prediction_run(id) ON DELETE CASCADE,
    dataset_id INTEGER NOT NULL,
    episode_index INTEGER NOT NULL,
    target_frame INTEGER NOT NULL,
    start_frame INTEGER NOT NULL,
    delta_frames INTEGER NOT NULL,
    predicted_label INTEGER NOT NULL CHECK(predicted_label IN (-1,0,1)),
    gt_label INTEGER CHECK(gt_label IN (-1,0,1)),
    predicted_probabilities_json TEXT,
    PRIMARY KEY(run_id, episode_index, target_frame)
) WITHOUT ROWID;

CREATE INDEX IF NOT EXISTS prediction_run_dataset ON prediction_run(dataset_id, id);
CREATE INDEX IF NOT EXISTS prediction_episode_dataset
ON prediction_episode(dataset_id, episode_index);
CREATE INDEX IF NOT EXISTS prediction_frame_dataset_episode
ON prediction_frame(dataset_id, episode_index, frame_index);
CREATE INDEX IF NOT EXISTS prediction_interval_dataset_episode
ON prediction_interval(dataset_id, episode_index, target_frame);
"""

# Reverses the prediction-run tables added by `SCHEMA`. Dropping a table also
# drops its indexes, so the four statements below are the whole down path. The
# annotation tables are never touched by it. See docs/model_predictions.md.
PREDICTION_SCHEMA_DOWN = """
DROP TABLE IF EXISTS prediction_interval;
DROP TABLE IF EXISTS prediction_frame;
DROP TABLE IF EXISTS prediction_episode;
DROP TABLE IF EXISTS prediction_run;
"""


def now() -> str:
    return datetime.now(UTC).isoformat()


def dumps(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def loads(value: str | None, default: Any) -> Any:
    return json.loads(value) if value else default


def connect(path: Path | str = DB_PATH, *, read_only: bool = False) -> sqlite3.Connection:
    ensure_state_dirs()
    if read_only:
        uri = f"file:{Path(path)}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=30)
    else:
        conn = sqlite3.connect(path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 30000")
    return conn


def migrate(path: Path | str = DB_PATH) -> None:
    with connect(path) as conn:
        conn.executescript(SCHEMA)


def drop_prediction_tables(path: Path | str = DB_PATH) -> None:
    """Apply the documented down path for the prediction-run tables.

    Every prediction run stored in the database is discarded; annotations,
    completions, datasets, and export jobs are left untouched. `migrate()`
    recreates the tables empty.
    """
    with connect(path) as conn:
        conn.executescript(PREDICTION_SCHEMA_DOWN)


@contextmanager
def transaction(path: Path | str = DB_PATH) -> Iterator[sqlite3.Connection]:
    conn = connect(path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        yield conn
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()


def row_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(row) if row is not None else None
