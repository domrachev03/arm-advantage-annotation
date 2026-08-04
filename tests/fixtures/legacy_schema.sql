-- Schema of the deployed database before the prediction-run tables existed.
-- Captured with sqlite_master from a read-only snapshot of the production
-- database at commit 293b060. tests/test_predictions.py migrates a database
-- created from this file to prove that existing annotations survive.

CREATE TABLE annotation (
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

CREATE TABLE annotation_history (
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

CREATE TABLE completion (
    dataset_id INTEGER NOT NULL REFERENCES dataset(id) ON DELETE CASCADE,
    episode_index INTEGER NOT NULL,
    state TEXT NOT NULL CHECK(state IN ('marked','never')),
    frame INTEGER,
    annotator TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY(dataset_id, episode_index),
    CHECK((state = 'marked' AND frame IS NOT NULL) OR (state = 'never' AND frame IS NULL))
);

CREATE TABLE dataset (
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

CREATE TABLE episode (
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

CREATE TABLE export_job (
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

CREATE INDEX annotation_dataset_episode
ON annotation(dataset_id, episode_index, target_frame);

CREATE INDEX completion_dataset ON completion(dataset_id, episode_index);
