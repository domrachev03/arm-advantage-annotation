from __future__ import annotations

import json
from pathlib import Path

import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from app import hf_import
from app.hf_import import (
    HFDatasetReference,
    HFIdentifierError,
    LeRobotV3ValidationError,
    download_lerobot_v3,
    parse_hf_dataset_reference,
    validate_lerobot_v3,
)

CAMERA_HIGH = "observation.images.cam_high"
CAMERA_WRIST = "observation.images.wrist"


@pytest.mark.parametrize(
    ("identifier", "expected"),
    [
        (
            "owner/repo",
            HFDatasetReference(repo_id="owner/repo"),
        ),
        (
            "https://huggingface.co/datasets/owner/repo",
            HFDatasetReference(repo_id="owner/repo"),
        ),
        (
            "https://huggingface.co/datasets/owner/repo/tree/v3.0/suite/task",
            HFDatasetReference(
                repo_id="owner/repo",
                revision="v3.0",
                subfolder="suite/task",
            ),
        ),
        (
            "hf://datasets/owner/repo@abc123/suite/task",
            HFDatasetReference(
                repo_id="owner/repo",
                revision="abc123",
                subfolder="suite/task",
            ),
        ),
        (
            "hf://datasets/owner/repo",
            HFDatasetReference(repo_id="owner/repo"),
        ),
    ],
)
def test_parse_hf_dataset_reference(
    identifier: str, expected: HFDatasetReference
) -> None:
    assert parse_hf_dataset_reference(identifier) == expected


@pytest.mark.parametrize(
    "identifier",
    [
        "",
        " owner/repo",
        "owner/repo/extra",
        "owner/../repo",
        "https://example.com/datasets/owner/repo",
        "http://huggingface.co/datasets/owner/repo",
        "https://huggingface.co/models/owner/repo",
        "https://huggingface.co/datasets/owner/repo/blob/main/file",
        "https://huggingface.co/datasets/owner/repo/tree/main/%2e%2e/secret",
        "https://huggingface.co/datasets/owner/repo/tree/main/a%2fb",
        "https://huggingface.co/datasets/owner/repo?download=1",
        "hf://models/owner/repo",
        "hf://datasets/owner/repo@@main/folder",
        "hf://datasets/owner/repo@main/folder/*",
    ],
)
def test_parse_hf_dataset_reference_rejects_unsafe_values(identifier: str) -> None:
    with pytest.raises(HFIdentifierError):
        parse_hf_dataset_reference(identifier)


def test_validate_lerobot_v3_indexes_shared_shards(tmp_path: Path) -> None:
    dataset_root = _write_synthetic_v3(tmp_path)

    manifest = validate_lerobot_v3(
        dataset_root,
        cameras=[CAMERA_HIGH, CAMERA_WRIST],
    )

    assert manifest.info["codebase_version"] == "v3.0"
    assert manifest.camera_keys == (CAMERA_HIGH, CAMERA_WRIST)
    assert manifest.tasks_by_index == {0: "Fold the towel"}
    assert tuple(manifest.episodes_by_index) == (0, 1)

    first = manifest.episodes_by_index[0]
    second = manifest.episodes_by_index[1]
    assert first.data_path == second.data_path
    assert first.task_texts == ("Fold the towel",)
    assert first.videos[CAMERA_HIGH].path == second.videos[CAMERA_HIGH].path
    assert first.videos[CAMERA_HIGH].from_timestamp == 0.0
    assert first.videos[CAMERA_HIGH].to_timestamp == 2.0
    assert second.videos[CAMERA_HIGH].from_timestamp == 2.0
    assert second.videos[CAMERA_HIGH].to_timestamp == 5.0


def test_validate_lerobot_v3_prefers_high_camera(tmp_path: Path) -> None:
    dataset_root = _write_synthetic_v3(tmp_path)

    manifest = validate_lerobot_v3(dataset_root)

    assert manifest.camera_keys == (CAMERA_HIGH,)


def test_metadata_only_validation_does_not_require_payload(tmp_path: Path) -> None:
    dataset_root = _write_synthetic_v3(tmp_path, write_payload=False)

    manifest = validate_lerobot_v3(dataset_root, require_payload=False)

    assert manifest.episodes_by_index[0].data_path.name == "file-000.parquet"
    assert manifest.episodes_by_index[0].videos[CAMERA_HIGH].path.name == "file-000.mp4"


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("version", "major 3"),
        ("missing_video", "missing video shard"),
        ("bad_range", "length does not match"),
        ("missing_data_column", "missing columns"),
    ],
)
def test_validate_lerobot_v3_rejects_invalid_snapshots(
    tmp_path: Path, mutation: str, match: str
) -> None:
    dataset_root = _write_synthetic_v3(tmp_path)

    if mutation == "version":
        info_path = dataset_root / "meta" / "info.json"
        info = json.loads(info_path.read_text())
        info["codebase_version"] = "v2.1"
        info_path.write_text(json.dumps(info))
    elif mutation == "missing_video":
        (dataset_root / "videos" / CAMERA_HIGH / "chunk-000" / "file-000.mp4").unlink()
    elif mutation == "bad_range":
        episode_path = (
            dataset_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
        )
        table = pq.read_table(episode_path).to_pylist()
        table[0]["dataset_to_index"] = 3
        pq.write_table(pa.Table.from_pylist(table), episode_path)
    elif mutation == "missing_data_column":
        data_path = dataset_root / "data" / "chunk-000" / "file-000.parquet"
        table = pq.read_table(data_path)
        pq.write_table(table.drop(["task_index"]), data_path)

    with pytest.raises(LeRobotV3ValidationError, match=match):
        validate_lerobot_v3(dataset_root)


def test_download_lerobot_v3_fetches_metadata_before_selected_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_root = _write_synthetic_v3(tmp_path / "suite" / "task")
    calls: list[dict] = []

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)
        return str(tmp_path)

    monkeypatch.setattr(hf_import, "_snapshot_download", fake_snapshot_download)

    manifest = download_lerobot_v3(
        "https://huggingface.co/datasets/owner/repo/tree/rev123/suite/task",
        cameras=[CAMERA_WRIST],
        cache_dir=tmp_path / "cache",
    )

    assert manifest.root == dataset_root
    assert manifest.source == HFDatasetReference(
        repo_id="owner/repo",
        revision="rev123",
        subfolder="suite/task",
    )
    assert manifest.camera_keys == (CAMERA_WRIST,)
    assert calls[0]["allow_patterns"] == ["suite/task/meta/**"]
    assert calls[1]["allow_patterns"] == [
        "suite/task/meta/**",
        "suite/task/data/**",
        f"suite/task/videos/{CAMERA_WRIST}/**",
    ]
    assert all(call["revision"] == "rev123" for call in calls)
    assert all(call["repo_type"] == "dataset" for call in calls)


def _write_synthetic_v3(root: Path, *, write_payload: bool = True) -> Path:
    meta_root = root / "meta"
    episode_root = meta_root / "episodes" / "chunk-000"
    episode_root.mkdir(parents=True)

    features = {
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
        "observation.state": {"dtype": "float32", "shape": [2], "names": None},
        CAMERA_HIGH: {
            "dtype": "video",
            "shape": [3, 16, 16],
            "names": ["channels", "height", "width"],
        },
        CAMERA_WRIST: {
            "dtype": "video",
            "shape": [3, 16, 16],
            "names": ["channels", "height", "width"],
        },
    }
    info = {
        "codebase_version": "v3.0",
        "robot_type": "test",
        "total_episodes": 2,
        "total_frames": 5,
        "total_tasks": 1,
        "chunks_size": 1000,
        "fps": 1,
        "splits": {"train": "0:2"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": (
            "videos/{video_key}/chunk-{chunk_index:03d}/"
            "file-{file_index:03d}.mp4"
        ),
        "features": features,
    }
    (meta_root / "info.json").write_text(json.dumps(info))
    (meta_root / "stats.json").write_text("{}")
    pq.write_table(
        pa.table(
            {
                "task_index": [0],
                "__index_level_0__": ["Fold the towel"],
            }
        ),
        meta_root / "tasks.parquet",
    )

    records = []
    for episode_index, (start, end) in enumerate(((0, 2), (2, 5))):
        record = {
            "episode_index": episode_index,
            "data/chunk_index": 0,
            "data/file_index": 0,
            "dataset_from_index": start,
            "dataset_to_index": end,
            "tasks": ["Fold the towel"],
            "length": end - start,
            "meta/episodes/chunk_index": 0,
            "meta/episodes/file_index": 0,
        }
        for camera_key in (CAMERA_HIGH, CAMERA_WRIST):
            record[f"videos/{camera_key}/chunk_index"] = 0
            record[f"videos/{camera_key}/file_index"] = 0
            record[f"videos/{camera_key}/from_timestamp"] = float(start)
            record[f"videos/{camera_key}/to_timestamp"] = float(end)
        records.append(record)
    pq.write_table(pa.Table.from_pylist(records), episode_root / "file-000.parquet")

    if write_payload:
        data_path = root / "data" / "chunk-000" / "file-000.parquet"
        data_path.parent.mkdir(parents=True)
        pq.write_table(
            pa.table(
                {
                    "timestamp": pa.array([0, 1, 0, 1, 2], type=pa.float32()),
                    "frame_index": [0, 1, 0, 1, 2],
                    "episode_index": [0, 0, 1, 1, 1],
                    "index": [0, 1, 2, 3, 4],
                    "task_index": [0, 0, 0, 0, 0],
                    "observation.state": [
                        [0.0, 0.0],
                        [0.0, 0.0],
                        [0.0, 0.0],
                        [0.0, 0.0],
                        [0.0, 0.0],
                    ],
                }
            ),
            data_path,
        )
        for camera_key in (CAMERA_HIGH, CAMERA_WRIST):
            video_path = root / "videos" / camera_key / "chunk-000" / "file-000.mp4"
            video_path.parent.mkdir(parents=True)
            video_path.write_bytes(b"synthetic-video")

    return root
