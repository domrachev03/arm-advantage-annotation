from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from app.exporter import (
    CompletionState,
    EpisodeMetadata,
    ExportError,
    PairLabel,
    deadband_label,
    export_dataset,
    export_fluxvla_dataset,
)


def _write_source(root: Path) -> None:
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    (root / "videos" / "chunk-000").mkdir(parents=True)
    info = {
        "codebase_version": "v3.0",
        "fps": 10,
        "total_episodes": 2,
        "total_frames": 62,
        "features": {
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        },
    }
    (root / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")
    pq.write_table(
        pa.table(
            {
                "episode_index": pa.array([0, 1], type=pa.int64()),
                "length": pa.array([31, 31], type=pa.int64()),
            }
        ),
        root / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
    )
    for episode_index in range(2):
        pq.write_table(
            pa.table(
                {
                    "episode_index": pa.array([episode_index] * 31, type=pa.int64()),
                    "frame_index": pa.array(range(31), type=pa.int64()),
                    "timestamp": pa.array([frame / 10 for frame in range(31)], type=pa.float32()),
                }
            ),
            root / "data" / "chunk-000" / f"file-{episode_index:03d}.parquet",
        )
    (root / "videos" / "chunk-000" / "camera.mp4").write_bytes(b"synthetic-video")


def _inputs() -> tuple[list[EpisodeMetadata], list[PairLabel], list[CompletionState]]:
    episodes = [
        EpisodeMetadata(0, length=31, delta_frames=5, fps=10, uid="ep-0"),
        EpisodeMetadata(1, length=31, delta_frames=5, fps=10, uid="ep-1"),
    ]
    labels = [
        PairLabel(0, 20, 5, 1, "alice", start_frame=0, annotation_id=1),
        PairLabel(0, 25, 5, -1, "alice", start_frame=5, annotation_id=2),
        PairLabel(0, 30, 5, 1, "alice", start_frame=10, annotation_id=3),
        PairLabel(1, 20, 5, 1, "bob", start_frame=0, annotation_id=4),
        PairLabel(1, 25, 5, 1, "bob", start_frame=5, annotation_id=5),
    ]
    completions = [
        CompletionState(0, "never", None, "alice"),
        CompletionState(1, "marked", 25, "bob"),
    ]
    return episodes, labels, completions


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
    }


def _progress_by_episode(root: Path) -> dict[int, list[float]]:
    output: dict[int, list[float]] = {}
    for path in sorted((root / "data").rglob("*.parquet")):
        table = pq.read_table(path)
        assert table.schema.field("progress").type == pa.float32()
        for episode, frame, progress in zip(
            table["episode_index"].to_pylist(),
            table["frame_index"].to_pylist(),
            table["progress"].to_pylist(),
            strict=True,
        ):
            values = output.setdefault(int(episode), [])
            assert int(frame) == len(values)
            values.append(float(progress))
    return output


def test_export_is_non_mutating_and_fluxvla_compatible(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "export"
    _write_source(source)
    before = _tree_hashes(source)
    episodes, labels, completions = _inputs()

    result = export_dataset(
        source,
        output,
        episodes,
        labels,
        completions,
        video_mode="symlink",
    )

    assert result.root == output
    assert (output / "videos").is_symlink()
    assert (output / "videos" / "chunk-000" / "camera.mp4").read_bytes() == b"synthetic-video"
    assert _tree_hashes(source) == before
    assert "progress" not in pq.read_table(
        source / "data" / "chunk-000" / "file-000.parquet"
    ).column_names

    info = json.loads((output / "meta" / "info.json").read_text())
    assert info["features"]["progress"] == {
        "dtype": "float32",
        "shape": [1],
        "names": None,
    }
    progress = _progress_by_episode(output)
    for label in labels:
        values = progress[label.episode_index]
        delta = values[label.target_frame] - values[label.target_frame - label.delta_frames]
        assert deadband_label(delta) == label.label
    assert max(progress[0]) == pytest.approx(0.95)
    assert all(value < 0.999 for value in progress[0])
    assert progress[1][25:] == [1.0] * 6

    pairs = pq.read_table(output / "meta" / "arm_pairs.parquet")
    assert pairs.num_rows == 5
    assert pairs["label"].to_pylist() == [1, -1, 1, 1, 1]
    assert pairs["annotator"].to_pylist() == ["alice", "alice", "alice", "bob", "bob"]
    assert pairs["frame_t"].to_pylist() == [15, 20, 25, 15, 20]
    manifest = json.loads((output / "meta" / "arm_export_manifest.json").read_text())
    assert manifest["lerobot_codebase_version"] == "v3.0"
    assert manifest["pair_deadband"] == 1e-3
    assert manifest["totals"] == {
        "episodes": 2,
        "frames": 62,
        "pairs": 5,
        "marked_completions": 1,
        "never_completions": 1,
        "pair_labels": {"-1": 1, "0": 0, "1": 4},
    }


def test_copy_and_none_video_modes(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_source(source)
    episodes, labels, completions = _inputs()

    copy_root = tmp_path / "copy"
    export_dataset(source, copy_root, episodes, labels, completions, video_mode="copy")
    copied_video = copy_root / "videos" / "chunk-000" / "camera.mp4"
    assert copied_video.is_file()
    assert not (copy_root / "videos").is_symlink()

    none_root = tmp_path / "none"
    export_dataset(source, none_root, episodes, labels, completions, video_mode="none")
    assert not (none_root / "videos").exists()


def test_service_adapter_accepts_db_shaped_records(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "export"
    _write_source(source)
    episodes, labels, completions = _inputs()

    manifest = export_fluxvla_dataset(
        source_root=source,
        output_root=output,
        episodes=[
            {"episode_index": item.episode_index, "length": item.length, "id": index + 10}
            for index, item in enumerate(episodes)
        ],
        annotations=[
            {
                "episode_index": item.episode_index,
                "target_frame": item.target_frame,
                "delta_frames": item.delta_frames,
                "label": item.label,
                "annotator": item.annotator,
                "start_frame": item.start_frame,
                "id": item.annotation_id,
                "revision": item.revision,
            }
            for item in labels
        ],
        completions=[
            {
                "episode_index": item.episode_index,
                "state": item.state,
                "frame": item.frame,
                "annotator": item.annotator,
            }
            for item in completions
        ],
        delta_frames=5,
        fps=10,
        video_mode="none",
    )

    assert manifest["out_root"] == str(output)
    assert manifest["totals"]["pairs"] == 5
    assert (output / "meta" / "arm_export_manifest.json").is_file()


def test_incomplete_grid_is_refused_without_partial_output(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "export"
    _write_source(source)
    episodes, labels, completions = _inputs()

    with pytest.raises(ExportError, match="missing target frames"):
        export_dataset(source, output, episodes, labels[:-1], completions)

    assert not output.exists()
    assert not list(tmp_path.glob(".export.tmp-*"))


def test_completion_curve_must_preserve_direct_label_signs(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "export"
    _write_source(source)
    episodes, labels, completions = _inputs()
    incompatible = [
        PairLabel(0, 20, 5, 1, "alice", start_frame=0),
        PairLabel(0, 25, 5, -1, "alice", start_frame=5),
    ]
    labels = incompatible + [label for label in labels if label.episode_index == 1]
    completions[0] = CompletionState(0, "marked", 25, "alice")

    with pytest.raises(ExportError, match="round-trips"):
        export_dataset(source, output, episodes, labels, completions)

    assert not output.exists()
