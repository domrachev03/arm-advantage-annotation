from __future__ import annotations

import copy
import gzip
import json
from pathlib import Path
from typing import Any

from app.db import connect, dumps, now, transaction

FIXTURES = Path(__file__).parent / "fixtures"
EXAMPLE_LENGTHS = {0: 13, 1: 11, 2: 9}


def load_artifact() -> dict[str, Any]:
    text = (FIXTURES / "arm_predictions_example.json").read_text(encoding="utf-8")
    return json.loads(text)


def seed_dataset(
    lengths: dict[int, int],
    *,
    fps: float = 30.0,
    delta: int = 2,
    status: str = "ready",
) -> int:
    """Register the dataset the example artifact binds to."""
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
                status,
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


def upload(client, artifact: dict[str, Any] | bytes | str):
    payload = artifact if isinstance(artifact, bytes | str) else json.dumps(artifact)
    return client.post("/api/predictions", content=payload)


def build_long_artifact(
    *,
    length: int = 1000,
    delta: int = 2,
    peak: int = 417,
    trough: int = 604,
    disagreements: tuple[int, ...] = (12, 500, 994),
) -> dict[str, Any]:
    """One long episode with a spiky prediction and known label disagreements."""
    artifact = load_artifact()
    ramp = [round(frame / (length - 1), 6) for frame in range(length)]
    predicted = list(ramp)
    predicted[peak] = 1.0
    predicted[trough] = 0.0
    intervals = [
        {
            "target_frame": target,
            "window_frames": [target - step * delta for step in (4, 3, 2, 1, 0)],
            "predicted_label": -1 if target in disagreements else 1,
            "gt_label": 1,
        }
        for target in range(4 * delta, length, delta)
    ]
    artifact["dataset"]["total_episodes"] = 1
    artifact["dataset"]["total_frames"] = length
    artifact["episodes"] = [
        {
            "episode_index": 0,
            "split": "train",
            "length": length,
            "predicted_progress": predicted,
            "gt_progress": ramp,
            "intervals": intervals,
            "success": {
                "predicted_probability": 0.91,
                "predicted": True,
                "gt": True,
                "gt_frame": length - 1,
            },
            "metrics": {
                "frames": length,
                "intervals": len(intervals),
                "spearman": 0.99,
                "pearson": 0.98,
                "mae": 0.002,
                "interval_accuracy": 1 - len(disagreements) / len(intervals),
            },
        }
    ]
    overall = artifact["aggregate_metrics"]["overall"]
    overall["episodes"] = 1
    overall["frames"] = length
    overall["intervals"] = len(intervals)
    artifact["aggregate_metrics"]["by_split"] = {"train": copy.deepcopy(overall)}
    return artifact


def test_upload_stores_a_run_and_exposes_it(client) -> None:
    artifact = load_artifact()
    dataset_id = seed_dataset(EXAMPLE_LENGTHS)

    created = upload(client, artifact)

    assert created.status_code == 201
    run = created.json()
    run_id = run["id"]
    assert run["dataset_id"] == dataset_id
    assert run["name"] == artifact["run"]["name"]
    assert run["episode_count"] == 3
    assert run["uploaded_by"] == "tester"
    assert run["aggregate_metrics"] == artifact["aggregate_metrics"]

    listed = client.get(f"/api/datasets/{dataset_id}/predictions")
    assert listed.status_code == 200
    assert [item["id"] for item in listed.json()] == [run_id]
    assert listed.json()[0]["notes"] == artifact["run"]["notes"]

    detail = client.get(f"/api/predictions/{run_id}")
    assert detail.status_code == 200
    episodes = detail.json()["episodes"]
    assert [item["episode_index"] for item in episodes] == [0, 1, 2]
    assert [item["split"] for item in episodes] == ["train", "val", "test"]
    assert episodes[0]["metrics"] == artifact["episodes"][0]["metrics"]
    assert episodes[2]["success"] == artifact["episodes"][2]["success"]


def test_series_returns_a_short_episode_whole(client) -> None:
    artifact = load_artifact()
    seed_dataset(EXAMPLE_LENGTHS)
    run_id = upload(client, artifact).json()["id"]
    expected = artifact["episodes"][1]

    response = client.get(f"/api/predictions/{run_id}/episodes/1")

    assert response.status_code == 200
    series = response.json()
    assert series["split"] == "val"
    assert series["length"] == 11
    assert series["delta_frames"] == 2
    assert series["fps"] == 30.0
    assert series["frame_indices"] == list(range(11))
    assert series["predicted_progress"] == expected["predicted_progress"]
    assert series["gt_progress"] == expected["gt_progress"]
    assert series["metrics"] == expected["metrics"]
    assert series["success"] == expected["success"]
    assert [item["gt_label"] for item in series["intervals"]] == [-1, 1]
    assert series["sampling"]["frames_downsampled"] is False
    assert series["sampling"]["intervals_downsampled"] is False
    assert series["sampling"]["interval_disagreements"] == 1


def test_series_downsamples_a_long_episode_but_keeps_its_shape(client) -> None:
    artifact = build_long_artifact()
    seed_dataset({0: 1000})
    run_id = upload(client, artifact).json()["id"]

    response = client.get(f"/api/predictions/{run_id}/episodes/0?max_points=200")

    assert response.status_code == 200
    series = response.json()
    sampling = series["sampling"]
    assert sampling["frames"] == 1000
    assert sampling["frames_downsampled"] is True
    assert sampling["frame_points"] <= 200
    assert len(series["predicted_progress"]) == sampling["frame_points"]
    indices = series["frame_indices"]
    assert indices == sorted(set(indices))
    assert indices[0] == 0
    assert indices[-1] == 999
    # The spike and the dip survive: a plain stride would have flattened both.
    assert 417 in indices
    assert 604 in indices
    assert max(series["predicted_progress"]) == 1.0
    assert min(series["predicted_progress"]) == 0.0


def test_series_never_drops_an_interval_disagreement(client) -> None:
    artifact = build_long_artifact(disagreements=(12, 500, 994))
    seed_dataset({0: 1000})
    run_id = upload(client, artifact).json()["id"]

    response = client.get(f"/api/predictions/{run_id}/episodes/0?max_points=60")

    series = response.json()
    sampling = series["sampling"]
    assert sampling["intervals"] == 496
    assert sampling["intervals_downsampled"] is True
    assert sampling["interval_points"] <= 60
    assert sampling["interval_disagreements"] == 3
    assert sampling["agreeing_intervals_dropped"] == 496 - 3 - (sampling["interval_points"] - 3)
    kept = {item["target_frame"] for item in series["intervals"]}
    assert {12, 500, 994} <= kept
    assert all(item["delta_frames"] == 2 for item in series["intervals"])


def test_upload_and_delete_require_a_session(client) -> None:
    artifact = load_artifact()
    seed_dataset(EXAMPLE_LENGTHS)
    run_id = upload(client, artifact).json()["id"]
    client.cookies.clear()

    assert upload(client, artifact).status_code == 401
    assert client.get("/api/predictions/1").status_code == 401
    assert client.delete(f"/api/predictions/{run_id}").status_code == 401
    with connect(read_only=True) as conn:
        assert conn.execute("SELECT COUNT(*) AS n FROM prediction_run").fetchone()["n"] == 1


def test_upload_accepts_a_gzipped_artifact(client) -> None:
    artifact = load_artifact()
    seed_dataset(EXAMPLE_LENGTHS)

    response = upload(client, gzip.compress(json.dumps(artifact).encode("utf-8")))

    assert response.status_code == 201
    assert response.json()["episode_count"] == 3


def test_upload_rejects_an_unregistered_dataset(client) -> None:
    artifact = load_artifact()
    seed_dataset(EXAMPLE_LENGTHS)
    artifact["dataset"]["repo_id"] = "orel-lab/other_task"

    response = upload(client, artifact)

    assert response.status_code == 404
    assert "no dataset is registered" in response.json()["detail"]
    assert "orel-lab/other_task" in response.json()["detail"]


def test_upload_rejects_a_dataset_that_is_still_importing(client) -> None:
    artifact = load_artifact()
    seed_dataset(EXAMPLE_LENGTHS, status="importing")

    response = upload(client, artifact)

    assert response.status_code == 409
    assert "not ready" in response.json()["detail"]


def test_upload_rejects_mismatched_dataset_identity(client) -> None:
    seed_dataset(EXAMPLE_LENGTHS)
    cases = {
        "fps": ({"fps": 60.0}, "fps"),
        "episodes": ({"total_episodes": 60}, "total_episodes"),
        "frames": ({"total_frames": 58742}, "total_frames"),
        "hint": ({"dataset_id": 999}, "dataset_id"),
    }
    for name, (patch, expected) in cases.items():
        artifact = load_artifact()
        artifact["dataset"].update(patch)
        artifact["run"]["name"] = f"run-{name}"

        response = upload(client, artifact)

        assert response.status_code == 409, name
        assert expected in response.json()["detail"], name


def test_upload_rejects_a_grid_the_dataset_is_not_annotated_on(client) -> None:
    artifact = load_artifact()
    seed_dataset(EXAMPLE_LENGTHS)
    artifact["grid"]["delta_frames"] = 3
    for episode in artifact["episodes"]:
        episode["intervals"] = []
        episode["metrics"]["intervals"] = 0

    response = upload(client, artifact)

    assert response.status_code == 409
    assert "2-frame grid" in response.json()["detail"]


def test_upload_rejects_mismatched_episode_identity(client) -> None:
    # The registered episode 2 is one frame longer than the artifact claims.
    seed_dataset({0: 13, 1: 11, 2: 10})

    unknown = load_artifact()
    unknown["dataset"]["total_frames"] = 34
    unknown["episodes"][2]["episode_index"] = 7
    unknown_response = upload(client, unknown)
    assert unknown_response.status_code == 409
    assert "episode 7 is not registered" in unknown_response.json()["detail"]

    resized = load_artifact()
    resized["dataset"]["total_frames"] = 34
    resized_response = upload(client, resized)
    assert resized_response.status_code == 409
    assert "9 frames in the artifact but 10 frames" in resized_response.json()["detail"]

    with connect(read_only=True) as conn:
        assert conn.execute("SELECT COUNT(*) AS n FROM prediction_run").fetchone()["n"] == 0


def test_upload_rejects_a_duplicate_run_name(client) -> None:
    artifact = load_artifact()
    seed_dataset(EXAMPLE_LENGTHS)
    assert upload(client, artifact).status_code == 201

    response = upload(client, artifact)

    assert response.status_code == 409
    assert "already has a prediction run named" in response.json()["detail"]


def test_upload_rejects_documents_that_break_the_schema(client) -> None:
    seed_dataset(EXAMPLE_LENGTHS)

    def reject(mutate) -> str:
        artifact = load_artifact()
        mutate(artifact)
        response = upload(client, artifact)
        assert response.status_code == 422
        return response.json()["detail"]

    def set_extra_key(artifact: dict[str, Any]) -> None:
        artifact["extra"] = True

    def drop_required_key(artifact: dict[str, Any]) -> None:
        del artifact["run"]["git_sha"]

    def use_a_string_label(artifact: dict[str, Any]) -> None:
        artifact["episodes"][0]["intervals"][0]["predicted_label"] = "1"

    def move_a_window(artifact: dict[str, Any]) -> None:
        artifact["episodes"][0]["intervals"][0]["window_frames"] = [0, 1, 2, 3, 8]

    def shorten_a_curve(artifact: dict[str, Any]) -> None:
        artifact["episodes"][0]["gt_progress"].pop()

    def contradict_the_ground_truth(artifact: dict[str, Any]) -> None:
        artifact["episodes"][0]["intervals"][1]["gt_label"] = -1

    def forget_a_split(artifact: dict[str, Any]) -> None:
        del artifact["aggregate_metrics"]["by_split"]["test"]

    assert "extra" in reject(set_extra_key)
    assert "git_sha" in reject(drop_required_key)
    assert "predicted_label" in reject(use_a_string_label)
    assert "window_frames" in reject(move_a_window)
    assert "gt_progress" in reject(shorten_a_curve)
    assert "disagrees with the ground-truth progress" in reject(contradict_the_ground_truth)
    assert "by_split" in reject(forget_a_split)


def test_upload_rejects_malformed_bodies(client) -> None:
    seed_dataset(EXAMPLE_LENGTHS)
    artifact = load_artifact()
    artifact["episodes"][0]["predicted_progress"][0] = float("nan")
    with_nan = json.dumps(artifact)

    gzipped = gzip.compress(json.dumps(load_artifact()).encode("utf-8"))
    corrupt = gzipped[:10] + bytes(len(gzipped) - 18) + gzipped[-8:]

    assert upload(client, b"").status_code == 422
    assert upload(client, b"[1, 2, 3]").status_code == 422
    assert upload(client, b"{not json").status_code == 422
    assert upload(client, b"\x1f\x8b\x08 truncated").status_code == 422
    assert upload(client, corrupt).status_code == 422
    assert upload(client, b"\xff\xfe not utf-8 \x80").status_code == 422
    nan_response = upload(client, with_nan)
    assert nan_response.status_code == 422
    assert "NaN" in nan_response.json()["detail"]


def test_missing_runs_and_episodes_are_reported_as_missing(client) -> None:
    artifact = load_artifact()
    dataset_id = seed_dataset(EXAMPLE_LENGTHS)
    run_id = upload(client, artifact).json()["id"]

    assert client.get(f"/api/datasets/{dataset_id + 99}/predictions").status_code == 404
    assert client.get(f"/api/predictions/{run_id + 99}").status_code == 404
    assert client.get(f"/api/predictions/{run_id + 99}/episodes/0").status_code == 404
    missing_episode = client.get(f"/api/predictions/{run_id}/episodes/5")
    assert missing_episode.status_code == 404
    assert "does not cover that episode" in missing_episode.json()["detail"]
    assert client.get(f"/api/predictions/{run_id}/episodes/0?max_points=3").status_code == 422


def test_delete_removes_the_run_and_its_series(client) -> None:
    artifact = load_artifact()
    dataset_id = seed_dataset(EXAMPLE_LENGTHS)
    run_id = upload(client, artifact).json()["id"]

    removed = client.delete(f"/api/predictions/{run_id}")

    assert removed.status_code == 200
    assert removed.json() == {"ok": True, "id": run_id}
    assert client.get(f"/api/datasets/{dataset_id}/predictions").json() == []
    assert client.get(f"/api/predictions/{run_id}/episodes/0").status_code == 404
    assert client.delete(f"/api/predictions/{run_id}").status_code == 404
    with connect(read_only=True) as conn:
        frames = conn.execute("SELECT COUNT(*) AS n FROM prediction_frame").fetchone()["n"]
    assert frames == 0
    # The same run name is free again once the run is gone.
    assert upload(client, artifact).status_code == 201
