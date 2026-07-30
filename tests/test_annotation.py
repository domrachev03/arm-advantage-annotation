from __future__ import annotations

from app.annotation import frame_window, required_targets, sample_targets
from app.db import connect, dumps, now, transaction


def seed_dataset(*, length: int = 260, fps: float = 30.0, delta: int = 30) -> int:
    stamp = now()
    info = {
        "codebase_version": "v3.0",
        "fps": fps,
        "features": {"observation.images.front": {"dtype": "video", "shape": [480, 640, 3]}},
    }
    with transaction() as conn:
        cursor = conn.execute(
            "INSERT INTO dataset(source_url,repo_id,revision,subpath,title,root_path,status,fps,"
            "delta_seconds,delta_frames,camera_keys_json,info_json,total_episodes,total_frames,"
            "created_by,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                "owner/repo",
                "owner/repo",
                "v3.0",
                "",
                "fixture",
                "/tmp/fixture",
                "ready",
                fps,
                delta / fps,
                delta,
                dumps(["observation.images.front"]),
                dumps(info),
                1,
                length,
                "tester",
                stamp,
                stamp,
            ),
        )
        dataset_id = int(cursor.lastrowid)
        conn.execute(
            "INSERT INTO episode(dataset_id,episode_index,length,task,data_from_index,"
            "data_to_index,data_path,cameras_json) VALUES(?,?,?,?,?,?,?,?)",
            (
                dataset_id,
                0,
                length,
                "move the cube",
                0,
                length,
                "data/chunk-000/file-000.parquet",
                dumps(
                    {
                        "observation.images.front": {
                            "path": "videos/observation.images.front/chunk-000/file-000.mp4",
                            "from_timestamp": 0.0,
                            "to_timestamp": length / fps,
                        }
                    }
                ),
            ),
        )
    return dataset_id


def add_episode(dataset_id: int, *, episode_index: int, length: int) -> None:
    with transaction() as conn:
        conn.execute(
            "INSERT INTO episode(dataset_id,episode_index,length,task,data_from_index,"
            "data_to_index,data_path,cameras_json) VALUES(?,?,?,?,?,?,?,?)",
            (
                dataset_id,
                episode_index,
                length,
                f"episode {episode_index}",
                0,
                length,
                "data/chunk-000/file-000.parquet",
                dumps(
                    {
                        "observation.images.front": {
                            "path": "videos/observation.images.front/chunk-000/file-000.mp4",
                            "from_timestamp": 0.0,
                            "to_timestamp": length / 30.0,
                        }
                    }
                ),
            ),
        )
        conn.execute(
            "UPDATE dataset SET total_episodes=total_episodes+1,total_frames=total_frames+?"
            " WHERE id=?",
            (length, dataset_id),
        )


def test_full_causal_window_starts_after_four_deltas() -> None:
    assert sample_targets(260, 30) == [120, 150, 180, 210, 240]
    assert frame_window(120, 30) == [0, 30, 60, 90, 120]
    assert required_targets(260, 30, {"state": "marked", "frame": 185}) == [120, 150, 180]


def test_label_autosaves_and_queue_advances(client) -> None:
    dataset_id = seed_dataset()
    first = client.get(f"/api/datasets/{dataset_id}/queue/current")
    assert first.status_code == 200
    assert first.json()["frame_indices"] == [0, 30, 60, 90, 120]

    saved = client.put(
        f"/api/datasets/{dataset_id}/episodes/0/samples/120/label",
        json={"label": 1},
    )
    assert saved.status_code == 200
    assert saved.json()["next"]["target_frame"] == 150

    relabeled = client.put(
        f"/api/datasets/{dataset_id}/episodes/0/samples/120/label",
        json={"label": -1},
    )
    assert relabeled.json()["revision"] == 2


def test_sample_includes_saved_labels_for_visible_transitions(client) -> None:
    dataset_id = seed_dataset()
    client.put(
        f"/api/datasets/{dataset_id}/episodes/0/samples/120/label",
        json={"label": 1},
    )
    sample = client.get(
        f"/api/datasets/{dataset_id}/episodes/0/samples/150"
    ).json()
    assert sample["frame_indices"] == [30, 60, 90, 120, 150]
    assert sample["transition_labels"][:2] == [None, None]
    assert sample["transition_labels"][2]["target_frame"] == 120
    assert sample["transition_labels"][2]["label"] == 1
    assert sample["transition_labels"][3] is None


def test_undo_removes_a_new_label_but_keeps_an_audit_record(client) -> None:
    dataset_id = seed_dataset()
    client.put(
        f"/api/datasets/{dataset_id}/episodes/0/samples/120/label",
        json={"label": -1},
    )
    undone = client.post(
        f"/api/datasets/{dataset_id}/episodes/0/samples/120/label/undo"
    )
    assert undone.status_code == 200
    assert undone.json()["label"] is None
    assert undone.json()["coverage"]["labeled_samples"] == 0
    assert (
        client.get(
            f"/api/datasets/{dataset_id}/episodes/0/samples/120"
        ).json()["label"]
        is None
    )
    with connect(read_only=True) as conn:
        history = conn.execute(
            "SELECT label,revision FROM annotation_history WHERE dataset_id=?",
            (dataset_id,),
        ).fetchall()
    assert [(row["label"], row["revision"]) for row in history] == [(-1, 1)]


def test_undo_a_relabel_restores_the_previous_value(client) -> None:
    dataset_id = seed_dataset()
    endpoint = f"/api/datasets/{dataset_id}/episodes/0/samples/120/label"
    client.put(endpoint, json={"label": 1})
    client.put(endpoint, json={"label": 0})
    undone = client.post(f"{endpoint}/undo")
    assert undone.status_code == 200
    assert undone.json()["label"] == 1
    assert undone.json()["revision"] == 3
    sample = client.get(
        f"/api/datasets/{dataset_id}/episodes/0/samples/120"
    ).json()
    assert sample["label"]["label"] == 1
    assert sample["label"]["revision"] == 3


def test_completion_shortens_required_tail(client) -> None:
    dataset_id = seed_dataset()
    for target, label in ((120, 1), (150, 0), (180, -1)):
        assert (
            client.put(
                f"/api/datasets/{dataset_id}/episodes/0/samples/{target}/label",
                json={"label": label},
            ).status_code
            == 200
        )
    completion = client.put(
        f"/api/datasets/{dataset_id}/episodes/0/completion",
        json={"state": "marked", "frame": 180},
    )
    assert completion.status_code == 200
    assert completion.json()["next"]["kind"] == "done"
    status = client.get(f"/api/datasets/{dataset_id}/status").json()
    assert status["coverage"]["export_ready"] is True


def test_early_completion_archives_post_completion_labels(client) -> None:
    dataset_id = seed_dataset()
    for target in sample_targets(260, 30):
        assert (
            client.put(
                f"/api/datasets/{dataset_id}/episodes/0/samples/{target}/label",
                json={"label": 1},
            ).status_code
            == 200
        )
    marked = client.put(
        f"/api/datasets/{dataset_id}/episodes/0/completion",
        json={"state": "marked", "frame": 180},
    )
    assert marked.status_code == 200
    assert marked.json()["discarded_post_completion"] == 2
    assert marked.json()["coverage"]["export_ready"] is True


def test_short_episode_can_answer_completion_without_inventing_a_label_window(client) -> None:
    dataset_id = seed_dataset(length=100)
    queued = client.get(f"/api/datasets/{dataset_id}/queue/current").json()
    assert queued == {"kind": "completion", "episode_index": 0, "suggested_frame": 99}
    sample = client.get(
        f"/api/datasets/{dataset_id}/episodes/0/samples/99"
    )
    assert sample.status_code == 200
    assert sample.json()["completion_only"] is True
    refused = client.put(
        f"/api/datasets/{dataset_id}/episodes/0/samples/99/label",
        json={"label": 1},
    )
    assert refused.status_code == 422


def test_next_episode_navigation_is_sequential_and_never_wraps(client) -> None:
    dataset_id = seed_dataset()
    add_episode(dataset_id, episode_index=3, length=200)
    add_episode(dataset_id, episode_index=8, length=20)

    next_sample = client.get(
        f"/api/datasets/{dataset_id}/episodes/0/next"
    )
    assert next_sample.status_code == 200
    assert next_sample.json()["episode_index"] == 3
    assert next_sample.json()["target_frame"] == 120

    short_sample = client.get(
        f"/api/datasets/{dataset_id}/episodes/3/next"
    )
    assert short_sample.status_code == 200
    assert short_sample.json()["episode_index"] == 8
    assert short_sample.json()["target_frame"] == 19
    assert short_sample.json()["completion_only"] is True

    end = client.get(f"/api/datasets/{dataset_id}/episodes/8/next")
    assert end.status_code == 200
    assert end.json() == {"kind": "end"}


def test_delta_change_refuses_to_orphan_labels(client) -> None:
    dataset_id = seed_dataset()
    client.put(
        f"/api/datasets/{dataset_id}/episodes/0/samples/120/label",
        json={"label": 1},
    )
    refused = client.patch(f"/api/datasets/{dataset_id}", json={"delta_seconds": 2})
    assert refused.status_code == 409
    reset = client.patch(
        f"/api/datasets/{dataset_id}",
        json={"delta_seconds": 2, "reset_annotations": True},
    )
    assert reset.status_code == 200
    assert reset.json()["delta_frames"] == 60
    assert reset.json()["coverage"]["labeled_samples"] == 0
