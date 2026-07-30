from __future__ import annotations

import json
import math
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import Depends, FastAPI, HTTPException, Query, Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import service
from .annotation import frame_window, sample_targets
from .auth import clean_name, login, logout, optional_session, password_matches, require_session
from .config import DATASETS_ROOT, EXPORTS_ROOT, MOUNT_PATH, STATIC_ROOT
from .db import connect, dumps, migrate, now, transaction
from .media import decode_frame, frame_cache_path


@asynccontextmanager
async def lifespan(_app: FastAPI):
    migrate()
    service.clean_interrupted_jobs()
    yield


app = FastAPI(
    title="ARM Advantage Annotation",
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
    lifespan=lifespan,
)


def current_user(session: str | None = Depends(optional_session)) -> str:
    return require_session(session)


User = Annotated[str, Depends(current_user)]


class LoginIn(BaseModel):
    name: str
    password: str


class ImportIn(BaseModel):
    source_url: str = Field(min_length=3, max_length=2048)
    title: str | None = Field(default=None, max_length=120)
    revision: str | None = Field(default=None, max_length=160)
    subpath: str | None = Field(default=None, max_length=500)
    camera_keys: list[str] = Field(default_factory=list, max_length=8)
    delta_seconds: float = Field(default=1.0, gt=0, le=60)


class DatasetPatch(BaseModel):
    camera_keys: list[str] | None = Field(default=None, max_length=8)
    delta_seconds: float | None = Field(default=None, gt=0, le=60)
    reset_annotations: bool = False


class LabelIn(BaseModel):
    label: Literal[-1, 0, 1]


class CompletionIn(BaseModel):
    state: Literal["marked", "never"]
    frame: int | None = Field(default=None, ge=0)


class ExportIn(BaseModel):
    video_mode: Literal["symlink", "copy", "none"] = "symlink"


@app.get("/api/healthz")
def healthz() -> dict[str, Any]:
    with connect(read_only=True) as conn:
        datasets = conn.execute("SELECT count(*) FROM dataset").fetchone()[0]
        labels = conn.execute("SELECT count(*) FROM annotation").fetchone()[0]
    return {
        "ok": True,
        "mount": MOUNT_PATH,
        "datasets": datasets,
        "labels": labels,
        "version": "0.1.0",
    }


@app.post("/api/login")
def api_login(body: LoginIn, response: Response) -> dict[str, Any]:
    name = clean_name(body.name)
    if not password_matches(body.password):
        raise HTTPException(401, "wrong password")
    expires_at = login(response, name)
    return {"name": name, "expires_at": expires_at}


@app.get("/api/me")
def me(session: str | None = Depends(optional_session)) -> dict[str, Any]:
    return {"authenticated": session is not None, "name": session}


@app.post("/api/logout")
def api_logout(response: Response) -> dict[str, bool]:
    logout(response)
    return {"ok": True}


@app.get("/api/datasets")
def datasets(_user: User) -> list[dict[str, Any]]:
    return [
        {**dataset, "coverage": service.coverage(dataset["id"])}
        for dataset in service.list_datasets()
    ]


def _parse_reference_for_insert(body: ImportIn) -> tuple[str, str, str]:
    from .hf_import import parse_hf_dataset_reference

    ref = parse_hf_dataset_reference(body.source_url)
    revision = body.revision or ref.revision or "main"
    subpath = body.subpath if body.subpath is not None else ref.subfolder
    return ref.repo_id, revision, str(subpath or "")


@app.post("/api/datasets", status_code=202)
def import_dataset(body: ImportIn, user: User) -> dict[str, Any]:
    repo_id, revision, subpath = _parse_reference_for_insert(body)
    stamp = now()
    with transaction() as conn:
        cursor = conn.execute(
            "INSERT INTO dataset(source_url,repo_id,revision,subpath,title,root_path,status,"
            "delta_seconds,camera_keys_json,created_by,created_at,updated_at)"
            " VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                body.source_url,
                repo_id,
                revision,
                subpath,
                body.title or (subpath.rsplit("/", 1)[-1] if subpath else repo_id),
                "pending",
                "importing",
                body.delta_seconds,
                dumps(body.camera_keys),
                user,
                stamp,
                stamp,
            ),
        )
        dataset_id = int(cursor.lastrowid)
        root = DATASETS_ROOT / f"dataset-{dataset_id:06d}"
        conn.execute("UPDATE dataset SET root_path=? WHERE id=?", (str(root), dataset_id))
    service.start_import(dataset_id, body.camera_keys)
    return {"id": dataset_id, "status": "importing"}


@app.get("/api/datasets/{dataset_id}/status")
def dataset_status(dataset_id: int, _user: User) -> dict[str, Any]:
    dataset = service.get_dataset(dataset_id)
    if not dataset:
        raise HTTPException(404, "dataset not found")
    return {**dataset, "coverage": service.coverage(dataset_id)}


@app.patch("/api/datasets/{dataset_id}")
def patch_dataset(dataset_id: int, body: DatasetPatch, _user: User) -> dict[str, Any]:
    dataset = service.get_dataset(dataset_id)
    if not dataset:
        raise HTTPException(404, "dataset not found")
    with transaction() as conn:
        count = conn.execute(
            "SELECT (SELECT count(*) FROM annotation WHERE dataset_id=?) +"
            " (SELECT count(*) FROM completion WHERE dataset_id=?)",
            (dataset_id, dataset_id),
        ).fetchone()[0]
        changing_grid = body.delta_seconds is not None and not math.isclose(
            body.delta_seconds, float(dataset["delta_seconds"])
        )
        if changing_grid and count and not body.reset_annotations:
            raise HTTPException(
                409,
                "changing delta changes the annotation grid; set reset_annotations=true",
            )
        if changing_grid and body.reset_annotations:
            conn.execute("DELETE FROM annotation WHERE dataset_id=?", (dataset_id,))
            conn.execute("DELETE FROM completion WHERE dataset_id=?", (dataset_id,))
        camera_keys = body.camera_keys if body.camera_keys is not None else dataset["camera_keys"]
        available = set(dataset["info"].get("features", {}))
        unknown = [key for key in camera_keys if key not in available]
        if unknown:
            raise HTTPException(422, f"unknown camera keys: {', '.join(unknown)}")
        delta_seconds = body.delta_seconds or float(dataset["delta_seconds"])
        delta_frames = max(1, round(float(dataset["fps"]) * delta_seconds))
        conn.execute(
            "UPDATE dataset SET camera_keys_json=?,delta_seconds=?,delta_frames=?,updated_at=?"
            " WHERE id=?",
            (dumps(camera_keys), delta_seconds, delta_frames, now(), dataset_id),
        )
    updated = service.get_dataset(dataset_id)
    return {**updated, "coverage": service.coverage(dataset_id)}


@app.get("/api/datasets/{dataset_id}/queue/current")
def current_queue(dataset_id: int, _user: User) -> dict[str, Any]:
    try:
        item = service.queue_current(dataset_id)
    except KeyError:
        raise HTTPException(404, "dataset not found") from None
    if item.get("kind") == "sample":
        return sample_payload(
            dataset_id,
            int(item["episode_index"]),
            int(item["target_frame"]),
        )
    return item


def sample_payload(dataset_id: int, episode_index: int, target: int) -> dict[str, Any]:
    dataset = service.get_dataset(dataset_id)
    if not dataset:
        raise HTTPException(404, "dataset not found")
    delta = int(dataset["delta_frames"])
    with connect(read_only=True) as conn:
        episode = conn.execute(
            "SELECT * FROM episode WHERE dataset_id=? AND episode_index=?",
            (dataset_id, episode_index),
        ).fetchone()
    if not episode:
        raise HTTPException(404, "episode not found")
    valid = sample_targets(int(episode["length"]), delta)
    completion_only = not valid and target == max(0, int(episode["length"]) - 1)
    if target not in valid and not completion_only:
        raise HTTPException(422, "target is not on the current five-frame annotation grid")
    frames = (
        frame_window(target, delta)
        if not completion_only
        else [max(0, target - step * delta) for step in range(4, -1, -1)]
    )
    transition_targets = frames[1:]
    placeholders = ",".join("?" for _ in transition_targets)
    with connect(read_only=True) as conn:
        transition_rows = conn.execute(
            "SELECT start_frame,target_frame,label,annotator,revision,updated_at FROM annotation"
            " WHERE dataset_id=? AND episode_index=? AND delta_frames=?"
            f" AND target_frame IN ({placeholders})",
            (dataset_id, episode_index, delta, *transition_targets),
        ).fetchall()
        completion = conn.execute(
            "SELECT state,frame,annotator,updated_at FROM completion"
            " WHERE dataset_id=? AND episode_index=?",
            (dataset_id, episode_index),
        ).fetchone()
    labels_by_target = {int(row["target_frame"]): dict(row) for row in transition_rows}
    camera_keys = dataset["camera_keys"]
    return {
        "kind": "sample",
        "dataset_id": dataset_id,
        "episode_index": episode_index,
        "episode_length": int(episode["length"]),
        "task": episode["task"],
        "target_frame": target,
        "start_frame": target - delta,
        "delta_frames": delta,
        "delta_seconds": float(dataset["delta_seconds"]),
        "realized_delta_seconds": delta / float(dataset["fps"]),
        "fps": float(dataset["fps"]),
        "frame_indices": frames,
        "camera_keys": camera_keys,
        "frame_rows": [
            {
                "camera": camera,
                "images": [
                    f"api/datasets/{dataset_id}/episodes/{episode_index}/frames/{frame}"
                    f"?camera={camera}"
                    for frame in frames
                ],
            }
            for camera in camera_keys
        ],
        "label": labels_by_target.get(target),
        "transition_labels": [labels_by_target.get(frame) for frame in transition_targets],
        "completion": dict(completion) if completion else None,
        "completion_only": completion_only,
        "coverage": service.coverage(dataset_id),
    }


@app.get("/api/datasets/{dataset_id}/episodes/{episode_index}/samples/{target}")
def get_sample(dataset_id: int, episode_index: int, target: int, _user: User) -> dict[str, Any]:
    return sample_payload(dataset_id, episode_index, target)


@app.get("/api/datasets/{dataset_id}/episodes/{episode_index}/next")
def get_next_episode(dataset_id: int, episode_index: int, _user: User) -> dict[str, Any]:
    dataset = service.get_dataset(dataset_id)
    if not dataset:
        raise HTTPException(404, "dataset not found")
    if dataset["status"] != "ready":
        raise HTTPException(409, "dataset is not ready")
    episodes = service.get_episodes(dataset_id)
    next_episode = next(
        (
            episode
            for episode in episodes
            if int(episode["episode_index"]) > episode_index
        ),
        None,
    )
    if next_episode is None:
        return {"kind": "end"}
    delta = int(dataset["delta_frames"])
    targets = sample_targets(int(next_episode["length"]), delta)
    target = targets[0] if targets else max(0, int(next_episode["length"]) - 1)
    return sample_payload(dataset_id, int(next_episode["episode_index"]), target)


@app.put("/api/datasets/{dataset_id}/episodes/{episode_index}/samples/{target}/label")
def put_label(
    dataset_id: int,
    episode_index: int,
    target: int,
    body: LabelIn,
    user: User,
) -> dict[str, Any]:
    sample = sample_payload(dataset_id, episode_index, target)
    if sample["completion_only"]:
        raise HTTPException(422, "this short episode has no full five-frame label window")
    stamp = now()
    with transaction() as conn:
        existing = conn.execute(
            "SELECT * FROM annotation WHERE dataset_id=? AND episode_index=?"
            " AND target_frame=? AND delta_frames=?",
            (dataset_id, episode_index, target, sample["delta_frames"]),
        ).fetchone()
        if existing:
            revision = int(existing["revision"]) + 1
            conn.execute(
                "INSERT INTO annotation_history(annotation_id,dataset_id,episode_index,"
                "start_frame,target_frame,delta_frames,label,annotator,revision,recorded_at)"
                " VALUES(?,?,?,?,?,?,?,?,?,?)",
                (
                    existing["id"],
                    dataset_id,
                    episode_index,
                    existing["start_frame"],
                    target,
                    sample["delta_frames"],
                    existing["label"],
                    existing["annotator"],
                    existing["revision"],
                    stamp,
                ),
            )
            conn.execute(
                "UPDATE annotation SET label=?,annotator=?,revision=?,updated_at=? WHERE id=?",
                (body.label, user, revision, stamp, existing["id"]),
            )
        else:
            revision = 1
            conn.execute(
                "INSERT INTO annotation(dataset_id,episode_index,start_frame,target_frame,"
                "delta_frames,label,annotator,revision,created_at,updated_at)"
                " VALUES(?,?,?,?,?,?,?,?,?,?)",
                (
                    dataset_id,
                    episode_index,
                    target - sample["delta_frames"],
                    target,
                    sample["delta_frames"],
                    body.label,
                    user,
                    revision,
                    stamp,
                    stamp,
                ),
            )
    return {
        "ok": True,
        "label": body.label,
        "revision": revision,
        "next": service.queue_current(dataset_id),
        "coverage": service.coverage(dataset_id),
    }


@app.post("/api/datasets/{dataset_id}/episodes/{episode_index}/samples/{target}/label/undo")
def undo_label(
    dataset_id: int,
    episode_index: int,
    target: int,
    user: User,
) -> dict[str, Any]:
    sample = sample_payload(dataset_id, episode_index, target)
    if sample["completion_only"]:
        raise HTTPException(422, "this short episode has no label to undo")
    stamp = now()
    restored_label: int | None = None
    revision: int | None = None
    with transaction() as conn:
        existing = conn.execute(
            "SELECT * FROM annotation WHERE dataset_id=? AND episode_index=?"
            " AND target_frame=? AND delta_frames=?",
            (dataset_id, episode_index, target, sample["delta_frames"]),
        ).fetchone()
        if not existing:
            raise HTTPException(409, "this transition has no saved label to undo")
        previous = conn.execute(
            "SELECT label FROM annotation_history WHERE annotation_id=? AND revision<?"
            " ORDER BY id DESC LIMIT 1",
            (existing["id"], existing["revision"]),
        ).fetchone()
        conn.execute(
            "INSERT INTO annotation_history(annotation_id,dataset_id,episode_index,"
            "start_frame,target_frame,delta_frames,label,annotator,revision,recorded_at)"
            " VALUES(?,?,?,?,?,?,?,?,?,?)",
            (
                existing["id"],
                dataset_id,
                episode_index,
                existing["start_frame"],
                target,
                existing["delta_frames"],
                existing["label"],
                existing["annotator"],
                existing["revision"],
                stamp,
            ),
        )
        if previous:
            restored_label = int(previous["label"])
            revision = int(existing["revision"]) + 1
            conn.execute(
                "UPDATE annotation SET label=?,annotator=?,revision=?,updated_at=? WHERE id=?",
                (restored_label, user, revision, stamp, existing["id"]),
            )
        else:
            conn.execute("DELETE FROM annotation WHERE id=?", (existing["id"],))
    return {
        "ok": True,
        "label": restored_label,
        "revision": revision,
        "undone_label": int(existing["label"]),
        "next": service.queue_current(dataset_id),
        "coverage": service.coverage(dataset_id),
    }


@app.put("/api/datasets/{dataset_id}/episodes/{episode_index}/completion")
def put_completion(
    dataset_id: int,
    episode_index: int,
    body: CompletionIn,
    user: User,
) -> dict[str, Any]:
    with connect(read_only=True) as conn:
        episode = conn.execute(
            "SELECT length FROM episode WHERE dataset_id=? AND episode_index=?",
            (dataset_id, episode_index),
        ).fetchone()
    if not episode:
        raise HTTPException(404, "episode not found")
    if body.state == "marked":
        if body.frame is None:
            raise HTTPException(422, "a marked completion needs a frame")
        if body.frame >= int(episode["length"]):
            raise HTTPException(422, "completion frame is outside the episode")
    elif body.frame is not None:
        raise HTTPException(422, "never-completes cannot have a frame")
    discarded_post_completion = 0
    with transaction() as conn:
        if body.state == "marked":
            stale = conn.execute(
                "SELECT * FROM annotation WHERE dataset_id=? AND episode_index=?"
                " AND target_frame>?",
                (dataset_id, episode_index, body.frame),
            ).fetchall()
            for item in stale:
                conn.execute(
                    "INSERT INTO annotation_history(annotation_id,dataset_id,episode_index,"
                    "start_frame,target_frame,delta_frames,label,annotator,revision,recorded_at)"
                    " VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (
                        item["id"],
                        dataset_id,
                        episode_index,
                        item["start_frame"],
                        item["target_frame"],
                        item["delta_frames"],
                        item["label"],
                        item["annotator"],
                        item["revision"],
                        now(),
                    ),
                )
            discarded_post_completion = len(stale)
            conn.execute(
                "DELETE FROM annotation WHERE dataset_id=? AND episode_index=? AND target_frame>?",
                (dataset_id, episode_index, body.frame),
            )
        conn.execute(
            "INSERT INTO completion(dataset_id,episode_index,state,frame,annotator,updated_at)"
            " VALUES(?,?,?,?,?,?) ON CONFLICT(dataset_id,episode_index) DO UPDATE SET"
            " state=excluded.state,frame=excluded.frame,annotator=excluded.annotator,"
            " updated_at=excluded.updated_at",
            (dataset_id, episode_index, body.state, body.frame, user, now()),
        )
    return {
        "ok": True,
        "completion": body.model_dump(),
        "discarded_post_completion": discarded_post_completion,
        "next": service.queue_current(dataset_id),
        "coverage": service.coverage(dataset_id),
    }


@app.get("/api/datasets/{dataset_id}/episodes/{episode_index}/frames/{frame}")
def get_frame(
    dataset_id: int,
    episode_index: int,
    frame: int,
    _user: User,
    camera: str = Query(...),
) -> FileResponse:
    dataset = service.get_dataset(dataset_id)
    if not dataset:
        raise HTTPException(404, "dataset not found")
    with connect(read_only=True) as conn:
        episode = conn.execute(
            "SELECT * FROM episode WHERE dataset_id=? AND episode_index=?",
            (dataset_id, episode_index),
        ).fetchone()
    if not episode:
        raise HTTPException(404, "episode not found")
    if not (0 <= frame < int(episode["length"])):
        raise HTTPException(416, "frame outside episode")
    cameras = json.loads(episode["cameras_json"])
    if camera not in cameras:
        raise HTTPException(404, "camera was not imported for this episode")
    meta = cameras[camera]
    relative = Path(meta["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise HTTPException(500, "unsafe indexed video path")
    video_path = Path(dataset["root_path"]) / relative
    timestamp = float(meta["from_timestamp"]) + frame / float(dataset["fps"])
    output = frame_cache_path(dataset_id, episode_index, camera, frame)
    decoded = decode_frame(video_path, timestamp=timestamp, output_path=output)
    return FileResponse(decoded, media_type="image/jpeg", headers={"Cache-Control": "private,max-age=86400"})


@app.post("/api/datasets/{dataset_id}/exports", status_code=202)
def create_export(dataset_id: int, body: ExportIn, user: User) -> dict[str, Any]:
    dataset = service.get_dataset(dataset_id)
    if not dataset:
        raise HTTPException(404, "dataset not found")
    progress = service.coverage(dataset_id)
    if not progress["export_ready"]:
        raise HTTPException(409, "all episode labels and completion answers are required")
    stamp = now()
    with transaction() as conn:
        cursor = conn.execute(
            "INSERT INTO export_job(dataset_id,status,output_path,video_mode,created_by,"
            "created_at,updated_at) VALUES(?,?,?,?,?,?,?)",
            (dataset_id, "queued", "pending", body.video_mode, user, stamp, stamp),
        )
        export_id = int(cursor.lastrowid)
        output = EXPORTS_ROOT / f"dataset-{dataset_id:06d}-export-{export_id:06d}"
        conn.execute(
            "UPDATE export_job SET output_path=? WHERE id=?", (str(output), export_id)
        )
    service.start_export(export_id)
    return {"id": export_id, "dataset_id": dataset_id, "status": "queued"}


@app.get("/api/exports/{export_id}")
def export_status(export_id: int, _user: User) -> dict[str, Any]:
    with connect(read_only=True) as conn:
        row = conn.execute("SELECT * FROM export_job WHERE id=?", (export_id,)).fetchone()
    if not row:
        raise HTTPException(404, "export not found")
    item = dict(row)
    item["manifest"] = json.loads(item.pop("manifest_json")) if item.get("manifest_json") else None
    return item


if STATIC_ROOT.is_dir():
    app.mount("/assets", StaticFiles(directory=STATIC_ROOT), name="assets")


@app.get("/{path:path}", response_model=None)
def spa(path: str) -> FileResponse | JSONResponse:
    if path.startswith("api/"):
        return JSONResponse(status_code=404, content={"detail": "route not found"})
    candidate = (STATIC_ROOT / path).resolve()
    if path and candidate.is_file() and STATIC_ROOT in candidate.parents:
        return FileResponse(candidate)
    index = STATIC_ROOT / "index.html"
    if index.is_file():
        return FileResponse(index)
    return JSONResponse(status_code=503, content={"detail": "frontend not built"})
