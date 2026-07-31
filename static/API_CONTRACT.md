# Browser API contract

`app.js` resolves every path relative to `document.baseURI`. This is
load-bearing for the production mount at `/arm/advantage_annotation/`: the UI
never assumes that `/api` exists at the domain root.

All mutations use the signed session cookie and JSON request bodies. FastAPI
errors are read from `{"detail": "..."}`.

## Session

- `GET api/me`
  - response: `{"authenticated": bool, "name": string|null}`
- `POST api/login`
  - body: `{"name": string, "password": string}`
  - response: `{"name": string, "expires_at": int}`
- `POST api/logout`
  - response: `{"ok": true}`

## Datasets

- `GET api/datasets`
  - response: an array of `Dataset`
- `POST api/datasets`
  - body:
    `{"source_url": string, "revision": string|null, "subpath": string|null,
    "camera_keys": string[], "delta_seconds": number}`
  - response (202): `{"id": int, "status": "importing"}`
- `POST api/datasets/upload?filename=dataset.zip&delta_seconds=1.0&camera_keys=...`
  - body: raw ZIP bytes (`Content-Type: application/zip`); repeat `camera_keys` to select
    multiple cameras
  - response (202): `{"id": int, "status": "importing"}`; extraction, LeRobot v3 validation,
    and indexing continue in the background
- `GET api/datasets/{id}/status`
  - response: one `Dataset`
- `PATCH api/datasets/{id}`
  - body:
    `{"delta_seconds": number, "camera_keys"?: string[],
    "reset_annotations"?: bool}`
  - response: updated `Dataset`
  - a grid change with existing labels returns 409 unless
    `reset_annotations=true`
- `GET api/datasets/{id}/episodes/{episode}/progress-curve`
  - response: episode metadata and ordered progress keypoints
- `PUT api/datasets/{id}/episodes/{episode}/progress-curve`
  - body: `{"points":[{"frame":0,"value":0.0}, ...]}`
  - the first and last frames are required; values must be in `[0,1]`
  - export linearly interpolates one float32 `progress` value per frame

A `Dataset` includes:

```json
{
  "id": 1,
  "title": "owner/repository",
  "source_url": "owner/repository",
  "repo_id": "owner/repository",
  "revision": "main",
  "subpath": "",
  "status": "importing | ready | failed",
  "error": null,
  "fps": 30.0,
  "delta_seconds": 1.0,
  "delta_frames": 30,
  "camera_keys": ["observation.images.front"],
  "info": {"codebase_version": "v3.0", "features": {}},
  "total_episodes": 10,
  "total_frames": 25000,
  "coverage": {
    "total_samples": 800,
    "labeled_samples": 320,
    "completed_episodes": 3,
    "total_episodes": 10,
    "percent": 40.0,
    "export_ready": false
  }
}
```

In curve mode, `coverage` also includes `curve_completed_episodes`, `curve_percent`,
`curve_export_ready`, and mode-aware `export_ready`. A curve is complete only after its first and
last-frame keypoints have been persisted; export requires one complete curve per episode.

## Queue, frames, and labels

- `GET api/datasets/{id}/queue/current`
  - response is one of:
    - a full `Sample` with `kind="sample"`
    - `{"kind":"completion","episode_index":int,"suggested_frame":int}`
    - `{"kind":"done"}`
    - `{"kind":"waiting","status":string,"error":string|null}`
- `GET api/datasets/{id}/episodes/{episode}/samples/{target}`
  - response: a full `Sample`
- `GET api/datasets/{id}/episodes/{episode}/next`
  - response: the first `Sample` in the immediate next episode, or
    `{"kind":"end"}`; this endpoint never wraps to an earlier episode
- `PUT api/datasets/{id}/episodes/{episode}/samples/{target}/label`
  - body: `{"label": -1|0|1}`
  - response:
    `{"ok":true,"label":int,"revision":int,"next":QueueHint,
    "coverage":Coverage}`
- `POST api/datasets/{id}/episodes/{episode}/samples/{target}/label/undo`
  - restores the previous revision when one exists, otherwise removes the
    newly created label
  - the superseded value is always retained in `annotation_history`
  - response:
    `{"ok":true,"label":int|null,"revision":int|null,
    "undone_label":int,"next":QueueHint,"coverage":Coverage}`
- `PUT api/datasets/{id}/episodes/{episode}/completion`
  - marked body: `{"state":"marked","frame":int}`
  - never body: `{"state":"never","frame":null}`
  - response:
    `{"ok":true,"completion":object,"next":QueueHint,
    "coverage":Coverage}`
- `GET api/datasets/{id}/episodes/{episode}/frames/{frame}?camera={key}`
  - response: `image/jpeg`

The `Sample` shape used by the five-card strip is:

```json
{
  "kind": "sample",
  "dataset_id": 1,
  "episode_index": 0,
  "episode_length": 260,
  "task": "move the cube",
  "target_frame": 120,
  "start_frame": 90,
  "delta_frames": 30,
  "delta_seconds": 1.0,
  "realized_delta_seconds": 1.0,
  "fps": 30.0,
  "frame_indices": [0, 30, 60, 90, 120],
  "camera_keys": ["observation.images.front"],
  "frame_rows": [
    {
      "camera": "observation.images.front",
      "images": [
        "api/datasets/1/episodes/0/frames/0?camera=observation.images.front"
      ]
    }
  ],
  "label": null,
  "transition_labels": [null, null, null, null],
  "completion": null,
  "coverage": {}
}
```

`frame_rows[].images` has five entries. A stored label has `label`,
`annotator`, `revision`, and `updated_at`. A stored completion has `state`,
`frame`, `annotator`, and `updated_at`. `transition_labels` has four entries,
one for each left-to-right frame transition; saved entries additionally carry
`start_frame` and `target_frame`.

## Export

- `POST api/datasets/{id}/exports`
  - body: `{"video_mode":"symlink"|"copy"|"none"}`
  - response (202): `{"id":int,"dataset_id":int,"status":"queued"}`
- `GET api/exports/{id}`
  - response includes `status` (`queued`, `running`, `ready`, or `failed`),
    `output_path`, `error`, and the parsed `manifest` when ready.

The browser enables export only when `Dataset.coverage.export_ready` is true;
the server independently enforces the same gate.
