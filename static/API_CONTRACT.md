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
    "camera_keys": string[], "delta_frames": integer}`
  - response (202): `{"id": int, "status": "importing"}`
- `GET api/datasets/{id}/status`
  - response: one `Dataset`
- `PATCH api/datasets/{id}`
  - body:
    `{"delta_frames": integer, "camera_keys"?: string[],
    "reset_annotations"?: bool}`
  - response: updated `Dataset`
  - a grid change with existing labels returns 409 unless
    `reset_annotations=true`

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
  "delta_seconds": 0.2666666667,
  "delta_frames": 8,
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

`delta_frames` is the canonical exact gap between selected LeRobot timestamp
rows. Responses retain the derived `delta_seconds = delta_frames / fps` for
display and export compatibility. Requests may still provide `delta_seconds`
instead of `delta_frames` as a deprecated compatibility path, but not both.

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

## Prediction runs

The comparison view is read-only; uploads happen out of band with
`POST api/predictions` (see `docs/model_predictions.md`).

- `GET api/datasets/{id}/predictions`
  - response: an array of `PredictionRun`, newest first
- `GET api/predictions/{run_id}`
  - response: one `PredictionRun` plus `episodes`, an array of
    `{"episode_index":int,"split":string,"length":int,"metrics":Metrics,
    "success":Success|null}`
- `GET api/predictions/{run_id}/episodes/{episode}?max_points={n}`
  - response: one downsampled `EpisodeSeries`

A `PredictionRun` carries the artifact's `run` and `grid` blocks flattened onto
the row (`name`, `config_id`, `git_sha`, `git_dirty`, `delta_frames`,
`window_size`, …), the parsed `dataset` and `aggregate_metrics` blocks,
`episode_count`, and the upload provenance (`uploaded_by`, `artifact_sha256`).
`trained_at` is the artifact's `run.created_at`; `created_at` is when the row
was written.

`aggregate_metrics` is `{"overall": Aggregate, "by_split": {split: Aggregate}}`.
An `Aggregate` and a per-episode `Metrics` both carry `spearman`, `pearson`,
`mae`, and `interval_accuracy`, plus an optional `linear_ramp_baseline` holding
the same four keys for a linear time ramp over the same frames. The browser
renders each metric against that baseline, so a missing baseline is displayed
rather than assumed.

An `EpisodeSeries` is:

```json
{
  "run_id": 1,
  "episode_index": 0,
  "split": "train",
  "length": 1039,
  "delta_frames": 1,
  "fps": 30.0,
  "metrics": {},
  "success": null,
  "frame_indices": [0, 1, 2],
  "predicted_progress": [0.0, 0.01, 0.02],
  "gt_progress": [0.0, 0.01, 0.02],
  "intervals": [
    {
      "target_frame": 4,
      "start_frame": 0,
      "delta_frames": 1,
      "predicted_label": 1,
      "gt_label": 1,
      "predicted_probabilities": null
    }
  ],
  "sampling": {}
}
```

`frame_indices` gives the true frame index of every plotted point, so the
curves stay correct after thinning. `sampling` reports what was thinned
(`frames`, `frame_points`, `intervals`, `interval_points`,
`interval_disagreements`, `agreeing_intervals_dropped`, and the two
`*_downsampled` flags). Every window whose `predicted_label` differs from a
non-null `gt_label` is kept, so `interval_points` may exceed `max_points`.

## Export

- `POST api/datasets/{id}/exports`
  - body: `{"video_mode":"symlink"|"copy"|"none"}`
  - response (202): `{"id":int,"dataset_id":int,"status":"queued"}`
- `GET api/exports/{id}`
  - response includes `status` (`queued`, `running`, `ready`, or `failed`),
    `output_path`, `error`, and the parsed `manifest` when ready.

The browser enables export only when `Dataset.coverage.export_ready` is true;
the server independently enforces the same gate.
