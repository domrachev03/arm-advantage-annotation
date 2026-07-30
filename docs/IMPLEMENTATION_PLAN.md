# ARM annotation platform implementation plan

Status: implementation contract, agreed from the ARM paper, FluxVLA's current
consumer, LeRobot v3, and the deployed Origami annotation platform.

## 1. Product contract

The platform imports a Hugging Face LeRobot v3 dataset, shows one causal ARM
window at a time, saves direct human advantage labels, records task completion
separately, and exports a non-destructive LeRobot v3 copy that FluxVLA can train
on.

The annotation unit is the final adjacent pair in a five-frame causal window:

```
[t - 4Δ, t - 3Δ, t - 2Δ, t - Δ, t]
                                  └── label this transition ──┘
```

The three labels are:

- `+1`: Progressive
- `0`: Stagnant
- `-1`: Regressive

`Δ` is persisted per dataset, specified in seconds at import, and resolved to
`round(fps * Δ)` frames. The default is one second, matching the paper. Changing
the grid after labels exist requires an explicit destructive reset.

The completion answer is separate from advantage: either a marked completion
frame or an explicit "never completes" answer. This supplies FluxVLA's success
head without pretending that an advantage label is a completion label.

## 2. Repository layout

The application is a standalone repository with its own lightweight runtime:

```
arm-advantage-annotation/
  app/                 FastAPI, SQLite, HF import, media, export
  static/              dependency-free browser UI
  tests/               unit and API integration tests
  tools/               CLI import/export/backup entry points
  deploy/systemd/      persistent local service
  docs/                contract and runbook
  pyproject.toml
```

It does not import Torch or the FluxVLA training runtime.

## 3. Import flow

1. Accept a canonical `huggingface.co/datasets/...` URL, `hf://datasets/...`,
   or `owner/repo` identifier.
2. Parse revision and optional subdirectory without accepting arbitrary hosts
   or filesystem traversal.
3. Download metadata first with `huggingface_hub.snapshot_download`.
4. Validate LeRobot major version 3 and its `info.json`, episode metadata,
   data shards, and video features.
5. Select a camera (explicitly requested, otherwise prefer a high/front view)
   and download only `meta/`, `data/`, and that camera's videos.
6. Index episodes, tasks, shared-video offsets, and annotation samples in
   SQLite. Imports run as restart-safe background jobs and expose progress and
   errors in the UI.

The original snapshot is immutable. Generated thumbnails, annotations, and
exports are stored outside it.

## 4. Annotation and persistence

SQLite stores dataset/import state, episode metadata, one label per
`(dataset, episode, target_frame, delta_frames)`, and the completion answer.
Every mutation requires a signed annotator session. Labels autosave immediately;
relabeling the same pair is an audited upsert.

The queue is deterministic and resumes the first unlabeled pair. At episode
boundaries the UI asks for completion. A completion mark ends the labeling
queue at that frame because exported progress is held at `1.0` afterward.

Frame images are decoded from the selected LeRobot v3 shared MP4 with:

```
video_from_timestamp + episode_frame / fps
```

and cached as JPEG thumbnails. Cache paths are derived from database IDs, never
from request paths.

## 5. FluxVLA export

FluxVLA's current `ARMDataset` does not consume raw tri-state rows. It requires
a scalar float32 `progress` column and derives its own interval targets with a
symmetric `1e-3` deadband. Therefore export will:

1. Refuse incomplete datasets rather than silently train missing episodes.
2. Copy `meta/` and `data/` to a new export root; never mutate the snapshot.
3. Link or copy `videos/` according to the selected export mode.
4. Reconstruct an unbounded cumulative signal from the direct labels on the
   persisted Δ grid.
5. Min-max scale to `[0, 1]`; anchor a marked completion frame at `1.0` and
   hold it there. Explicit non-completion is capped below FluxVLA's success
   threshold.
6. Write `progress` as float32 to every data row and declare the feature in
   `meta/info.json`.
7. Write `meta/arm_pairs.parquet` with raw labels/provenance and
   `meta/arm_export_manifest.json` with the exact reconstruction contract.

The exporter also verifies that recomputing labels from exported progress gives
the intended sign on the annotation grid.

## 6. HTTP/UI surface

- login/me
- dataset import, status, selection, camera and Δ configuration
- episode/sample queue and coverage
- five frame-image endpoint set
- label upsert and completion answer
- export creation/status
- health/readiness

The UI is a responsive single page with the five-frame strip, the final pair
visually emphasized, large `-1 / 0 / +1` controls, keyboard shortcuts, current
task/episode context, import progress, and export readiness.

## 7. Verification and deployment

Tests cover URL parsing, v3 validation, timestamp/frame resolution, Δ math,
annotation idempotency, completion semantics, reconstruction, source
immutability, progress dtype/declaration, and the FluxVLA deadband round trip.
A synthetic v3 fixture exercises import -> label -> export end to end.

The production service runs as one local Uvicorn worker on `127.0.0.1:8120`
with a generated signing secret and shared password, using a user systemd unit.
Tailscale Serve/Funnel, Caddy, or nginx can publish it at a dedicated hostname
or stripped subpath. Deployment is complete only after local and public health
checks plus a browser/API smoke test succeed.
