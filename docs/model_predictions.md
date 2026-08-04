# Model prediction artifact

Status: upload contract between the ARM training track and this application.

An ARM checkpoint predicts a relative advantage in `{-1, 0, +1}` for the final
adjacent pair of a causal five-frame window and a per-episode success answer.
Those interval predictions are reconstructed into a dense per-frame progress
curve in `[0, 1]`. A *prediction artifact* carries one evaluated checkpoint's
curves, interval labels, and metrics into this application so the compare view
can render them against the human ground truth.

This document is normative. The training track writes files against it and the
upload endpoint validates against it; neither side may extend it silently. A
change that is not backwards compatible requires a `schema_version` bump agreed
on both sides.

## 1. Transport

One artifact is one UTF-8 JSON document holding a single object.

- File name: `<run.name>.arm_predictions.json`, optionally gzipped as
  `<run.name>.arm_predictions.json.gz`.
- Upload: the file is the raw request body of `POST /api/predictions`
  (section 13), so any HTTP client can send it with `--data-binary`. The server
  detects gzip from the magic bytes, not from the file name or the content
  type. `prediction_run.artifact_sha256` and `prediction_run.artifact_bytes`
  record the bytes exactly as uploaded, so a gzipped upload is digested
  compressed.
- Encoding: UTF-8 without a byte order mark, LF line endings, one trailing
  newline.
- Serialization: any JSON writer is acceptable. Producers should use compact
  separators for the dense arrays; the example fixture is indented only because
  it is small enough to stay readable in review.
- Size ceiling: the server rejects anything above 64 MiB after decompression.

### 1.1 Why JSON and not a parquet sidecar

The reference run is 60 episodes and 58742 frames. Serialized at the precision
this schema requires, that whole run is:

| `delta_frames` | JSON | gzipped |
| --- | --- | --- |
| 1 | 6.4 MB | 1.0 MB |
| 2 | 3.7 MB | 0.7 MB |
| 8 | 1.7 MB | 0.5 MB |

Even the densest grid is an ordinary single HTTP upload, so a parquet sidecar
would buy bandwidth the workflow does not need while costing real complexity: a
multi-file or zipped container, an ordering contract between the container
members, `pyarrow` on the producer side, and a binary blob in place of a
reviewable test fixture. Ingest writes row-wise into SQLite, so a columnar
container saves nothing on the consumer side either.

Dense per-frame values therefore travel as inline JSON arrays of numbers. There
is no sidecar and no alternative container. If a future run genuinely outgrows
the ceiling, that is a `schema_version` bump, not an undocumented second path.

## 2. Conventions

### 2.1 Units and dtypes

| Concept | Representation |
| --- | --- |
| Progress | dimensionless JSON number in `[0, 1]`; produced as float32, serialized rounded to at most 6 decimal places |
| Frame index | 0-based integer within its episode, matching LeRobot `frame_index` and this application's `episode.length` |
| Interval label | integer, exactly `-1`, `0`, or `1`; never a float, never a string |
| Frame rate | `fps` in Hz as a JSON number |
| Gap | `delta_frames` as an exact positive integer count of frame-index steps; seconds are derived as `delta_frames / fps` and are never carried in the artifact |
| Timestamp | RFC 3339 with an explicit UTC offset, for example `2026-08-04T09:58:41+00:00` |
| Digest | lowercase hexadecimal SHA-256 |
| Path | string as seen on the training host; informational provenance only |

### 2.2 Null and absence rules

- Absence is JSON `null`. Sentinels such as `NaN`, `Infinity`, `"NaN"`, `-1.0`,
  or `-999` are rejected. Every progress value must be finite.
- Optional keys may be omitted; an omitted optional key is exactly equivalent
  to that key present with value `null`. Required keys must be present, even
  when their value is `null`.
- `spearman` and `pearson` are `null` when the episode has fewer than two
  frames or when either series has zero variance. Rank correlation uses average
  ranks for ties.
- `gt_label` is `null` for a window with no human label, which is normal after
  a marked completion frame.
- `success.gt` and `success.gt_frame` are `null` when the episode has no
  recorded completion answer.
- `success_f1` is `null` when a scope contains no positive ground truth and no
  positive prediction.
- An aggregate that averages only `null` inputs is itself `null`.

## 3. Top-level object

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `schema_version` | integer | yes | `1` for this document |
| `artifact_kind` | string | yes | the constant `"arm_prediction_run"` |
| `generated_at` | timestamp | yes | when the artifact file was written |
| `tool` | string | yes | producer identifier, for example `"fluxvla.tools.arm_eval.evaluate_arm:export_prediction_artifact"` |
| `run` | object | yes | see section 4 |
| `dataset` | object | yes | see section 5 |
| `grid` | object | yes | see section 6 |
| `episodes` | array of object | yes | see section 7; non-empty, unique `episode_index`, sorted ascending |
| `aggregate_metrics` | object | yes | see section 9 |

No other top-level key is permitted.

`episodes` need not cover every episode of the dataset. The compare view renders
what is present and reports the covered fraction.

## 4. `run`

Run metadata identifies the checkpoint well enough to reproduce it.

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `name` | string | yes | human-readable run name, 1–120 characters; unique per dataset within this application |
| `checkpoint_path` | string | yes | absolute path to the evaluated checkpoint on the training host |
| `checkpoint_sha256` | string or null | yes | digest of that checkpoint file |
| `config_id` | string | yes | stable configuration identifier, for example `"arm/kuka_assemble_front_r3"` |
| `config_path` | string or null | yes | path to the configuration file in the training repository |
| `git_sha` | string | yes | 40-character commit SHA of the training repository |
| `git_dirty` | boolean | yes | whether the working tree had uncommitted changes at training time |
| `training_command` | string | yes | the exact command line that produced the checkpoint |
| `seed` | integer or null | yes | training seed |
| `created_at` | timestamp | yes | when the checkpoint was produced; distinct from `generated_at` |
| `split_file` | string | yes | path to the train/val/test split definition in the training repository |
| `split_file_sha256` | string or null | yes | digest of that split file |
| `notes` | string or null | no | free-text remark shown next to the run |

The split *assignment* is not repeated here. It lives on each episode as
`episodes[].split`, which is the single authoritative source, so an artifact
cannot disagree with itself. `split_file` and `split_file_sha256` exist only to
trace that assignment back to the training repository.

## 5. `dataset`

The artifact must bind to a dataset that is already registered and imported in
this application. It does not create one.

The application identifies a dataset by the triple `(repo_id, revision,
subpath)` from the `dataset` table. Import normalizes an unpinned reference to
`revision = "main"` and an absent subdirectory to the empty string, so those
same normalized values must be written here.

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `repo_id` | string | yes | canonical `owner/repository`, matching `dataset.repo_id` |
| `revision` | string | yes | branch, tag, or commit, matching `dataset.revision`; `"main"` when unpinned |
| `subpath` | string | yes | subdirectory within the repository, matching `dataset.subpath`; `""` when none |
| `source_url` | string or null | yes | the reference the run was resolved from; informational, never used for binding |
| `dataset_id` | integer or null | yes | optional hint; when non-null the server checks it against the resolved row and rejects a mismatch |
| `fps` | number | yes | frame rate of the source dataset |
| `total_episodes` | integer | yes | episode count of the source dataset |
| `total_frames` | integer | yes | frame count of the source dataset |

Binding rules enforced on upload:

1. Exactly one registered dataset must match `(repo_id, revision, subpath)`;
   zero matches is a rejection, and so is a match whose status is not `ready`.
2. `fps`, `total_episodes`, and `total_frames` must equal the registered
   dataset's values.
3. Every `episodes[].episode_index` must exist for that dataset, and every
   `episodes[].length` must equal the registered `episode.length`.
4. `run.name` must be free for that dataset. A second upload under a name the
   dataset already carries is refused; delete the stored run or rename this
   one. Uploads never overwrite a stored run.

## 6. `grid`

The window geometry the run was trained and evaluated on. It must match the
registered dataset's configuration, otherwise predicted and human labels would
describe different frame pairs.

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `delta_frames` | integer | yes | positive frame gap, matching `dataset.delta_frames` |
| `window_size` | integer | yes | must be `5`, matching `dataset.window_size` |
| `interval_eps` | number | yes | symmetric deadband used to derive interval labels from progress differences, `1e-3` for FluxVLA |
| `no_completion_ceiling` | number | yes | ceiling applied to an episode with an explicit non-completion, `0.95` by default |
| `gt_progress_source` | string | yes | where the ground-truth curves came from, normally the exported `meta/arm_export_manifest.json` |

## 7. `episodes[]`

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `episode_index` | integer | yes | 0-based index within the dataset |
| `split` | string | yes | one of `"train"`, `"val"`, `"test"` |
| `length` | integer | yes | frame count; both progress arrays must have exactly this length |
| `predicted_progress` | array of number | yes | frame-indexed reconstructed prediction in `[0, 1]` |
| `gt_progress` | array of number | yes | frame-indexed human ground truth in `[0, 1]` |
| `intervals` | array of object | yes | see section 7.1; sorted by `target_frame`, unique, may be empty for an episode shorter than one full window |
| `success` | object or null | yes | see section 7.2 |
| `metrics` | object | yes | see section 8 |

Both progress arrays are dense: index `i` is frame `i`, there are no gaps, and
no element may be `null`.

### 7.1 `episodes[].intervals[]`

One entry per evaluated causal window.

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `target_frame` | integer | yes | the window's final frame, the right endpoint of the labeled pair |
| `window_frames` | array of 5 integers | yes | the window's frame indices, oldest first |
| `predicted_label` | integer | yes | interval-head output in `{-1, 0, 1}` |
| `gt_label` | integer or null | yes | the human label for the same pair, or `null` when unlabeled |
| `predicted_probabilities` | array of 3 numbers or null | no | interval-head probabilities ordered `[-1, 0, +1]`, summing to 1 within `1e-4` |

Invariants checked on upload:

- `window_frames == [t - 4Δ, t - 3Δ, t - 2Δ, t - Δ, t]` for `t = target_frame`
  and `Δ = grid.delta_frames`, and `window_frames[0] >= 0`.
- `target_frame < length`, so the whole window lies inside the episode, and
  `target_frame` is a multiple of `Δ`. This is the same grid the annotation
  queue uses: the first target is `4Δ` and targets advance by `Δ`.
- When `gt_label` is not `null` it must equal the deadband of the ground-truth
  progress difference, that is `deadband(gt_progress[t] - gt_progress[t - Δ])`
  with `grid.interval_eps`. The export guarantees this round trip, so a
  mismatch means the curves and the labels came from different sources.

`predicted_label` is the interval head's own output and is deliberately *not*
required to agree with the deadband of `predicted_progress`. Reconstruction
normalizes and anchors the curve, so the two can legitimately differ; keeping
both is what makes the interval agreement strip meaningful.

### 7.2 `episodes[].success`

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `predicted_probability` | number | yes | success-head output in `[0, 1]` |
| `predicted` | boolean | yes | the thresholded prediction the metrics were computed from |
| `gt` | boolean or null | yes | `true` for a marked completion, `false` for an explicit non-completion, `null` when unanswered |
| `gt_frame` | integer or null | yes | the marked completion frame; `null` unless `gt` is `true` |

The whole object is `null` when the run has no success head.

## 8. `episodes[].metrics`

All four required scores compare `predicted_progress` against `gt_progress` over
the full episode, except `interval_accuracy`, which compares labels.

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `frames` | integer | yes | equals `length` |
| `intervals` | integer | yes | number of entries in `intervals` whose `gt_label` is not `null` |
| `spearman` | number or null | yes | Spearman rank correlation, average ranks for ties |
| `pearson` | number or null | yes | Pearson correlation |
| `mae` | number | yes | mean absolute error, `mean(abs(predicted_progress - gt_progress))` |
| `interval_accuracy` | number or null | yes | fraction of windows with a non-null `gt_label` where `predicted_label == gt_label`; `null` when there are none |
| `linear_ramp_baseline` | object or null | no | the same four scores for the baseline of section 10 |

Producers round every score to 6 decimal places. The application stores the
supplied values and does not recompute them, so an artifact whose metrics
disagree with its own arrays will display that disagreement. `frames` and
`intervals` are the exception: they are counts of the artifact's own contents
rather than scores, so upload checks them and refuses a document whose counts
do not match its arrays.

## 9. `aggregate_metrics`

```json
{
  "overall": { "...": "aggregate entry" },
  "by_split": {
    "train": { "...": "aggregate entry" },
    "val": { "...": "aggregate entry" },
    "test": { "...": "aggregate entry" }
  }
}
```

`overall` covers every episode in the artifact. `by_split` has one entry for
each split value that appears in `episodes[].split`, and no entry for a split
with no episodes. Both are required; `by_split` may be an empty object only when
`episodes` is empty, which is itself rejected.

An aggregate entry is:

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `episodes` | integer | yes | episodes in scope |
| `frames` | integer | yes | summed frames in scope |
| `intervals` | integer | yes | summed windows with a non-null `gt_label` |
| `scored_episodes` | integer | yes | episodes contributing a non-null correlation |
| `spearman` | number or null | yes | unweighted mean of the per-episode values, skipping nulls |
| `pearson` | number or null | yes | unweighted mean of the per-episode values, skipping nulls |
| `mae` | number | yes | frame-weighted mean of the per-episode values |
| `interval_accuracy` | number or null | yes | window-weighted mean of the per-episode values |
| `success_f1` | number or null | yes | F1 over episodes in scope with a non-null `success.gt` |
| `linear_ramp_baseline` | object | yes | `spearman`, `pearson`, `mae`, and `interval_accuracy` for the baseline, aggregated by the same rules |

The two averaging modes are deliberate and must not be swapped. Correlations are
macro-averaged so a long episode cannot dominate the ranking; error and accuracy
are micro-averaged so they remain interpretable as per-frame and per-window
rates.

## 10. Linear time-ramp baseline

The baseline is the trivial predictor that assumes progress advances uniformly
with time. For an episode of length `L`:

```text
baseline[i] = i / (L - 1)   for L > 1
baseline[0] = 0.0           for L == 1
```

Its interval labels are derived from that curve with the same deadband, which
makes every label `+1` for `L > 1`. This is intended: the baseline exists so a
reader can tell at a glance how much of a run's correlation comes from episodes
simply progressing over time, and how much of its interval accuracy comes from
the `+1` class being the majority. A run that does not beat this reference has
not learned advantage.

Per-episode baselines are optional, aggregate baselines are required.

## 11. Example

`tests/fixtures/arm_predictions_example.json` is a complete, valid artifact and
is the file the upload tests load. It is deliberately tiny — 3 episodes,
lengths 13, 11, and 9, `delta_frames` 2, six windows in total — but it is not a
stub: every metric in it is the true value computed from its own arrays.

It exercises the cases that break naive parsers:

- one episode per split, so `by_split` has all three keys;
- all three interval labels, including a predicted/ground-truth disagreement;
- both completion answers, including an explicit non-completion capped at
  `no_completion_ceiling`;
- `success_f1` of `null` in the `test` split, which contains no positive ground
  truth and no positive prediction;
- a model that beats the linear ramp on interval accuracy, 0.833 against 0.667.

Regenerate it only alongside a `schema_version` change, and recompute its
metrics rather than editing them by hand.

## 12. Storage

Ingest maps one artifact onto four tables, all created by the additive
migration in `app/db.py` that runs at application startup:

| table | grain | contents |
| --- | --- | --- |
| `prediction_run` | one uploaded artifact | the `run` and `grid` blocks as columns, the `dataset` and `aggregate_metrics` blocks as JSON, plus the resolved `dataset_id`, the uploader, and the artifact's digest and size |
| `prediction_episode` | one episode of one run | `split`, `length`, the `metrics` block, and the flattened `success` object |
| `prediction_frame` | one frame of one episode | `predicted_progress` and `gt_progress` |
| `prediction_interval` | one evaluated window | `predicted_label`, `gt_label`, and the optional probabilities |

Column names follow the artifact's field names, with three deliberate
exceptions:

- `prediction_run.trained_at` holds `run.created_at`, because `created_at` on
  that table is when the row was written;
- `prediction_interval.start_frame` holds `window_frames[0]`, matching the
  `annotation` table's own `start_frame`; the remaining window frames are
  redundant with `target_frame` and `delta_frames` and are not stored;
- `success` is flattened into four nullable columns, where a null
  `success_predicted_probability` means the run had no success head.

`prediction_frame` and `prediction_interval` are `WITHOUT ROWID` tables keyed by
`(run_id, episode_index, frame_index)` and `(run_id, episode_index,
target_frame)`, so one episode's series is a single index range scan. Both also
carry a `(dataset_id, episode_index, …)` index for reading one episode across
runs. Rows are written inside one transaction in batches of
`app.predictions.INSERT_BATCH_ROWS`; the reference run's 58742 frames insert in
well under a second.

Stored metrics are the artifact's own values. The application never recomputes
them, so an artifact whose metrics disagree with its arrays displays that
disagreement rather than hiding it.

### 12.1 Down path

The migration is additive and idempotent: it only issues
`CREATE TABLE IF NOT EXISTS` and `CREATE INDEX IF NOT EXISTS`, it alters no
existing table, and re-running it on an already-migrated database is a no-op.
To reverse it, drop the four tables in foreign-key order:

```sql
DROP TABLE IF EXISTS prediction_interval;
DROP TABLE IF EXISTS prediction_frame;
DROP TABLE IF EXISTS prediction_episode;
DROP TABLE IF EXISTS prediction_run;
```

That statement list is `app.db.PREDICTION_SCHEMA_DOWN`, applied by
`app.db.drop_prediction_tables()`. It discards every stored prediction run and
leaves datasets, episodes, annotations, revision history, completion answers,
and export jobs untouched; a later startup recreates the tables empty. Take a
backup first, because uploaded runs are not recoverable from the database
afterwards.

## 13. Endpoints

Every route below requires the same session cookie as the annotation routes;
there is no unauthenticated write path and no unauthenticated read path.
`app/prediction_artifact.py` is the executable form of sections 2 to 10, and
`app/predictions.py` reads the stored run back.

| Method and path | Purpose |
| --- | --- |
| `POST /api/predictions` | upload one artifact as the raw request body |
| `GET /api/datasets/{dataset_id}/predictions` | runs of one dataset, newest first, each with its `aggregate_metrics` |
| `GET /api/predictions/{run_id}` | one run plus its per-episode `split`, `length`, `metrics`, and `success` |
| `GET /api/predictions/{run_id}/episodes/{episode_index}` | one episode's curves, interval strip, and metrics |
| `DELETE /api/predictions/{run_id}` | discard one run and everything stored under it |

Uploading the example fixture:

```bash
curl -sS --fail-with-body -b cookies.txt \
  -H 'Content-Type: application/json' \
  --data-binary @arm-kuka-assemble-front-r3.arm_predictions.json \
  https://<host>/arm/advantage_annotation/api/predictions
```

A successful upload answers `201` with the stored run. Refusals carry the
reason in `detail` and never a stack trace:

| Status | Meaning |
| --- | --- |
| `401` | no session; the upload path is a write and is authenticated like any other |
| `404` | no dataset is registered for the artifact's `(repo_id, revision, subpath)`, or the run or episode does not exist |
| `409` | the artifact binds to a registered dataset but disagrees with it: `dataset_id` hint, `fps`, `total_episodes`, `total_frames`, `delta_frames`, an unregistered episode, an episode length, a dataset that is still importing, or a run name already in use |
| `413` | the body or its decompressed contents exceed 64 MiB |
| `422` | the document breaks this schema; `detail` names the offending path, for example `episodes.1.intervals.0.window_frames` |

### 13.1 Downsampling the series

An episode of the reference dataset is about a thousand frames, and a compare
chart is a few hundred pixels wide, so the series endpoint thins what it sends.
`max_points` (default `app.predictions.DEFAULT_SERIES_POINTS`, range 50 to
20000) bounds both the curves and the interval strip.

- Frames are bucketed, and each bucket contributes its first frame plus the
  minimum and maximum of *both* curves. A plain stride would flatten a spike
  that falls between two kept frames; keeping bucket extremes preserves the
  envelope a reader compares. The first and last frame are always present, and
  `frame_indices` carries the original frame index of every kept point, so the
  x-axis stays exact.
- The interval strip keeps **every** window whose `predicted_label` differs
  from a non-null `gt_label`. Only agreeing windows are thinned, evenly. A
  disagreement is the whole point of the comparison and is never dropped to
  save bytes.
- The `sampling` block reports `frames`, `frame_points`, `intervals`,
  `interval_points`, `interval_disagreements`, `agreeing_intervals_dropped`,
  and the two `…_downsampled` flags, so a client can state what it is showing
  instead of silently implying it has everything.

Passing a `max_points` at or above the episode length returns every frame.
