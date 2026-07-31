# ARM Advantage Annotation

A paper-faithful web annotation platform for
[ARM: Advantage Reward Modeling for Long-Horizon Manipulation](https://arxiv.org/abs/2604.03037).
It imports arbitrary Hugging Face datasets that follow the LeRobot dataset v3
layout, presents causal five-frame comparison windows, records tri-state
advantage labels, and exports data consumable by FluxVLA.

## What it provides

- LeRobot v3 imports from a repository ID, Hugging Face dataset URL, or local ZIP archive.
- Configurable temporal delta, resolved to frames using the dataset FPS.
- Five visible frames: `t - 4Δ`, `t - 3Δ`, `t - 2Δ`, `t - Δ`, and `t`.
- Paper-style labels on the final transition: regressive (`-1`), stagnant
  (`0`), or progressive (`+1`).
- An alternative video-editor-style progress curve mode with per-episode keypoints,
  frame scrubbing, and linear per-frame interpolation.
- A separate episode-completion target: first successful frame or explicit
  non-completion.
- Fast keyboard annotation with rapid sequences or paced holds on `Z`, `X`,
  and `C`.
- Undo with retained revision history, transition labels in the visible
  history, frame/sample prefetching, and non-looping episode navigation.
- Immediate transactional autosave to SQLite.
- Non-destructive LeRobot v3 exports with raw ARM pairs and FluxVLA-compatible
  dense progress.

## Annotation and export flow

```text
Hugging Face LeRobot v3 dataset
              |
              v
 selective metadata, parquet, and camera download
              |
              v
 five-frame ARM decisions + episode completion
              |
              v
 SQLite annotations and revision history
              |
              v
 new LeRobot v3 export
   ├── meta/arm_pairs.parquet  (raw -1/0/+1 decisions)
   └── progress               (dense float32 feature for FluxVLA)
```

The source dataset is never modified. Each export is written to a separate
directory. Video handling can use symlinks, copies, or metadata-only mode.

## Quick start

Requirements:

- Linux or macOS
- `git`
- `ffmpeg` available on `PATH`
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/)
- Network access to Hugging Face for imports

```bash
git clone https://github.com/domrachev03/arm-advantage-annotation.git
cd arm-advantage-annotation
uv sync --locked --dev

ARM_ANNOT_STATE="$PWD/var" \
ARM_ANNOT_MOUNT= \
ARM_ANNOT_COOKIE_SECURE=0 \
uv run uvicorn app.main:app --host 127.0.0.1 --port 8120
```

Open <http://127.0.0.1:8120>. Development login uses password `arm` only when
`ARM_ANNOT_PASSWORD_SHA256` is unset. Never expose that fallback configuration
to a network.

From the UI:

1. Select **Import dataset**.
2. Enter `owner/repository`, a full Hugging Face dataset URL, or select **ZIP archive** and
   choose a local `.zip` file. The archive may contain the dataset at its root or in one wrapper
   directory, but must contain exactly one LeRobot v3 dataset.
3. Optionally select a revision, repository subdirectory, and camera keys.
4. Choose delta seconds and wait for indexing to finish.
5. Annotate with `Z`, `X`, and `C`; record completion with `F` or `Shift+F`.
6. Export after every required transition and completion answer is present.

Alternatively, select **Progress curve** in the top bar. Scrub through each episode, add
increased, decreased, or unchanged keypoints, and save a curve for every episode. The first and
last frames are always endpoints; exported `progress` values between keypoints are linearly
interpolated as float32 values. In this mode `Z`, `X`, and `C` add decreased, unchanged, and
increased keypoints; arrow keys scrub frames, `B`/`N` change episodes, `S` saves, and `U` removes
the selected non-endpoint keypoint.

Private or gated datasets require a read token in `HF_TOKEN`. See the
[Hugging Face authentication guide](https://huggingface.co/docs/huggingface_hub/quick-start#authentication).

## Keyboard workflow

| key | action |
| --- | --- |
| `Z` | save `-1` / regressive |
| `X` | save `0` / stagnant |
| `C` | save `+1` / progressive |
| hold `Z`, `X`, or `C` | continuously label, paced by visible frame readiness |
| `F` | mark the current frame as the first completed frame |
| `Shift+F` | record that the episode never completes |
| `U` | undo the most recently saved label |
| `B` | previous transition |
| `N` | next transition or immediate next episode |

A hold keeps at most one choice ahead of the visible sample. It stops at an
episode change so unseen frames in the next episode are not labeled
accidentally.

## Persistence model

There is no draft layer or separate save button. Every advantage decision is
committed by an authenticated API request inside a SQLite transaction before
the annotation pipeline advances. The UI shows `Autosaving` until the server
acknowledges the write and warns before closing a tab with pending work.

Relabels and undo operations retain superseded values in
`annotation_history`. Completion answers are transactional upserts. Back up
both the database and the imported dataset directory; the database contains
labels and provenance, while dataset directories contain the indexed source
material needed for frame rendering and exports.

## Development

```bash
uv sync --locked --dev
uv run pytest -q
uv run ruff check app tests
```

The test suite uses isolated temporary state and does not touch `var/`.

Repository layout:

```text
app/                  FastAPI API, import, media, persistence, and export logic
static/               dependency-free browser application
tests/                annotation, import, media, and exporter regression tests
tools/backup_db.py    online SQLite backup utility
deploy/systemd/       portable user-service and backup units
docs/RUN.md           production installation and self-hosting guide
```

## Production deployment

Read the comprehensive [installation and self-hosting guide](docs/RUN.md). It
covers:

- host preparation and locked dependency installation;
- passwords, session signing, private Hugging Face access, and file
  permissions;
- a persistent systemd user service;
- HTTPS at a domain root or stripped subpath;
- Tailscale Serve/Funnel and conventional reverse proxies;
- backups, restoration, upgrades, rollback, monitoring, and troubleshooting.

See also the [browser API contract](static/API_CONTRACT.md) and
[implementation notes](docs/IMPLEMENTATION_PLAN.md).

## Compatibility notes

- The importer intentionally validates the LeRobot v3 metadata, parquet, and
  video layout. Earlier LeRobot formats should be converted to v3 first.
- The exporter creates `meta/arm_pairs.parquet` for lossless raw labels and a
  dense float32 `progress` feature for the current FluxVLA ARM training path.
- Frame rendering requires FFmpeg and caches JPEGs under `ARM_ANNOT_CACHE`.
- SQLite is appropriate for one host and modest concurrent annotation. Do not
  place the live database on NFS or run multiple application workers against
  separate local copies.

## Acknowledgements

Built around the ARM paper, the LeRobot v3 dataset convention, and FluxVLA's
ARM data-consumption format. This repository is an independent annotation
tool; refer to the respective upstream projects and papers for their licenses
and citation requirements.
