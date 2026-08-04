# Installation and self-hosting

This guide installs ARM Advantage Annotation as a persistent single-host
service. Commands target a current Debian/Ubuntu host with systemd user
services; adapt package-manager commands for other Linux distributions.

## 1. Deployment model

The application is one FastAPI process listening on loopback:

```text
browser
  |
  | HTTPS
  v
Tailscale Funnel / Caddy / nginx
  |
  | HTTP on 127.0.0.1:8120
  v
ARM Advantage Annotation
  ├── SQLite database
  ├── imported LeRobot datasets
  ├── decoded-frame cache
  └── generated exports
```

Use one application worker. Frame decoding is already concurrent and SQLite
serializes writes; multiple workers add complexity without improving the
annotation workflow.

## 2. Host requirements

Recommended minimum:

- x86-64 or ARM64 Linux;
- 2 CPU cores and 4 GB RAM;
- enough disk for selected camera videos plus exports and cache;
- outbound HTTPS access to `huggingface.co`;
- `git`, `curl`, and `ffmpeg`;
- a reverse proxy or Tailscale for HTTPS.

On Debian or Ubuntu:

```bash
sudo apt update
sudo apt install -y ca-certificates curl ffmpeg git
ffmpeg -version
```

Install `uv` using the
[official installer](https://docs.astral.sh/uv/getting-started/installation/)
or your operating-system package manager:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
exec "$SHELL" -l
uv --version
```

## 3. Install the application

The included systemd units assume this clone location:

```bash
mkdir -p "$HOME/src"
git clone https://github.com/domrachev03/arm-advantage-annotation.git \
  "$HOME/src/arm-advantage-annotation"
cd "$HOME/src/arm-advantage-annotation"
uv sync --locked
uv run pytest -q
```

`uv sync --locked` creates `.venv` and refuses dependency resolution that
would change `uv.lock`. If you clone elsewhere, update `WorkingDirectory` and
`ExecStart` in the systemd units before installing them.

## 4. Create persistent state

Keep runtime data outside the Git checkout:

```bash
install -d -m 700 "$HOME/.local/share/arm-advantage-annotation"
install -d -m 700 "$HOME/arm-advantage-annotation-backups"
install -d -m 700 "$HOME/.config"
```

Default state layout:

| path below `ARM_ANNOT_STATE` | purpose | backup priority |
| --- | --- | --- |
| `annotations.db` | labels, revisions, completion answers, dataset index, uploaded model prediction runs | critical |
| `datasets/` | downloaded LeRobot metadata, parquet, and selected videos | critical for rendering/export |
| `exports/` | generated FluxVLA/LeRobot v3 outputs | optional if reproducible |
| `cache/` | decoded JPEG frames | disposable |

Size the filesystem for the selected camera videos. Copy-mode exports can
temporarily require another full video copy; symlink-mode exports consume much
less space but are not portable away from the host.

## 5. Configure authentication and storage

Generate a session signing secret:

```bash
python3 -c 'import secrets; print(secrets.token_hex(32))'
```

Generate a SHA-256 digest from a long, random shared password:

```bash
python3 -c \
  'import getpass,hashlib; print(hashlib.sha256(getpass.getpass("Password: ").encode()).hexdigest())'
```

Copy the example and replace the absolute state path and generated values:

```bash
install -m 600 .env.example "$HOME/.config/arm-advantage-annotation.env"
"${EDITOR:-vi}" "$HOME/.config/arm-advantage-annotation.env"
chmod 600 "$HOME/.config/arm-advantage-annotation.env"
```

For a root-domain deployment:

```text
ARM_ANNOT_STATE=/home/YOUR_USER/.local/share/arm-advantage-annotation
ARM_ANNOT_MOUNT=
ARM_ANNOT_COOKIE_SECURE=1
ARM_ANNOT_SECRET=<generated secret>
ARM_ANNOT_PASSWORD_SHA256=<generated password digest>
```

For a public path such as `/arm/advantage_annotation/`, set:

```text
ARM_ANNOT_MOUNT=/arm/advantage_annotation
```

The proxy must strip that prefix before forwarding to the local application.
`ARM_ANNOT_MOUNT` controls cookie scope and generated deployment metadata; it
does not make FastAPI serve a second nested route.

For private or gated datasets, add a least-privilege Hugging Face read token:

```text
HF_TOKEN=hf_...
```

Do not put the environment file, token, password, database, dataset media, or
exports in Git. `HF_TOKEN` takes precedence over locally cached Hugging Face
credentials.

### Environment reference

| variable | default | description |
| --- | --- | --- |
| `ARM_ANNOT_STATE` | repository `var/` | root for all mutable state |
| `ARM_ANNOT_DB` | `$STATE/annotations.db` | SQLite database |
| `ARM_ANNOT_DATASETS` | `$STATE/datasets` | imported Hugging Face snapshots |
| `ARM_ANNOT_EXPORTS` | `$STATE/exports` | completed exports |
| `ARM_ANNOT_CACHE` | `$STATE/cache` | disposable decoded-frame cache |
| `ARM_ANNOT_MOUNT` | `/arm/advantage_annotation` | external cookie path; empty for root |
| `ARM_ANNOT_COOKIE_SECURE` | `0` | set `1` behind HTTPS |
| `ARM_ANNOT_SECRET` | unsafe development value | session HMAC secret |
| `ARM_ANNOT_PASSWORD_SHA256` | unset | SHA-256 digest of shared login password |
| `ARM_ANNOT_SESSION_TTL_S` | `2592000` | session lifetime in seconds |
| `HF_TOKEN` | unset | Hugging Face read token |
| `ARM_NO_COMPLETION_CEILING` | `0.95` | exported progress ceiling for unsuccessful episodes |

## 6. Install the systemd user service

```bash
install -Dm644 deploy/systemd/arm-advantage-annotation.service \
  "$HOME/.config/systemd/user/arm-advantage-annotation.service"
install -Dm644 deploy/systemd/arm-advantage-annotation-backup.service \
  "$HOME/.config/systemd/user/arm-advantage-annotation-backup.service"
install -Dm644 deploy/systemd/arm-advantage-annotation-backup.timer \
  "$HOME/.config/systemd/user/arm-advantage-annotation-backup.timer"

systemctl --user daemon-reload
systemctl --user enable --now arm-advantage-annotation.service
systemctl --user enable --now arm-advantage-annotation-backup.timer
```

To keep user services running after logout:

```bash
sudo loginctl enable-linger "$USER"
```

Verify startup:

```bash
systemctl --user status arm-advantage-annotation.service --no-pager
curl -fsS http://127.0.0.1:8120/api/healthz
journalctl --user -u arm-advantage-annotation.service -n 100 --no-pager
```

Do not bind Uvicorn directly to a public interface. Keep it on
`127.0.0.1` and terminate TLS at a maintained reverse proxy.

## 7. Publish with HTTPS

Choose one of the following exposure models.

### Tailscale Serve: tailnet-only

This is appropriate when only devices in your tailnet should annotate:

```bash
tailscale serve --bg --set-path=/arm/advantage_annotation \
  http://127.0.0.1:8120
tailscale serve status
```

Set:

```text
ARM_ANNOT_MOUNT=/arm/advantage_annotation
ARM_ANNOT_COOKIE_SECURE=1
```

### Tailscale Funnel: public internet

Funnel exposes the route publicly through the node's `*.ts.net` hostname:

```bash
tailscale funnel --bg --set-path=/arm/advantage_annotation \
  http://127.0.0.1:8120
tailscale funnel status
```

Follow the current
[Tailscale Funnel documentation](https://tailscale.com/docs/reference/tailscale-cli/funnel)
for account requirements and allowed public ports. Keep the application's
shared password enabled even if Tailscale provides the HTTPS endpoint.

### Caddy at a dedicated hostname

For root hosting at `annotate.example.org`, set `ARM_ANNOT_MOUNT=` and use:

```caddyfile
annotate.example.org {
    reverse_proxy 127.0.0.1:8120
}
```

Caddy obtains and renews TLS certificates when DNS and firewall configuration
permit it.

For subpath hosting:

```caddyfile
example.org {
    handle_path /arm/advantage_annotation/* {
        reverse_proxy 127.0.0.1:8120
    }
}
```

`handle_path` strips the prefix, so set
`ARM_ANNOT_MOUNT=/arm/advantage_annotation`.

### nginx at a subpath

```nginx
location /arm/advantage_annotation/ {
    proxy_http_version 1.1;
    proxy_set_header Host $host;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
    proxy_pass http://127.0.0.1:8120/;
}
```

The trailing slash on `proxy_pass` strips the public prefix. Configure TLS at
nginx and set both `ARM_ANNOT_MOUNT=/arm/advantage_annotation` and
`ARM_ANNOT_COOKIE_SECURE=1`.

## 8. First-run verification

Check the external health endpoint:

```bash
curl -fsS https://YOUR_HOST/arm/advantage_annotation/api/healthz
```

Then use a browser to:

1. sign in with a non-empty annotator name and the shared password;
2. import a small public LeRobot v3 dataset;
3. confirm all five frames render;
4. save one label and undo it;
5. reload the page and confirm the queue resumes from server state.

Watch logs during the smoke test:

```bash
journalctl --user -u arm-advantage-annotation.service -f
```

## 9. Backups and restoration

The included timer creates an online, integrity-checked, compressed SQLite
snapshot every night and retains the newest 14 archives:

```bash
systemctl --user list-timers arm-advantage-annotation-backup.timer
systemctl --user start arm-advantage-annotation-backup.service
ls -lh "$HOME/arm-advantage-annotation-backups"
```

The SQLite database alone preserves annotations and audit history. Also back
up `datasets/` if you need disaster recovery without downloading and indexing
the source datasets again. `cache/` can be excluded. Back up `exports/` only
when they cannot be regenerated.

To restore a database:

```bash
systemctl --user stop arm-advantage-annotation.service
cp "$HOME/.local/share/arm-advantage-annotation/annotations.db" \
  "$HOME/.local/share/arm-advantage-annotation/annotations.db.pre-restore"
gzip -dc "$HOME/arm-advantage-annotation-backups/annotations-TIMESTAMP.db.gz" \
  > "$HOME/.local/share/arm-advantage-annotation/annotations.db.restored"
python3 -c \
  'import sqlite3; p="'"$HOME"'/.local/share/arm-advantage-annotation/annotations.db.restored"; print(sqlite3.connect(p).execute("PRAGMA integrity_check").fetchone()[0])'
mv "$HOME/.local/share/arm-advantage-annotation/annotations.db.restored" \
  "$HOME/.local/share/arm-advantage-annotation/annotations.db"
systemctl --user start arm-advantage-annotation.service
```

Proceed only when the integrity command prints `ok`. The
`.pre-restore` copy is a local rollback point.

## 10. Upgrades

Review upstream changes, then update without rewriting local history:

```bash
cd "$HOME/src/arm-advantage-annotation"
git fetch --prune
git pull --ff-only
uv sync --locked
uv run pytest -q
systemctl --user restart arm-advantage-annotation.service
curl -fsS http://127.0.0.1:8120/api/healthz
```

Database migrations run at startup and are designed to be additive. Take a
fresh backup before upgrading. Record the previously deployed commit:

```bash
git rev-parse HEAD
```

If application rollback is necessary, stop the service, restore the previous
Git commit and matching database backup, run `uv sync --locked`, and restart.

Rolling back to a commit that predates model prediction runs needs no database
change, because the prediction tables are additive and older code ignores them.
Section 12.1 of `docs/model_predictions.md` documents how to drop them if you
want the older schema exactly.

## 11. Operations and troubleshooting

### Service does not start

```bash
systemctl --user status arm-advantage-annotation.service --no-pager
journalctl --user -u arm-advantage-annotation.service -n 200 --no-pager
```

Common causes are a wrong clone path in the unit, an unreadable environment
file, or a missing `.venv`.

### Login works over HTTP but not HTTPS/subpath

- Use `ARM_ANNOT_COOKIE_SECURE=1` for HTTPS.
- Use an empty `ARM_ANNOT_MOUNT` at the domain root.
- At a subpath, set the exact public prefix with no trailing slash.
- Ensure the proxy strips the prefix before forwarding.
- Restart the application after changing environment variables.

### Frames fail to render

```bash
command -v ffmpeg
ffmpeg -version
df -h
```

Check that the dataset video files remain under the indexed dataset root and
that the service user can read them and write to the cache. Detailed decode
errors appear in the service journal.

### Import fails

- Confirm the repository is a LeRobot v3 dataset, not a model repository.
- Check revision and optional subdirectory spelling.
- For private/gated data, verify `HF_TOKEN` has read access.
- Confirm sufficient free disk space and outbound HTTPS access.

### Annotation appears slow

The browser prefetches sample metadata and future frames; the server decodes up
to four distinct frames concurrently. First access is decode-bound, while
subsequent access should use the JPEG cache. Faster local storage and avoiding
remote filesystems materially improve performance.

### Uploading a model prediction run

Prediction artifacts are uploaded over the same authenticated API the browser
uses, so log in once and reuse the cookie:

```bash
BASE=https://<host>/arm/advantage_annotation
curl -sS --fail-with-body -c cookies.txt -H 'Content-Type: application/json' \
  -d '{"name":"<your name>","password":"<shared password>"}' "$BASE/api/login"
curl -sS --fail-with-body -b cookies.txt -H 'Content-Type: application/json' \
  --data-binary @<run>.arm_predictions.json "$BASE/api/predictions"
```

The dataset is resolved from the artifact itself, which must bind to a dataset
that is already imported and ready. A refusal explains what disagreed; the
status codes are listed in section 13 of `docs/model_predictions.md`. A run
name is unique per dataset, so re-uploading a corrected artifact means deleting
the stored run first with `DELETE /api/predictions/<run_id>`.

Once stored, the run is visible in the browser under **Compare with model** in
the dataset sidebar, which charts the predicted progress curve against the
human ground truth per episode and summarises the run against a linear time
ramp. Section 14 of `docs/model_predictions.md` describes what it shows.

### Database health

Use the supplied online backup tool rather than copying a live WAL database:

```bash
cd "$HOME/src/arm-advantage-annotation"
.venv/bin/python tools/backup_db.py \
  --db "$HOME/.local/share/arm-advantage-annotation/annotations.db" \
  --out "$HOME/arm-advantage-annotation-backups" \
  --keep 14
```

Never publish runtime state or credentials. The repository `.gitignore`
excludes common state locations, but deployment security still depends on
correct filesystem permissions and secret management.
