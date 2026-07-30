from __future__ import annotations

import argparse
import gzip
import shutil
import sqlite3
from datetime import UTC, datetime
from pathlib import Path


def backup(source: Path, destination: Path, keep: int = 14) -> Path:
    if not source.is_file():
        raise SystemExit(f"database does not exist: {source}")
    destination.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    plain = destination / f"annotations-{stamp}.db"
    archive = plain.with_suffix(".db.gz")
    with (
        sqlite3.connect(f"file:{source}?mode=ro", uri=True) as src,
        sqlite3.connect(plain) as dst,
    ):
        src.backup(dst)
        if dst.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise SystemExit("backup integrity check failed")
        # `backup()` copies the source's WAL journal mode. Collapse the snapshot
        # back into one self-contained file before compression.
        dst.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        dst.execute("PRAGMA journal_mode=DELETE").fetchone()
    with plain.open("rb") as raw, gzip.open(archive, "wb", compresslevel=6) as compressed:
        shutil.copyfileobj(raw, compressed)
    plain.unlink()
    plain.with_name(f"{plain.name}-wal").unlink(missing_ok=True)
    plain.with_name(f"{plain.name}-shm").unlink(missing_ok=True)
    backups = sorted(destination.glob("annotations-*.db.gz"), reverse=True)
    for old in backups[keep:]:
        old.unlink()
    for pattern in ("annotations-*.db-wal", "annotations-*.db-shm"):
        for orphan in destination.glob(pattern):
            orphan.unlink()
    return archive


def main() -> None:
    parser = argparse.ArgumentParser(description="Online SQLite backup for ARM annotations")
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--keep", type=int, default=14)
    args = parser.parse_args()
    print(backup(args.db, args.out, args.keep))


if __name__ == "__main__":
    main()
