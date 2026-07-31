"""Safe extraction and discovery of locally uploaded LeRobot v3 ZIP files."""

from __future__ import annotations

import shutil
import stat
import zipfile
from pathlib import Path, PurePosixPath

from .hf_import import LeRobotV3ValidationError

MAX_ARCHIVE_MEMBERS = 100_000
MAX_EXTRACTED_BYTES = 200 * 1024**3


def extract_dataset_zip(archive: Path, destination: Path) -> Path:
    """Extract an archive safely and return its unambiguous LeRobot v3 root."""
    destination.mkdir(parents=True, exist_ok=False)
    total_size = 0
    seen: set[Path] = set()
    try:
        with zipfile.ZipFile(archive) as source:
            members = source.infolist()
            if len(members) > MAX_ARCHIVE_MEMBERS:
                raise LeRobotV3ValidationError("ZIP archive contains too many entries.")
            for member in members:
                relative = _safe_member_path(member)
                if relative is None:
                    continue
                target = destination.joinpath(*relative.parts)
                if target in seen:
                    raise LeRobotV3ValidationError(f"ZIP contains a duplicate path: {member.filename}")
                seen.add(target)
                total_size += member.file_size
                if total_size > MAX_EXTRACTED_BYTES:
                    raise LeRobotV3ValidationError("ZIP expands beyond the 200 GiB safety limit.")
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with source.open(member) as src, target.open("xb") as dst:
                    shutil.copyfileobj(src, dst, length=1024 * 1024)
    except zipfile.BadZipFile as exc:
        raise LeRobotV3ValidationError("Uploaded file is not a valid ZIP archive.") from exc

    candidates = sorted(
        info.parent.parent for info in destination.rglob("meta/info.json") if info.is_file()
    )
    if not candidates:
        raise LeRobotV3ValidationError(
            "ZIP does not contain a LeRobot v3 dataset (meta/info.json was not found)."
        )
    if len(candidates) != 1:
        raise LeRobotV3ValidationError(
            "ZIP contains multiple dataset roots; upload one LeRobot v3 dataset per archive."
        )
    return candidates[0]


def _safe_member_path(member: zipfile.ZipInfo) -> PurePosixPath | None:
    name = member.filename
    if not name or "\\" in name or "\x00" in name:
        raise LeRobotV3ValidationError("ZIP contains an unsafe file name.")
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise LeRobotV3ValidationError(f"ZIP contains an unsafe path: {name}")
    mode = member.external_attr >> 16
    if stat.S_ISLNK(mode):
        raise LeRobotV3ValidationError(f"ZIP links are not allowed: {name}")
    if member.flag_bits & 0x1:
        raise LeRobotV3ValidationError("Encrypted ZIP entries are not supported.")
    if path.name in {".DS_Store"} or "__MACOSX" in path.parts:
        return None
    return path
