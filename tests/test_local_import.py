from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from app.hf_import import LeRobotV3ValidationError
from app.local_import import extract_dataset_zip


def _write_zip(path: Path, entries: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in entries.items():
            archive.writestr(name, content)
    return path


def test_extract_dataset_zip_discovers_wrapped_root(tmp_path: Path) -> None:
    archive = _write_zip(
        tmp_path / "dataset.zip",
        {
            "my-dataset/meta/info.json": b"{}",
            "my-dataset/meta/stats.json": b"{}",
            "my-dataset/data/file.parquet": b"payload",
        },
    )

    root = extract_dataset_zip(archive, tmp_path / "extracted")

    assert root == tmp_path / "extracted" / "my-dataset"
    assert (root / "data/file.parquet").read_bytes() == b"payload"


@pytest.mark.parametrize("unsafe_name", ["../escape", "/absolute", "folder\\file"])
def test_extract_dataset_zip_rejects_unsafe_paths(tmp_path: Path, unsafe_name: str) -> None:
    archive = _write_zip(
        tmp_path / "unsafe.zip",
        {unsafe_name: b"bad", "meta/info.json": b"{}"},
    )

    with pytest.raises(LeRobotV3ValidationError, match="unsafe"):
        extract_dataset_zip(archive, tmp_path / "extracted")


def test_extract_dataset_zip_rejects_multiple_dataset_roots(tmp_path: Path) -> None:
    archive = _write_zip(
        tmp_path / "multiple.zip",
        {"one/meta/info.json": b"{}", "two/meta/info.json": b"{}"},
    )

    with pytest.raises(LeRobotV3ValidationError, match="multiple dataset roots"):
        extract_dataset_zip(archive, tmp_path / "extracted")
