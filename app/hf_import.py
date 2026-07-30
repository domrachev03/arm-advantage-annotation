"""Safe Hugging Face import and indexing for LeRobot v3 datasets.

The module intentionally has no database or web-framework dependencies.  It
performs the import in two Hugging Face snapshots:

1. metadata only, followed by structural validation and camera selection;
2. the data shards and selected camera shards, followed by full validation.

LeRobot v3 stores multiple episodes in shared parquet and video files.  The
returned manifest therefore resolves episode records through their relational
chunk/file indices rather than deriving paths from episode indices.
"""

from __future__ import annotations

import json
import math
import re
import string
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any
from urllib.parse import unquote, urlsplit

_REPO_COMPONENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_REVISION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SAFE_PATH_SEGMENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SAFE_CAMERA_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_VERSION_RE = re.compile(r"^v?(?P<major>\d+)(?:\.\d+)*(?:[-+].*)?$")
_ALLOWED_TEMPLATE_FIELDS = frozenset({"chunk_index", "file_index", "video_key"})
_REQUIRED_FRAME_FEATURES = frozenset(
    {"timestamp", "frame_index", "episode_index", "index", "task_index"}
)
_DEFAULT_CAMERA_PREFERENCES = (
    "observation.images.cam_high",
    "observation.images.high",
    "observation.images.front",
    "observation.images.cam_front",
)


class HFImportError(ValueError):
    """Base class for importer input and dataset validation errors."""


class HFIdentifierError(HFImportError):
    """Raised when a Hugging Face dataset identifier is unsafe or malformed."""


class LeRobotV3ValidationError(HFImportError):
    """Raised when a local snapshot does not satisfy the LeRobot v3 contract."""


@dataclass(frozen=True, slots=True)
class HFDatasetReference:
    """Parsed Hugging Face dataset location.

    Attributes:
        repo_id: Canonical ``owner/repo`` Hugging Face repository identifier.
        revision: Optional branch, tag, or commit identifier.
        subfolder: Optional POSIX-relative dataset root within the repository.
    """

    repo_id: str
    revision: str | None = None
    subfolder: str | None = None

    @property
    def metadata_pattern(self) -> str:
        """Return the snapshot allow-pattern for this dataset's metadata."""
        return f"{self._prefix}meta/**"

    @property
    def _prefix(self) -> str:
        return f"{self.subfolder}/" if self.subfolder else ""

    def payload_patterns(self, camera_keys: Sequence[str]) -> tuple[str, ...]:
        """Return selective snapshot patterns for data and camera payloads."""
        return (
            self.metadata_pattern,
            f"{self._prefix}data/**",
            *(f"{self._prefix}videos/{camera_key}/**" for camera_key in camera_keys),
        )


@dataclass(frozen=True, slots=True)
class VideoReference:
    """One episode's slice within a shared LeRobot video shard."""

    camera_key: str
    path: Path
    chunk_index: int
    file_index: int
    from_timestamp: float
    to_timestamp: float


@dataclass(frozen=True, slots=True)
class EpisodeRecord:
    """Indexed episode metadata with resolved data and video locations."""

    episode_index: int
    length: int
    dataset_from_index: int
    dataset_to_index: int
    data_path: Path
    data_chunk_index: int
    data_file_index: int
    task_texts: tuple[str, ...]
    videos: Mapping[str, VideoReference]
    raw: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class LeRobotV3Manifest:
    """Validated, indexed LeRobot v3 dataset snapshot."""

    root: Path
    info: Mapping[str, Any]
    episodes: tuple[EpisodeRecord, ...]
    episodes_by_index: Mapping[int, EpisodeRecord]
    tasks_by_index: Mapping[int, str]
    camera_keys: tuple[str, ...]
    source: HFDatasetReference | None = None


def parse_hf_dataset_reference(identifier: str) -> HFDatasetReference:
    """Parse a safe Hugging Face dataset identifier.

    Accepted forms are:

    - ``owner/repo``
    - ``https://huggingface.co/datasets/owner/repo``
    - ``https://huggingface.co/datasets/owner/repo/tree/revision/subfolder``
    - ``hf://datasets/owner/repo@revision/subfolder``

    Revisions containing slashes are intentionally not inferred from URLs
    because they are ambiguous with subfolders.  Use a tag, commit, or
    slash-free branch name.
    """
    if not isinstance(identifier, str) or not identifier.strip():
        raise HFIdentifierError("Dataset identifier must be a non-empty string.")
    if identifier != identifier.strip():
        raise HFIdentifierError("Dataset identifier must not contain surrounding whitespace.")
    if _contains_control_character(identifier) or "\\" in identifier:
        raise HFIdentifierError("Dataset identifier contains unsafe characters.")

    if "://" not in identifier:
        return HFDatasetReference(repo_id=_validate_repo_id(identifier))

    parsed = urlsplit(identifier)
    try:
        port = parsed.port
    except ValueError as exc:
        raise HFIdentifierError("Dataset URL contains an invalid port.") from exc
    if parsed.query or parsed.fragment or parsed.username or parsed.password or port is not None:
        raise HFIdentifierError(
            "Dataset URL must not contain credentials, ports, queries, or fragments."
        )

    if parsed.scheme == "https":
        return _parse_https_reference(parsed)
    if parsed.scheme == "hf":
        return _parse_hf_uri(parsed)
    raise HFIdentifierError(
        "Only owner/repo, https://huggingface.co, and hf:// identifiers are accepted."
    )


def select_video_keys(
    info: Mapping[str, Any],
    requested: Sequence[str] | str | None = None,
) -> tuple[str, ...]:
    """Select and validate one or more LeRobot video feature keys.

    With no explicit request, a single high/front camera is preferred and the
    lexicographically first video feature is the deterministic fallback.
    """
    features = info.get("features")
    if not isinstance(features, Mapping):
        raise LeRobotV3ValidationError("meta/info.json must contain a features mapping.")

    available = sorted(
        str(key)
        for key, value in features.items()
        if isinstance(value, Mapping) and value.get("dtype") == "video"
    )
    if not available:
        raise LeRobotV3ValidationError("Dataset has no dtype=video feature.")
    for camera_key in available:
        _validate_camera_key(camera_key)

    if requested is None:
        for preferred in _DEFAULT_CAMERA_PREFERENCES:
            if preferred in available:
                return (preferred,)
        for token in ("high", "front"):
            matches = [key for key in available if token in key.lower()]
            if matches:
                return (matches[0],)
        return (available[0],)

    requested_keys = (requested,) if isinstance(requested, str) else tuple(requested)
    if not requested_keys:
        raise LeRobotV3ValidationError("At least one camera must be selected.")

    selected: list[str] = []
    for camera_key in requested_keys:
        _validate_camera_key(camera_key)
        if camera_key not in available:
            raise LeRobotV3ValidationError(
                f"Requested camera {camera_key!r} is not a dtype=video feature."
            )
        if camera_key not in selected:
            selected.append(camera_key)
    return tuple(selected)


def validate_lerobot_v3(
    dataset_root: str | Path,
    *,
    cameras: Sequence[str] | str | None = None,
    require_payload: bool = True,
    source: HFDatasetReference | None = None,
) -> LeRobotV3Manifest:
    """Validate and index a local LeRobot v3 dataset root.

    Args:
        dataset_root: Directory containing ``meta/``, ``data/``, and ``videos/``.
        cameras: Requested video feature key(s), or ``None`` for deterministic
            high/front camera selection.
        require_payload: When false, validate metadata and compute paths before
            the selective data/video download.  When true, require each
            referenced data/video shard and validate data parquet schemas.
        source: Optional parsed Hugging Face provenance attached to the result.

    Returns:
        A fully indexed immutable :class:`LeRobotV3Manifest`.
    """
    root = Path(dataset_root).expanduser()
    if not root.is_dir():
        raise LeRobotV3ValidationError(f"Dataset root does not exist: {root}")
    root = root.absolute()

    meta_root = root / "meta"
    info = _load_json_mapping(meta_root / "info.json", "meta/info.json")
    _validate_info(info)
    _load_json_mapping(meta_root / "stats.json", "meta/stats.json")

    camera_keys = select_video_keys(info, cameras)
    tasks_by_index = _load_tasks(meta_root / "tasks.parquet")
    if int(info["total_tasks"]) != len(tasks_by_index):
        raise LeRobotV3ValidationError(
            "meta/info.json total_tasks does not match task metadata."
        )
    raw_episodes = _load_episode_records(meta_root / "episodes")
    episodes = _index_episodes(
        root,
        info,
        raw_episodes,
        tasks_by_index,
        camera_keys,
        require_payload=require_payload,
    )
    _validate_dataset_totals(info, episodes)

    if require_payload:
        _validate_data_shards(episodes)

    episode_tuple = tuple(episodes)
    by_index = MappingProxyType({episode.episode_index: episode for episode in episode_tuple})
    return LeRobotV3Manifest(
        root=root,
        info=MappingProxyType(dict(info)),
        episodes=episode_tuple,
        episodes_by_index=by_index,
        tasks_by_index=MappingProxyType(dict(tasks_by_index)),
        camera_keys=camera_keys,
        source=source,
    )


def download_lerobot_v3(
    identifier: str | HFDatasetReference,
    *,
    cache_dir: str | Path | None = None,
    cameras: Sequence[str] | str | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
) -> LeRobotV3Manifest:
    """Download selected LeRobot v3 content and return its indexed manifest.

    Metadata is downloaded and validated before data or video payloads.  The
    Hugging Face cache snapshot is never modified.
    """
    reference = (
        parse_hf_dataset_reference(identifier) if isinstance(identifier, str) else identifier
    )
    if not isinstance(reference, HFDatasetReference):
        raise HFIdentifierError("identifier must be a string or HFDatasetReference.")

    common_kwargs: dict[str, Any] = {
        "repo_id": reference.repo_id,
        "repo_type": "dataset",
        "allow_patterns": [reference.metadata_pattern],
        "token": token,
        "local_files_only": local_files_only,
    }
    if reference.revision is not None:
        common_kwargs["revision"] = reference.revision
    if cache_dir is not None:
        common_kwargs["cache_dir"] = str(Path(cache_dir).expanduser())

    try:
        metadata_snapshot = Path(_snapshot_download(**common_kwargs))
    except Exception as exc:  # huggingface_hub exposes several transport errors
        raise HFImportError(f"Failed to download metadata for {reference.repo_id}: {exc}") from exc

    metadata_root = _dataset_root_in_snapshot(metadata_snapshot, reference.subfolder)
    metadata_manifest = validate_lerobot_v3(
        metadata_root,
        cameras=cameras,
        require_payload=False,
        source=reference,
    )

    payload_kwargs = dict(common_kwargs)
    payload_kwargs["allow_patterns"] = list(
        reference.payload_patterns(metadata_manifest.camera_keys)
    )
    try:
        payload_snapshot = Path(_snapshot_download(**payload_kwargs))
    except Exception as exc:
        raise HFImportError(f"Failed to download payload for {reference.repo_id}: {exc}") from exc

    payload_root = _dataset_root_in_snapshot(payload_snapshot, reference.subfolder)
    return validate_lerobot_v3(
        payload_root,
        cameras=metadata_manifest.camera_keys,
        require_payload=True,
        source=reference,
    )


def _snapshot_download(**kwargs: Any) -> str:
    """Late-bound wrapper to keep parser/validator use independent of hub I/O."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - packaging failure path
        raise HFImportError(
            "huggingface_hub is required to download datasets."
        ) from exc
    return snapshot_download(**kwargs)


def _parse_https_reference(parsed: Any) -> HFDatasetReference:
    if parsed.hostname not in {"huggingface.co", "www.huggingface.co"}:
        raise HFIdentifierError("Only huggingface.co dataset URLs are accepted.")

    parts = _decode_url_path(parsed.path)
    if len(parts) < 3 or parts[0] != "datasets":
        raise HFIdentifierError("Expected https://huggingface.co/datasets/owner/repo.")
    repo_id = _validate_repo_id("/".join(parts[1:3]))
    remainder = parts[3:]
    if not remainder:
        return HFDatasetReference(repo_id=repo_id)
    if remainder[0] != "tree" or len(remainder) < 2:
        raise HFIdentifierError("Dataset URLs may only append /tree/revision[/subfolder].")
    revision = _validate_revision(remainder[1])
    subfolder = _validate_subfolder(remainder[2:])
    return HFDatasetReference(repo_id=repo_id, revision=revision, subfolder=subfolder)


def _parse_hf_uri(parsed: Any) -> HFDatasetReference:
    if parsed.netloc != "datasets":
        raise HFIdentifierError("Expected hf://datasets/owner/repo[@revision][/subfolder].")
    parts = _decode_url_path(parsed.path)
    if len(parts) < 2:
        raise HFIdentifierError("Expected hf://datasets/owner/repo.")

    owner = parts[0]
    repo_and_revision = parts[1]
    if repo_and_revision.count("@") > 1:
        raise HFIdentifierError("Dataset repository contains an invalid revision separator.")
    if "@" in repo_and_revision:
        repo, revision_value = repo_and_revision.split("@", 1)
        revision = _validate_revision(revision_value)
    else:
        repo = repo_and_revision
        revision = None
    repo_id = _validate_repo_id(f"{owner}/{repo}")
    subfolder = _validate_subfolder(parts[2:])
    return HFDatasetReference(repo_id=repo_id, revision=revision, subfolder=subfolder)


def _decode_url_path(path: str) -> list[str]:
    if not path:
        return []
    raw_parts = path.split("/")
    if raw_parts and raw_parts[0] == "":
        raw_parts = raw_parts[1:]
    if raw_parts and raw_parts[-1] == "":
        raw_parts = raw_parts[:-1]

    decoded: list[str] = []
    for raw_part in raw_parts:
        part = unquote(raw_part)
        if not part or part in {".", ".."}:
            raise HFIdentifierError("Dataset URL contains an empty or traversal path component.")
        if "/" in part or "\\" in part or _contains_control_character(part):
            raise HFIdentifierError(
                "Dataset URL contains an encoded separator or control character."
            )
        decoded.append(part)
    return decoded


def _validate_repo_id(repo_id: str) -> str:
    parts = repo_id.split("/")
    if len(parts) != 2 or not all(_REPO_COMPONENT_RE.fullmatch(part) for part in parts):
        raise HFIdentifierError("Repository must be a safe owner/repo identifier.")
    if any(part in {".", ".."} or ".." in part for part in parts):
        raise HFIdentifierError("Repository identifier contains traversal syntax.")
    return f"{parts[0]}/{parts[1]}"


def _validate_revision(revision: str) -> str:
    if (
        not revision
        or revision in {".", ".."}
        or ".." in revision
        or not _REVISION_RE.fullmatch(revision)
    ):
        raise HFIdentifierError("Revision must be a slash-free branch, tag, or commit.")
    return revision


def _validate_subfolder(parts: Sequence[str]) -> str | None:
    if not parts:
        return None
    for part in parts:
        if (
            part in {"", ".", ".."}
            or ".." in part
            or not _SAFE_PATH_SEGMENT_RE.fullmatch(part)
        ):
            raise HFIdentifierError("Subfolder contains an unsafe path component.")
    return "/".join(parts)


def _validate_camera_key(camera_key: Any) -> str:
    if not isinstance(camera_key, str) or not _SAFE_CAMERA_RE.fullmatch(camera_key):
        raise LeRobotV3ValidationError(f"Unsafe camera feature key: {camera_key!r}")
    return camera_key


def _contains_control_character(value: str) -> bool:
    return any(ord(character) < 32 or ord(character) == 127 for character in value)


def _dataset_root_in_snapshot(snapshot_root: Path, subfolder: str | None) -> Path:
    root = snapshot_root.absolute()
    dataset_root = root / subfolder if subfolder else root
    if not dataset_root.is_dir():
        raise LeRobotV3ValidationError(
            f"Dataset subfolder was not downloaded or does not exist: {dataset_root}"
        )
    return dataset_root


def _load_json_mapping(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise LeRobotV3ValidationError(f"Missing required {label}.")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LeRobotV3ValidationError(f"Invalid {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise LeRobotV3ValidationError(f"{label} must contain a JSON object.")
    return value


def _load_pyarrow_modules() -> tuple[Any, Any]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - packaging failure path
        raise LeRobotV3ValidationError(
            "pyarrow is required to inspect LeRobot v3 parquet metadata."
        ) from exc
    return pa, pq


def _load_tasks(path: Path) -> dict[int, str]:
    if not path.is_file():
        raise LeRobotV3ValidationError("Missing required meta/tasks.parquet.")
    _, pq = _load_pyarrow_modules()
    try:
        records = pq.read_table(path).to_pylist()
    except Exception as exc:
        raise LeRobotV3ValidationError(f"Invalid meta/tasks.parquet: {exc}") from exc

    tasks: dict[int, str] = {}
    for fallback_index, record in enumerate(records):
        raw_index = record.get("task_index", fallback_index)
        task_index = _require_nonnegative_int(raw_index, "task_index")
        text = _extract_task_text(record)
        if not text:
            raise LeRobotV3ValidationError(
                f"Task record {task_index} has no task text."
            )
        if task_index in tasks:
            raise LeRobotV3ValidationError(f"Duplicate task_index {task_index}.")
        tasks[task_index] = text
    if not tasks:
        raise LeRobotV3ValidationError("meta/tasks.parquet contains no tasks.")
    return tasks


def _extract_task_text(record: Mapping[str, Any]) -> str:
    for key in ("task", "text", "__index_level_0__"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    for key, value in record.items():
        if key != "task_index" and isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _load_episode_records(episodes_root: Path) -> list[dict[str, Any]]:
    if not episodes_root.is_dir():
        raise LeRobotV3ValidationError("Missing required meta/episodes directory.")
    paths = sorted(episodes_root.glob("chunk-*/file-*.parquet"))
    if not paths:
        raise LeRobotV3ValidationError(
            "meta/episodes must contain chunk-*/file-*.parquet."
        )
    _, pq = _load_pyarrow_modules()
    records: list[dict[str, Any]] = []
    for path in paths:
        try:
            records.extend(pq.read_table(path).to_pylist())
        except Exception as exc:
            relative = path.relative_to(episodes_root.parent.parent)
            raise LeRobotV3ValidationError(
                f"Invalid episode metadata parquet {relative}: {exc}"
            ) from exc
    if not records:
        raise LeRobotV3ValidationError("Episode metadata contains no records.")
    return records


def _validate_info(info: Mapping[str, Any]) -> None:
    version = info.get("codebase_version")
    match = _VERSION_RE.fullmatch(str(version)) if version is not None else None
    if match is None or int(match.group("major")) != 3:
        raise LeRobotV3ValidationError(
            f"Expected LeRobot codebase_version major 3, got {version!r}."
        )

    fps = info.get("fps")
    if (
        isinstance(fps, bool)
        or not isinstance(fps, (int, float))
        or not math.isfinite(fps)
        or fps <= 0
    ):
        raise LeRobotV3ValidationError("meta/info.json fps must be a positive number.")

    features = info.get("features")
    if not isinstance(features, Mapping):
        raise LeRobotV3ValidationError("meta/info.json features must be a mapping.")
    missing = sorted(_REQUIRED_FRAME_FEATURES.difference(features))
    if missing:
        raise LeRobotV3ValidationError(
            f"meta/info.json is missing required frame features: {', '.join(missing)}"
        )

    _validate_path_template(info.get("data_path"), {"chunk_index", "file_index"}, "data_path")
    _validate_path_template(
        info.get("video_path"),
        {"video_key", "chunk_index", "file_index"},
        "video_path",
    )

    for field in ("total_episodes", "total_frames", "total_tasks"):
        _require_nonnegative_int(info.get(field), field)


def _validate_path_template(value: Any, required_fields: set[str], label: str) -> None:
    if not isinstance(value, str) or not value:
        raise LeRobotV3ValidationError(f"meta/info.json {label} must be a path template.")
    if "\\" in value or value.startswith("/") or _contains_control_character(value):
        raise LeRobotV3ValidationError(f"meta/info.json {label} is unsafe.")

    fields: set[str] = set()
    try:
        parsed = list(string.Formatter().parse(value))
    except ValueError as exc:
        raise LeRobotV3ValidationError(f"Invalid {label} template: {exc}") from exc
    for _, field_name, format_spec, conversion in parsed:
        if field_name is None:
            continue
        if field_name not in _ALLOWED_TEMPLATE_FIELDS or conversion:
            raise LeRobotV3ValidationError(f"Invalid field in {label} template.")
        if "{" in format_spec or "}" in format_spec:
            raise LeRobotV3ValidationError(f"Nested formatting is not allowed in {label}.")
        fields.add(field_name)
    if fields != required_fields:
        raise LeRobotV3ValidationError(
            f"{label} must contain exactly {sorted(required_fields)}."
        )

    rendered = _format_relative_path(
        value,
        chunk_index=0,
        file_index=0,
        video_key="camera",
    )
    if rendered.suffix not in {".parquet", ".mp4"}:
        raise LeRobotV3ValidationError(f"{label} has an unexpected file extension.")


def _format_relative_path(template: str, **values: Any) -> PurePosixPath:
    try:
        rendered = template.format(**values)
    except (KeyError, ValueError) as exc:
        raise LeRobotV3ValidationError(f"Could not format dataset path template: {exc}") from exc
    path = PurePosixPath(rendered)
    if path.is_absolute() or not path.parts:
        raise LeRobotV3ValidationError("Dataset path template rendered an unsafe path.")
    if any(part in {"", ".", ".."} for part in path.parts) or "\\" in rendered:
        raise LeRobotV3ValidationError("Dataset path template rendered traversal syntax.")
    return path


def _index_episodes(
    root: Path,
    info: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    tasks_by_index: Mapping[int, str],
    camera_keys: Sequence[str],
    *,
    require_payload: bool,
) -> list[EpisodeRecord]:
    data_template = str(info["data_path"])
    video_template = str(info["video_path"])
    episodes: list[EpisodeRecord] = []
    seen_indices: set[int] = set()

    for raw_record in records:
        record = dict(raw_record)
        episode_index = _require_nonnegative_int(
            record.get("episode_index"), "episode_index"
        )
        if episode_index in seen_indices:
            raise LeRobotV3ValidationError(f"Duplicate episode_index {episode_index}.")
        seen_indices.add(episode_index)

        length = _require_positive_int(record.get("length"), "length")
        dataset_from_index = _require_nonnegative_int(
            record.get("dataset_from_index"), "dataset_from_index"
        )
        dataset_to_index = _require_nonnegative_int(
            record.get("dataset_to_index"), "dataset_to_index"
        )
        if dataset_to_index - dataset_from_index != length:
            raise LeRobotV3ValidationError(
                f"Episode {episode_index} length does not match its dataset index range."
            )

        data_chunk_index = _require_nonnegative_int(
            record.get("data/chunk_index"), "data/chunk_index"
        )
        data_file_index = _require_nonnegative_int(
            record.get("data/file_index"), "data/file_index"
        )
        data_relative = _format_relative_path(
            data_template,
            chunk_index=data_chunk_index,
            file_index=data_file_index,
        )
        data_path = root.joinpath(*data_relative.parts)
        if require_payload and not data_path.is_file():
            raise LeRobotV3ValidationError(
                f"Episode {episode_index} references missing data shard {data_relative}."
            )

        videos: dict[str, VideoReference] = {}
        for camera_key in camera_keys:
            prefix = f"videos/{camera_key}"
            chunk_index = _require_nonnegative_int(
                record.get(f"{prefix}/chunk_index"),
                f"{prefix}/chunk_index",
            )
            file_index = _require_nonnegative_int(
                record.get(f"{prefix}/file_index"),
                f"{prefix}/file_index",
            )
            from_timestamp = _require_finite_float(
                record.get(f"{prefix}/from_timestamp"),
                f"{prefix}/from_timestamp",
            )
            to_timestamp = _require_finite_float(
                record.get(f"{prefix}/to_timestamp"),
                f"{prefix}/to_timestamp",
            )
            if from_timestamp < 0 or to_timestamp <= from_timestamp:
                raise LeRobotV3ValidationError(
                    f"Episode {episode_index} has invalid video offsets for {camera_key}."
                )
            video_relative = _format_relative_path(
                video_template,
                video_key=camera_key,
                chunk_index=chunk_index,
                file_index=file_index,
            )
            video_path = root.joinpath(*video_relative.parts)
            if require_payload and not video_path.is_file():
                raise LeRobotV3ValidationError(
                    f"Episode {episode_index} references missing video shard {video_relative}."
                )
            videos[camera_key] = VideoReference(
                camera_key=camera_key,
                path=video_path,
                chunk_index=chunk_index,
                file_index=file_index,
                from_timestamp=from_timestamp,
                to_timestamp=to_timestamp,
            )

        task_texts = _episode_task_texts(record, tasks_by_index)
        if not task_texts:
            raise LeRobotV3ValidationError(
                f"Episode {episode_index} has no resolvable task text."
            )
        episodes.append(
            EpisodeRecord(
                episode_index=episode_index,
                length=length,
                dataset_from_index=dataset_from_index,
                dataset_to_index=dataset_to_index,
                data_path=data_path,
                data_chunk_index=data_chunk_index,
                data_file_index=data_file_index,
                task_texts=task_texts,
                videos=MappingProxyType(videos),
                raw=MappingProxyType(record),
            )
        )

    episodes.sort(key=lambda episode: episode.episode_index)
    _validate_episode_ranges(episodes)
    return episodes


def _episode_task_texts(
    record: Mapping[str, Any], tasks_by_index: Mapping[int, str]
) -> tuple[str, ...]:
    raw_tasks = record.get("tasks")
    if raw_tasks is None:
        raw_tasks = record.get("task_indices")
    if raw_tasks is None and record.get("task_index") is not None:
        raw_tasks = [record["task_index"]]
    if isinstance(raw_tasks, (str, int)):
        raw_tasks = [raw_tasks]
    if not isinstance(raw_tasks, Sequence):
        return ()

    texts: list[str] = []
    for value in raw_tasks:
        if isinstance(value, str):
            text = value.strip()
        elif isinstance(value, int) and not isinstance(value, bool):
            text = tasks_by_index.get(value, "")
        else:
            text = ""
        if text and text not in texts:
            texts.append(text)
    return tuple(texts)


def _validate_episode_ranges(episodes: Sequence[EpisodeRecord]) -> None:
    by_start = sorted(episodes, key=lambda episode: episode.dataset_from_index)
    expected_start = 0
    for episode in by_start:
        if episode.dataset_from_index != expected_start:
            raise LeRobotV3ValidationError(
                "Episode dataset index ranges must be contiguous and begin at zero."
            )
        expected_start = episode.dataset_to_index


def _validate_dataset_totals(
    info: Mapping[str, Any], episodes: Sequence[EpisodeRecord]
) -> None:
    if int(info["total_episodes"]) != len(episodes):
        raise LeRobotV3ValidationError(
            "meta/info.json total_episodes does not match episode metadata."
        )
    total_frames = sum(episode.length for episode in episodes)
    if int(info["total_frames"]) != total_frames:
        raise LeRobotV3ValidationError(
            "meta/info.json total_frames does not match episode lengths."
        )


def _validate_data_shards(episodes: Sequence[EpisodeRecord]) -> None:
    _, pq = _load_pyarrow_modules()
    for path in sorted({episode.data_path for episode in episodes}):
        try:
            names = set(pq.read_schema(path).names)
        except Exception as exc:
            raise LeRobotV3ValidationError(f"Invalid data parquet {path}: {exc}") from exc
        missing = sorted(_REQUIRED_FRAME_FEATURES.difference(names))
        if missing:
            raise LeRobotV3ValidationError(
                f"Data shard {path} is missing columns: {', '.join(missing)}"
            )


def _require_nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise LeRobotV3ValidationError(f"{label} must be a non-negative integer.")
    return value


def _require_positive_int(value: Any, label: str) -> int:
    result = _require_nonnegative_int(value, label)
    if result == 0:
        raise LeRobotV3ValidationError(f"{label} must be positive.")
    return result


def _require_finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LeRobotV3ValidationError(f"{label} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise LeRobotV3ValidationError(f"{label} must be finite.")
    return result


__all__ = [
    "EpisodeRecord",
    "HFDatasetReference",
    "HFIdentifierError",
    "HFImportError",
    "LeRobotV3Manifest",
    "LeRobotV3ValidationError",
    "VideoReference",
    "download_lerobot_v3",
    "parse_hf_dataset_reference",
    "select_video_keys",
    "validate_lerobot_v3",
]
