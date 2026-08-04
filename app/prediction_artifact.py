"""Validation for uploaded ARM prediction artifacts.

`docs/model_predictions.md` is the normative contract; this module is its
executable form. `parse` turns the uploaded bytes into a validated artifact
document, and `bind_dataset` resolves the dataset that document claims to
describe, refusing anything whose identity disagrees with what is registered.
Every refusal raises `ArtifactError` with a specific reason and the status code
the upload endpoint should answer with.
"""

from __future__ import annotations

import gzip
import io
import json
import zlib
from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, ValidationError, model_validator

from . import service
from .db import connect

GZIP_MAGIC = b"\x1f\x8b"
MAX_ARTIFACT_BYTES = 64 * 1024 * 1024
PROBABILITY_SUM_TOLERANCE = 1e-4
FPS_TOLERANCE = 1e-6


class ArtifactError(Exception):
    """An artifact that cannot be accepted, with the status code to answer."""

    def __init__(self, message: str, *, status: int = 422) -> None:
        super().__init__(message)
        self.status = status


def deadband(difference: float, interval_eps: float) -> int:
    """Map a progress difference onto the interval label the schema defines."""
    if difference > interval_eps:
        return 1
    if difference < -interval_eps:
        return -1
    return 0


def _reject_bool(value: Any) -> Any:
    # `bool` is a subclass of `int`, so `true` would otherwise pass as the label 1.
    if value is True or value is False:
        raise ValueError("must be an integer, not a boolean")
    return value


def _timestamp(value: str) -> str:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        raise ValueError("must be an RFC 3339 timestamp") from None
    if parsed.tzinfo is None:
        raise ValueError("must carry an explicit UTC offset")
    return value


Label = Annotated[Literal[-1, 0, 1], BeforeValidator(_reject_bool)]
Progress = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]
Correlation = Annotated[float, Field(ge=-1.0, le=1.0, allow_inf_nan=False)]
Rate = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]
NonNegative = Annotated[float, Field(ge=0.0, allow_inf_nan=False)]
Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Timestamp = Annotated[str, BeforeValidator(_timestamp)]
Window = Annotated[list[int], Field(min_length=5, max_length=5)]
Probabilities = Annotated[list[Rate], Field(min_length=3, max_length=3)]


class ArtifactModel(BaseModel):
    """Strict base: unknown keys are refused and values are never coerced."""

    model_config = ConfigDict(extra="forbid", strict=True)


class RunBlock(ArtifactModel):
    name: str = Field(min_length=1, max_length=120)
    checkpoint_path: str = Field(min_length=1)
    checkpoint_sha256: Digest | None
    config_id: str = Field(min_length=1)
    config_path: str | None
    git_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    git_dirty: bool
    training_command: str = Field(min_length=1)
    seed: int | None
    created_at: Timestamp
    split_file: str = Field(min_length=1)
    split_file_sha256: Digest | None
    notes: str | None = None


class DatasetBlock(ArtifactModel):
    repo_id: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    subpath: str
    source_url: str | None
    dataset_id: int | None
    fps: float = Field(gt=0, allow_inf_nan=False)
    total_episodes: int = Field(ge=0)
    total_frames: int = Field(ge=0)


class GridBlock(ArtifactModel):
    delta_frames: int = Field(ge=1)
    window_size: Literal[5]
    interval_eps: float = Field(ge=0, allow_inf_nan=False)
    no_completion_ceiling: Rate
    gt_progress_source: str = Field(min_length=1)


class IntervalBlock(ArtifactModel):
    target_frame: int = Field(ge=0)
    window_frames: Window
    predicted_label: Label
    gt_label: Label | None
    predicted_probabilities: Probabilities | None = None


class BaselineBlock(ArtifactModel):
    spearman: Correlation | None
    pearson: Correlation | None
    mae: NonNegative
    interval_accuracy: Rate | None


class MetricsBlock(ArtifactModel):
    frames: int = Field(ge=0)
    intervals: int = Field(ge=0)
    spearman: Correlation | None
    pearson: Correlation | None
    mae: NonNegative
    interval_accuracy: Rate | None
    linear_ramp_baseline: BaselineBlock | None = None


class SuccessBlock(ArtifactModel):
    predicted_probability: Rate
    predicted: bool
    gt: bool | None
    gt_frame: int | None


class EpisodeBlock(ArtifactModel):
    episode_index: int = Field(ge=0)
    split: Literal["train", "val", "test"]
    length: int = Field(ge=0)
    predicted_progress: list[Progress]
    gt_progress: list[Progress]
    intervals: list[IntervalBlock]
    success: SuccessBlock | None
    metrics: MetricsBlock

    @model_validator(mode="after")
    def check_episode(self) -> EpisodeBlock:
        where = f"episode {self.episode_index}"
        for field, values in (
            ("predicted_progress", self.predicted_progress),
            ("gt_progress", self.gt_progress),
        ):
            if len(values) != self.length:
                raise ValueError(
                    f"{where}: {field} carries {len(values)} values but length is {self.length}"
                )
        if self.metrics.frames != self.length:
            raise ValueError(
                f"{where}: metrics.frames is {self.metrics.frames} but length is {self.length}"
            )
        targets = [interval.target_frame for interval in self.intervals]
        if targets != sorted(set(targets)):
            raise ValueError(f"{where}: intervals must be sorted by unique ascending target_frame")
        labeled = sum(1 for interval in self.intervals if interval.gt_label is not None)
        if self.metrics.intervals != labeled:
            raise ValueError(
                f"{where}: metrics.intervals is {self.metrics.intervals} but {labeled} windows"
                " carry a gt_label"
            )
        if self.success is not None and self.success.gt_frame is not None:
            if self.success.gt is not True:
                raise ValueError(f"{where}: success.gt_frame needs success.gt to be true")
            if not 0 <= self.success.gt_frame < self.length:
                raise ValueError(
                    f"{where}: success.gt_frame {self.success.gt_frame} is outside the episode"
                )
        return self


class AggregateEntry(ArtifactModel):
    episodes: int = Field(ge=0)
    frames: int = Field(ge=0)
    intervals: int = Field(ge=0)
    scored_episodes: int = Field(ge=0)
    spearman: Correlation | None
    pearson: Correlation | None
    mae: NonNegative
    interval_accuracy: Rate | None
    success_f1: Rate | None
    linear_ramp_baseline: BaselineBlock


class AggregateBlock(ArtifactModel):
    overall: AggregateEntry
    by_split: dict[Literal["train", "val", "test"], AggregateEntry]


class ArtifactDocument(ArtifactModel):
    schema_version: Literal[1]
    artifact_kind: Literal["arm_prediction_run"]
    generated_at: Timestamp
    tool: str = Field(min_length=1)
    run: RunBlock
    dataset: DatasetBlock
    grid: GridBlock
    episodes: list[EpisodeBlock] = Field(min_length=1)
    aggregate_metrics: AggregateBlock

    @model_validator(mode="after")
    def check_document(self) -> ArtifactDocument:
        indices = [episode.episode_index for episode in self.episodes]
        if len(set(indices)) != len(indices):
            raise ValueError("episodes repeats an episode_index")
        if indices != sorted(indices):
            raise ValueError("episodes must be sorted by ascending episode_index")
        splits = {episode.split for episode in self.episodes}
        if set(self.aggregate_metrics.by_split) != splits:
            raise ValueError(
                "aggregate_metrics.by_split must have one entry per split present in episodes:"
                f" expected {sorted(splits)}, got {sorted(self.aggregate_metrics.by_split)}"
            )
        for episode in self.episodes:
            for interval in episode.intervals:
                _check_interval(interval, episode=episode, grid=self.grid)
        return self


def _check_interval(interval: IntervalBlock, *, episode: EpisodeBlock, grid: GridBlock) -> None:
    delta = grid.delta_frames
    target = interval.target_frame
    where = f"episode {episode.episode_index} window {target}"
    expected = [target - step * delta for step in (4, 3, 2, 1, 0)]
    if interval.window_frames != expected:
        raise ValueError(
            f"{where}: window_frames must be {expected} for delta_frames {delta},"
            f" not {interval.window_frames}"
        )
    if expected[0] < 0:
        raise ValueError(f"{where}: the causal window starts before the first frame")
    if target >= episode.length:
        raise ValueError(f"{where}: target_frame is outside the episode of length {episode.length}")
    if target % delta:
        raise ValueError(f"{where}: target_frame is off the {delta}-frame annotation grid")
    if interval.predicted_probabilities is not None:
        total = sum(interval.predicted_probabilities)
        if abs(total - 1.0) > PROBABILITY_SUM_TOLERANCE:
            raise ValueError(f"{where}: predicted_probabilities sum to {total:.6f}, not 1")
    if interval.gt_label is not None:
        difference = episode.gt_progress[target] - episode.gt_progress[target - delta]
        derived = deadband(difference, grid.interval_eps)
        if interval.gt_label != derived:
            raise ValueError(
                f"{where}: gt_label {interval.gt_label} disagrees with the ground-truth progress"
                f" difference {difference:+.6f}, whose deadband is {derived}"
            )


def _reject_constant(name: str) -> Any:
    raise ValueError(f"{name} is not a permitted JSON value")


def _describe(error: ValidationError) -> str:
    problems = error.errors()
    first = problems[0]
    location = ".".join(str(part) for part in first["loc"]) or "artifact"
    message = f"{location}: {first['msg']}"
    if len(problems) > 1:
        message += f" (and {len(problems) - 1} further validation problems)"
    return message


def _gunzip(raw: bytes) -> bytes:
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(raw)) as stream:
            payload = stream.read(MAX_ARTIFACT_BYTES + 1)
    except (OSError, EOFError, zlib.error) as error:
        # A damaged deflate stream surfaces as zlib.error rather than an OSError.
        raise ArtifactError(f"artifact is not readable gzip: {error}") from None
    if len(payload) > MAX_ARTIFACT_BYTES:
        raise ArtifactError(
            f"artifact expands past the {MAX_ARTIFACT_BYTES // (1024 * 1024)} MiB ceiling",
            status=413,
        )
    return payload


def decode(raw: bytes) -> dict[str, Any]:
    """Turn uploaded bytes into a JSON object, transparently gunzipping."""
    if not raw:
        raise ArtifactError("the request body is empty")
    if len(raw) > MAX_ARTIFACT_BYTES:
        raise ArtifactError(
            f"artifact exceeds the {MAX_ARTIFACT_BYTES // (1024 * 1024)} MiB ceiling", status=413
        )
    payload = _gunzip(raw) if raw[:2] == GZIP_MAGIC else raw
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError:
        raise ArtifactError("artifact is not UTF-8 text") from None
    try:
        document = json.loads(text, parse_constant=_reject_constant)
    except ValueError as error:
        raise ArtifactError(f"artifact is not valid JSON: {error}") from None
    if not isinstance(document, dict):
        raise ArtifactError("artifact must be a JSON object")
    return document


def validate(document: dict[str, Any]) -> ArtifactDocument:
    try:
        return ArtifactDocument.model_validate(document)
    except ValidationError as error:
        raise ArtifactError(_describe(error)) from None


def parse(raw: bytes) -> dict[str, Any]:
    """Validate uploaded bytes and return the artifact as a plain document."""
    document = decode(raw)
    validate(document)
    return document


def dataset_delta_frames(dataset: dict[str, Any]) -> int:
    """The registered annotation grid, derived for pre-timestep dataset rows."""
    if dataset["delta_frames"] is not None:
        return int(dataset["delta_frames"])
    return max(1, round(float(dataset["fps"]) * float(dataset["delta_seconds"])))


def bind_dataset(document: dict[str, Any]) -> dict[str, Any]:
    """Resolve the registered dataset an artifact describes, or refuse it."""
    block = document["dataset"]
    grid = document["grid"]
    repo_id = block["repo_id"]
    revision = block["revision"]
    subpath = block["subpath"]
    reference = f"repo_id={repo_id!r} revision={revision!r} subpath={subpath!r}"
    with connect(read_only=True) as conn:
        rows = conn.execute(
            "SELECT * FROM dataset WHERE repo_id=? AND revision=? AND subpath=?",
            (repo_id, revision, subpath),
        ).fetchall()
        if not rows:
            raise ArtifactError(
                f"no dataset is registered for {reference}; import it before uploading predictions",
                status=404,
            )
        if len(rows) > 1:
            raise ArtifactError(
                f"{len(rows)} registered datasets match {reference}, so the binding is ambiguous",
                status=409,
            )
        dataset = service.dataset_row(rows[0])
        dataset_id = int(dataset["id"])
        lengths = {
            int(row["episode_index"]): int(row["length"])
            for row in conn.execute(
                "SELECT episode_index,length FROM episode WHERE dataset_id=?", (dataset_id,)
            )
        }
        clash = conn.execute(
            "SELECT id FROM prediction_run WHERE dataset_id=? AND name=?",
            (dataset_id, document["run"]["name"]),
        ).fetchone()
    if dataset["status"] != "ready":
        raise ArtifactError(
            f"dataset {dataset_id} is {dataset['status']}, not ready for predictions", status=409
        )
    hint = block["dataset_id"]
    if hint is not None and int(hint) != dataset_id:
        raise ArtifactError(
            f"artifact dataset_id {hint} does not match dataset {dataset_id}, which is the one"
            f" registered for {reference}",
            status=409,
        )
    if dataset["fps"] is None or abs(float(block["fps"]) - float(dataset["fps"])) > FPS_TOLERANCE:
        raise ArtifactError(
            f"artifact fps {block['fps']} does not match dataset {dataset_id} fps {dataset['fps']}",
            status=409,
        )
    for field, registered in (
        ("total_episodes", int(dataset["total_episodes"])),
        ("total_frames", int(dataset["total_frames"])),
    ):
        if int(block[field]) != registered:
            raise ArtifactError(
                f"artifact {field} {block[field]} does not match dataset {dataset_id},"
                f" which has {registered}",
                status=409,
            )
    registered_delta = dataset_delta_frames(dataset)
    if int(grid["delta_frames"]) != registered_delta:
        raise ArtifactError(
            f"artifact delta_frames {grid['delta_frames']} does not match dataset {dataset_id},"
            f" which is annotated on a {registered_delta}-frame grid",
            status=409,
        )
    for episode in document["episodes"]:
        index = int(episode["episode_index"])
        length = int(episode["length"])
        if index not in lengths:
            raise ArtifactError(
                f"episode {index} is not registered for dataset {dataset_id}", status=409
            )
        if lengths[index] != length:
            raise ArtifactError(
                f"episode {index} is {length} frames in the artifact but {lengths[index]} frames"
                f" in dataset {dataset_id}",
                status=409,
            )
    if clash:
        raise ArtifactError(
            f"dataset {dataset_id} already has a prediction run named"
            f" {document['run']['name']!r} (run {int(clash['id'])}); delete it or rename the run",
            status=409,
        )
    return dataset
