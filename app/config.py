from __future__ import annotations

import os
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
STATE_ROOT = Path(os.environ.get("ARM_ANNOT_STATE", APP_ROOT / "var")).expanduser().resolve()
DB_PATH = Path(os.environ.get("ARM_ANNOT_DB", STATE_ROOT / "annotations.db")).expanduser().resolve()
DATASETS_ROOT = Path(
    os.environ.get("ARM_ANNOT_DATASETS", STATE_ROOT / "datasets")
).expanduser().resolve()
EXPORTS_ROOT = Path(
    os.environ.get("ARM_ANNOT_EXPORTS", STATE_ROOT / "exports")
).expanduser().resolve()
CACHE_ROOT = Path(os.environ.get("ARM_ANNOT_CACHE", STATE_ROOT / "cache")).expanduser().resolve()
STATIC_ROOT = Path(os.environ.get("ARM_ANNOT_STATIC", APP_ROOT / "static")).expanduser().resolve()
MOUNT_PATH = os.environ.get("ARM_ANNOT_MOUNT", "/arm/advantage_annotation").rstrip("/")
COOKIE_SECURE = os.environ.get("ARM_ANNOT_COOKIE_SECURE", "0") == "1"
SESSION_TTL_S = int(os.environ.get("ARM_ANNOT_SESSION_TTL_S", str(30 * 24 * 3600)))
PASSWORD_SHA256 = os.environ.get("ARM_ANNOT_PASSWORD_SHA256", "")
SESSION_SECRET = os.environ.get("ARM_ANNOT_SECRET", "development-only-change-me")
HF_TOKEN = os.environ.get("HF_TOKEN") or None
NO_COMPLETION_CEILING = float(os.environ.get("ARM_NO_COMPLETION_CEILING", "0.95"))


def ensure_state_dirs() -> None:
    for path in (STATE_ROOT, DATASETS_ROOT, EXPORTS_ROOT, CACHE_ROOT):
        path.mkdir(parents=True, exist_ok=True)
