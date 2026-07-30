from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

TEST_STATE = Path(tempfile.mkdtemp(prefix="arm-advantage-annotation-tests-"))
os.environ["ARM_ANNOT_STATE"] = str(TEST_STATE)
os.environ["ARM_ANNOT_DB"] = str(TEST_STATE / "annotations.db")
os.environ["ARM_ANNOT_DATASETS"] = str(TEST_STATE / "datasets")
os.environ["ARM_ANNOT_EXPORTS"] = str(TEST_STATE / "exports")
os.environ["ARM_ANNOT_CACHE"] = str(TEST_STATE / "cache")
os.environ["ARM_ANNOT_MOUNT"] = ""
os.environ["ARM_ANNOT_COOKIE_SECURE"] = "0"
os.environ["ARM_ANNOT_SECRET"] = "test-secret"

from app.db import connect, migrate
from app.main import app


@pytest.fixture()
def client() -> TestClient:
    migrate()
    with connect() as conn:
        for table in (
            "annotation_history",
            "annotation",
            "completion",
            "export_job",
            "episode",
            "dataset",
        ):
            conn.execute(f"DELETE FROM {table}")
    with TestClient(app) as test_client:
        response = test_client.post("/api/login", json={"name": "tester", "password": "arm"})
        assert response.status_code == 200
        yield test_client
