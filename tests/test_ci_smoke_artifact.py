"""Tests for the CI-only smoke artifact generator (no PaySim / real portfolio model)."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest

from fraud_risk.inference import FraudPredictor
from fraud_risk.train_final import (
    CALIBRATOR_FILENAME,
    METADATA_FILENAME,
    MODEL_FILENAME,
    REQUIRED_ARTIFACT_FILES,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "create_ci_smoke_artifact.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("create_ci_smoke_artifact", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def smoke_script():
    return _load_script_module()


def test_ci_smoke_artifact_writes_required_files(tmp_path: Path, smoke_script) -> None:
    out = smoke_script.create_ci_smoke_artifact(tmp_path / "xgb-transformed-v1")
    for name in REQUIRED_ARTIFACT_FILES:
        assert (out / name).is_file()
    assert (out / MODEL_FILENAME).stat().st_size > 0
    assert (out / CALIBRATOR_FILENAME).stat().st_size > 0


def test_ci_smoke_metadata_identifies_ci_smoke_v1(tmp_path: Path, smoke_script) -> None:
    out = smoke_script.create_ci_smoke_artifact(tmp_path / "pkg")
    metadata = json.loads((out / METADATA_FILENAME).read_text(encoding="utf-8"))
    assert metadata["model_version"] == smoke_script.CI_SMOKE_MODEL_VERSION
    assert metadata["model_version"] == "ci-smoke-v1"
    assert metadata["model_version"] != "xgb-transformed-v1"
    assert metadata["threshold"] == smoke_script.CI_SMOKE_THRESHOLD
    assert "purpose" in metadata
    assert "ci" in metadata["purpose"].lower()


def test_fraud_predictor_loads_and_predicts(tmp_path: Path, smoke_script) -> None:
    out = smoke_script.create_ci_smoke_artifact(tmp_path / "pkg")
    predictor = FraudPredictor.load(out)
    assert predictor.model_version == "ci-smoke-v1"

    result = predictor.predict_one(
        transaction_type="TRANSFER",
        amount=1000.0,
        origin_balance=5000.0,
    )
    proba = float(result["fraud_probability"])
    assert math.isfinite(proba)
    assert 0.0 <= proba <= 1.0
    assert result["decision"] in {"pass", "review"}
    assert result["threshold"] == smoke_script.CI_SMOKE_THRESHOLD
    assert result["model_version"] == "ci-smoke-v1"


def test_script_main_writes_to_requested_dir(tmp_path: Path, smoke_script) -> None:
    target = tmp_path / "custom-out"
    assert smoke_script.main(["--artifact-dir", str(target)]) == 0
    assert (target / METADATA_FILENAME).is_file()
    metadata = json.loads((target / METADATA_FILENAME).read_text(encoding="utf-8"))
    assert metadata["model_version"] == "ci-smoke-v1"
