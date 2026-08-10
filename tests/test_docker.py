"""Lightweight checks for Docker packaging configuration (no Docker daemon calls)."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _non_comment_lines(text: str) -> list[str]:
    return [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def test_dockerfile_exists_and_sets_runtime_contract() -> None:
    dockerfile = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
    body = "\n".join(_non_comment_lines(dockerfile))
    assert "python:3.13-slim" in dockerfile
    assert "FRAUD_MODEL_DIR=/app/artifacts/xgb-transformed-v1" in body
    assert "PYTHONDONTWRITEBYTECODE=1" in body
    assert "PYTHONUNBUFFERED=1" in body
    assert "EXPOSE 8000" in body
    assert "appuser" in body
    assert "uvicorn" in body and "fraud_risk.api.main:app" in body
    assert "--reload" not in body
    assert "HEALTHCHECK" in body
    assert "/health" in body
    assert "urllib.request" in body
    # Image must bake in the frozen artifact; do not train during build.
    assert "train_final" not in body
    assert "pip install --no-cache-dir ." in body
    assert "[dev]" not in body
    assert "libgomp1" in body


def test_dockerignore_keeps_artifact_and_excludes_heavy_paths() -> None:
    dockerignore = (REPO_ROOT / ".dockerignore").read_text(encoding="utf-8")
    patterns = set(_non_comment_lines(dockerignore))
    for required in (".git", ".venv", "data", "notebooks", "tests", "**/__pycache__"):
        assert required in patterns
    # Must not exclude the frozen artifact path used by the Dockerfile COPY.
    assert "artifacts" not in patterns
    assert "artifacts/" not in patterns
    assert "artifacts/**" not in patterns
    assert "artifacts/xgb-transformed-v1" not in patterns
    assert "artifacts/xgb-transformed-v1/" not in patterns
