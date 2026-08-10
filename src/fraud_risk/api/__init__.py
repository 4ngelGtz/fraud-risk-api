"""Minimal FastAPI serving layer for the frozen fraud-risk model."""

from fraud_risk.api.main import app, create_app

__all__ = ["app", "create_app"]
