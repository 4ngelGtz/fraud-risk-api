"""Pydantic request/response models for the fraud-risk API.

Field names and semantics match ``docs/inference_contract.md``.
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, Field, field_validator

Decision = Literal["pass", "review"]
TransactionType = Literal["TRANSFER", "CASH_OUT"]


class PredictionRequest(BaseModel):
    """Public scoring request (business / source variables only)."""

    transaction_type: TransactionType
    amount: float = Field(..., description="Transaction amount; finite and >= 0.")
    origin_balance: float = Field(
        ...,
        description="Origin account pre-balance; finite and >= 0.",
    )

    @field_validator("amount", "origin_balance")
    @classmethod
    def _finite_non_negative(cls, value: float) -> float:
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("must be a finite number (rejected NaN / Inf)")
        if number < 0.0:
            raise ValueError("must be >= 0")
        return number


class PredictionResponse(BaseModel):
    """Inference-contract prediction response."""

    fraud_probability: float
    decision: Decision
    threshold: float
    model_version: str


class HealthResponse(BaseModel):
    """Liveness plus confirmation that the predictor loaded at startup."""

    status: str
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    """Selected non-sensitive metadata from the loaded artifact."""

    model_version: str
    threshold: float
    allowed_transaction_types: list[str]
    prediction_moment: str
