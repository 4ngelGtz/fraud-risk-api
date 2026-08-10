"""FastAPI application exposing the frozen FraudPredictor as a local service.

The API is a thin serving layer: request validation, one startup artifact load,
and delegation to ``FraudPredictor.predict_one``. Feature engineering,
calibration, and threshold logic stay in the inference package.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Protocol

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from fraud_risk.api.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
)
from fraud_risk.calibration import DEPLOYMENT_MODEL_VERSION
from fraud_risk.inference import FraudPredictor

MODEL_DIR_ENV: str = "FRAUD_MODEL_DIR"
DEFAULT_MODEL_DIR: Path = Path("artifacts") / DEPLOYMENT_MODEL_VERSION


class PredictorProtocol(Protocol):
    """Minimal surface the API needs from a loaded (or test) predictor."""

    @property
    def model_version(self) -> str: ...

    @property
    def threshold(self) -> float: ...

    @property
    def allowed_transaction_types(self) -> tuple[str, ...]: ...

    @property
    def metadata(self) -> Mapping[str, Any]: ...

    def predict_one(
        self,
        *,
        transaction_type: str,
        amount: float,
        origin_balance: float,
    ) -> dict[str, Any]: ...


def resolve_model_dir(*, env: Mapping[str, str] | None = None) -> Path:
    """Resolve the artifact directory from ``FRAUD_MODEL_DIR`` or the local default.

    The default is the relative path ``artifacts/xgb-transformed-v1`` so local
    development does not hard-code machine-specific absolute paths.
    """
    mapping = os.environ if env is None else env
    configured = mapping.get(MODEL_DIR_ENV)
    if configured:
        return Path(configured)
    return DEFAULT_MODEL_DIR


def _get_predictor(request: Request) -> PredictorProtocol:
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model predictor is not loaded.")
    return predictor


def create_app(
    *,
    predictor: PredictorProtocol | None = None,
    model_dir: Path | str | None = None,
) -> FastAPI:
    """Build the FastAPI app, optionally injecting a predictor for tests.

    When ``predictor`` is omitted, the lifespan loads ``FraudPredictor`` once from
    ``model_dir`` or ``resolve_model_dir()``. Startup fails if the artifact cannot
    be loaded.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if predictor is not None:
            app.state.predictor = predictor
        else:
            artifact_dir = Path(model_dir) if model_dir is not None else resolve_model_dir()
            app.state.predictor = FraudPredictor.load(artifact_dir)
        yield

    app = FastAPI(
        title="Fraud Risk API",
        description=(
            "Portfolio fraud risk scoring API for the frozen xgb-transformed-v1 model. "
            "See docs/inference_contract.md for the public request/response contract."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.exception_handler(ValueError)
    async def value_error_handler(_request: Request, exc: ValueError) -> JSONResponse:
        """Map defensive FraudPredictor validation failures to HTTP 422."""
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        loaded = getattr(request.app.state, "predictor", None) is not None
        return HealthResponse(status="ok", model_loaded=loaded)

    @app.get("/model/info", response_model=ModelInfoResponse)
    def model_info(request: Request) -> ModelInfoResponse:
        pred = _get_predictor(request)
        meta = pred.metadata
        return ModelInfoResponse(
            model_version=str(meta["model_version"]),
            threshold=float(meta["threshold"]),
            allowed_transaction_types=list(meta["allowed_transaction_types"]),
            prediction_moment=str(meta["prediction_moment"]),
        )

    @app.post("/predict", response_model=PredictionResponse)
    def predict(body: PredictionRequest, request: Request) -> PredictionResponse:
        pred = _get_predictor(request)
        result = pred.predict_one(
            transaction_type=body.transaction_type,
            amount=body.amount,
            origin_balance=body.origin_balance,
        )
        return PredictionResponse(**result)

    return app


app = create_app()
