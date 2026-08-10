"""Unit tests for the FastAPI serving layer (no PaySim / real artifact)."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from fraud_risk.api.main import create_app, resolve_model_dir
from fraud_risk.calibration import (
    DEPLOYMENT_MODEL_VERSION,
    FROZEN_OPERATING_THRESHOLD,
    PREDICTION_MOMENT,
)
from fraud_risk.dataset import MODELING_TYPES


class FakePredictor:
    """Deterministic stand-in for ``FraudPredictor`` used by API unit tests."""

    def __init__(self) -> None:
        self.predict_calls: list[dict[str, Any]] = []
        self.metadata: dict[str, Any] = {
            "model_version": DEPLOYMENT_MODEL_VERSION,
            "threshold": FROZEN_OPERATING_THRESHOLD,
            "prediction_moment": PREDICTION_MOMENT,
            "allowed_transaction_types": list(MODELING_TYPES),
        }

    @property
    def model_version(self) -> str:
        return str(self.metadata["model_version"])

    @property
    def threshold(self) -> float:
        return float(self.metadata["threshold"])

    @property
    def allowed_transaction_types(self) -> tuple[str, ...]:
        return tuple(self.metadata["allowed_transaction_types"])

    def predict_one(
        self,
        *,
        transaction_type: str,
        amount: float,
        origin_balance: float,
    ) -> dict[str, Any]:
        self.predict_calls.append(
            {
                "transaction_type": transaction_type,
                "amount": amount,
                "origin_balance": origin_balance,
            }
        )
        # Deterministic score from public fields only (no feature engineering).
        score = min(0.99, (amount / (origin_balance + 1.0)) * 0.01)
        decision = "review" if score >= self.threshold else "pass"
        return {
            "fraud_probability": float(score),
            "decision": decision,
            "threshold": self.threshold,
            "model_version": self.model_version,
        }


@pytest.fixture
def fake_predictor() -> FakePredictor:
    return FakePredictor()


@pytest.fixture
def client(fake_predictor: FakePredictor) -> TestClient:
    with TestClient(create_app(predictor=fake_predictor)) as test_client:
        yield test_client


def test_resolve_model_dir_uses_env_and_default() -> None:
    assert resolve_model_dir(env={}) == Path("artifacts") / DEPLOYMENT_MODEL_VERSION
    assert resolve_model_dir(env={"FRAUD_MODEL_DIR": "artifacts/custom-v1"}) == Path(
        "artifacts/custom-v1"
    )


def test_health_ok(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_model_info(client: TestClient, fake_predictor: FakePredictor) -> None:
    response = client.get("/model/info")
    assert response.status_code == 200
    payload = response.json()
    assert payload == {
        "model_version": fake_predictor.model_version,
        "threshold": fake_predictor.threshold,
        "allowed_transaction_types": list(fake_predictor.allowed_transaction_types),
        "prediction_moment": PREDICTION_MOMENT,
    }
    # No filesystem paths or training payloads in the public info response.
    assert "artifact_dir" not in payload
    assert "model" not in payload


def test_predict_transfer_ok(client: TestClient) -> None:
    response = client.post(
        "/predict",
        json={
            "transaction_type": "TRANSFER",
            "amount": 8500.0,
            "origin_balance": 9000.0,
        },
    )
    assert response.status_code == 200


def test_predict_cash_out_ok(client: TestClient) -> None:
    response = client.post(
        "/predict",
        json={
            "transaction_type": "CASH_OUT",
            "amount": 1200.0,
            "origin_balance": 5000.0,
        },
    )
    assert response.status_code == 200


def test_predict_response_matches_inference_contract(
    client: TestClient,
    fake_predictor: FakePredictor,
) -> None:
    response = client.post(
        "/predict",
        json={
            "transaction_type": "TRANSFER",
            "amount": 8500.0,
            "origin_balance": 9000.0,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert set(payload.keys()) == {
        "fraud_probability",
        "decision",
        "threshold",
        "model_version",
    }
    assert isinstance(payload["fraud_probability"], float)
    assert payload["decision"] in {"review", "pass"}
    assert payload["threshold"] == fake_predictor.threshold
    assert payload["model_version"] == fake_predictor.model_version


@pytest.mark.parametrize(
    "transaction_type",
    ["PAYMENT", "CASH_IN", "DEBIT", "UNKNOWN", "transfer"],
)
def test_rejects_unsupported_transaction_type(
    client: TestClient,
    transaction_type: str,
) -> None:
    response = client.post(
        "/predict",
        json={
            "transaction_type": transaction_type,
            "amount": 100.0,
            "origin_balance": 200.0,
        },
    )
    assert response.status_code == 422


def test_rejects_payment_explicitly(client: TestClient) -> None:
    response = client.post(
        "/predict",
        json={
            "transaction_type": "PAYMENT",
            "amount": 100.0,
            "origin_balance": 200.0,
        },
    )
    assert response.status_code == 422


def test_rejects_negative_amount(client: TestClient) -> None:
    response = client.post(
        "/predict",
        json={
            "transaction_type": "TRANSFER",
            "amount": -1.0,
            "origin_balance": 200.0,
        },
    )
    assert response.status_code == 422


def test_rejects_negative_origin_balance(client: TestClient) -> None:
    response = client.post(
        "/predict",
        json={
            "transaction_type": "TRANSFER",
            "amount": 100.0,
            "origin_balance": -0.01,
        },
    )
    assert response.status_code == 422


@pytest.mark.parametrize("missing_field", ["transaction_type", "amount", "origin_balance"])
def test_rejects_missing_required_field(client: TestClient, missing_field: str) -> None:
    body: dict[str, Any] = {
        "transaction_type": "TRANSFER",
        "amount": 100.0,
        "origin_balance": 200.0,
    }
    del body[missing_field]
    response = client.post("/predict", json=body)
    assert response.status_code == 422


@pytest.mark.parametrize(
    ("field", "literal"),
    [
        ("amount", "NaN"),
        ("amount", "Infinity"),
        ("amount", "-Infinity"),
        ("origin_balance", "NaN"),
        ("origin_balance", "Infinity"),
        ("origin_balance", "-Infinity"),
    ],
)
def test_rejects_non_finite_values(client: TestClient, field: str, literal: str) -> None:
    # Send raw JSON so non-finite tokens reach Pydantic (httpx refuses to encode them).
    other = "origin_balance" if field == "amount" else "amount"
    content = (
        '{"transaction_type":"TRANSFER",'
        f'"{field}":{literal},'
        f'"{other}":200.0}}'
    )
    response = client.post(
        "/predict",
        content=content,
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422


def test_predictor_not_reloaded_per_request(fake_predictor: FakePredictor) -> None:
    with (
        patch("fraud_risk.api.main.FraudPredictor.load") as load_mock,
        TestClient(create_app(predictor=fake_predictor)) as client,
    ):
        for _ in range(3):
            response = client.post(
                "/predict",
                json={
                    "transaction_type": "TRANSFER",
                    "amount": 10.0,
                    "origin_balance": 20.0,
                },
            )
            assert response.status_code == 200

    load_mock.assert_not_called()
    assert len(fake_predictor.predict_calls) == 3


def test_api_does_not_perform_feature_engineering(
    client: TestClient,
    fake_predictor: FakePredictor,
) -> None:
    response = client.post(
        "/predict",
        json={
            "transaction_type": "CASH_OUT",
            "amount": 250.0,
            "origin_balance": 1000.0,
        },
    )
    assert response.status_code == 200
    assert len(fake_predictor.predict_calls) == 1
    call = fake_predictor.predict_calls[0]
    assert set(call.keys()) == {"transaction_type", "amount", "origin_balance"}
    assert call == {
        "transaction_type": "CASH_OUT",
        "amount": 250.0,
        "origin_balance": 1000.0,
    }
    # Engineered / internal fields must not be computed or forwarded by the API.
    for forbidden in (
        "log_amount",
        "log_origin_balance",
        "origin_balance_zero",
        "oldbalanceOrg",
        "type",
        "amount_to_balance_ratio",
        "amount_exceeds_balance",
    ):
        assert forbidden not in call


def test_predictor_value_error_becomes_422(fake_predictor: FakePredictor) -> None:
    def boom(**_kwargs: Any) -> dict[str, Any]:
        raise ValueError("Unsupported transaction_type 'WEIRD'.")

    fake_predictor.predict_one = boom  # type: ignore[method-assign]
    with TestClient(create_app(predictor=fake_predictor)) as client:
        response = client.post(
            "/predict",
            json={
                "transaction_type": "TRANSFER",
                "amount": 1.0,
                "origin_balance": 1.0,
            },
        )
    assert response.status_code == 422
    assert "Unsupported transaction_type" in response.json()["detail"]
