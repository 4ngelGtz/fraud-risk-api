#!/usr/bin/env python3
"""Bounded HTTP smoke checks for the fraud-risk-api Docker CI job.

Transport-layer failures are retried within a fixed budget. Semantic failures
(non-200 responses, invalid JSON, assertion mismatches) fail immediately.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.error
import urllib.request
from typing import Any

DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_MODEL_VERSION = "ci-smoke-v1"
DEFAULT_RETRIES = 8
DEFAULT_TIMEOUT_S = 5.0
DEFAULT_RETRY_DELAY_S = 0.5

# Explicit transport failures. ConnectionResetError may surface from the socket
# layer without being wrapped as urllib.error.URLError.
_TRANSPORT_ERRORS: tuple[type[BaseException], ...] = (
    urllib.error.URLError,
    ConnectionError,
    ConnectionResetError,
    ConnectionRefusedError,
    TimeoutError,
)

_PREDICT_BODY = {
    "transaction_type": "TRANSFER",
    "amount": 1000.0,
    "origin_balance": 5000.0,
}


def request_json(
    url: str,
    *,
    method: str = "GET",
    body: dict[str, Any] | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    retries: int = DEFAULT_RETRIES,
    retry_delay_s: float = DEFAULT_RETRY_DELAY_S,
) -> tuple[int, dict[str, Any]]:
    """Perform an HTTP request and return ``(status, json_object)``.

    Retries only transient transport errors. Does not retry HTTPError,
    JSONDecodeError, or TypeError from a non-object JSON body.
    """
    data: bytes | None = None
    headers = {"Accept": "application/json"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    last_error: BaseException | None = None

    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                status = int(response.status)
                raw = response.read().decode("utf-8")
            # Semantic: invalid JSON / non-object payload must not be retried.
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise TypeError(f"expected JSON object from {url}, got {type(payload).__name__}")
            return status, payload
        except urllib.error.HTTPError:
            # A response was received; treat as semantic (do not retry).
            raise
        except _TRANSPORT_ERRORS as exc:
            last_error = exc
            if attempt >= retries:
                break
            print(
                f"transport retry {attempt}/{retries} for {method} {url}: {exc!r}",
                file=sys.stderr,
            )
            time.sleep(retry_delay_s)

    assert last_error is not None
    raise RuntimeError(
        f"transport failed after {retries} attempts for {method} {url}: {last_error!r}"
    ) from last_error


def assert_health(payload: dict[str, Any]) -> None:
    assert payload.get("status") == "ok", payload
    assert payload.get("model_loaded") is True, payload


def assert_model_info(payload: dict[str, Any], *, expected_model_version: str) -> None:
    assert payload.get("model_version") == expected_model_version, payload


def assert_predict(payload: dict[str, Any], *, expected_model_version: str) -> None:
    assert "fraud_probability" in payload, payload
    proba = float(payload["fraud_probability"])
    assert math.isfinite(proba), payload
    assert 0.0 <= proba <= 1.0, payload
    assert payload.get("decision") in {"pass", "review"}, payload
    assert payload.get("model_version") == expected_model_version, payload


def run_smoke(
    base_url: str,
    *,
    expected_model_version: str = DEFAULT_MODEL_VERSION,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    retries: int = DEFAULT_RETRIES,
    retry_delay_s: float = DEFAULT_RETRY_DELAY_S,
) -> None:
    root = base_url.rstrip("/")

    status, health = request_json(
        f"{root}/health",
        timeout_s=timeout_s,
        retries=retries,
        retry_delay_s=retry_delay_s,
    )
    assert status == 200, (status, health)
    assert_health(health)
    print("health ok:", health)

    status, info = request_json(
        f"{root}/model/info",
        timeout_s=timeout_s,
        retries=retries,
        retry_delay_s=retry_delay_s,
    )
    assert status == 200, (status, info)
    assert_model_info(info, expected_model_version=expected_model_version)
    print("model/info ok:", info)

    status, pred = request_json(
        f"{root}/predict",
        method="POST",
        body=_PREDICT_BODY,
        timeout_s=timeout_s,
        retries=retries,
        retry_delay_s=retry_delay_s,
    )
    assert status == 200, (status, pred)
    assert_predict(pred, expected_model_version=expected_model_version)
    print("predict ok:", pred)
    print("docker-smoke assertions passed")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument(
        "--expected-model-version",
        default=DEFAULT_MODEL_VERSION,
        help="Expected model_version (CI default: ci-smoke-v1)",
    )
    parser.add_argument("--timeout-s", type=float, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES)
    parser.add_argument("--retry-delay-s", type=float, default=DEFAULT_RETRY_DELAY_S)
    args = parser.parse_args(argv)

    run_smoke(
        args.base_url,
        expected_model_version=args.expected_model_version,
        timeout_s=args.timeout_s,
        retries=args.retries,
        retry_delay_s=args.retry_delay_s,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
