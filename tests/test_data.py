"""Tests for PaySim raw data loading and schema validation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from fraud_risk.data import EXPECTED_COLUMNS, load_raw_data, resolve_raw_csv, validate_paysim_schema


def _sample_frame(**overrides: object) -> pd.DataFrame:
    base = {
        "step": [1, 2],
        "type": ["PAYMENT", "TRANSFER"],
        "amount": [100.0, 250.5],
        "nameOrig": ["C1", "C2"],
        "oldbalanceOrg": [1000.0, 500.0],
        "newbalanceOrig": [900.0, 249.5],
        "nameDest": ["M1", "C3"],
        "oldbalanceDest": [0.0, 100.0],
        "newbalanceDest": [100.0, 350.5],
        "isFraud": [0, 1],
        "isFlaggedFraud": [0, 0],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def _write_csv(directory: Path, name: str, df: pd.DataFrame) -> Path:
    path = directory / name
    df.to_csv(path, index=False)
    return path


def test_load_raw_data_success(tmp_path: Path) -> None:
    _write_csv(tmp_path, "paysim_sample.csv", _sample_frame())

    df = load_raw_data(tmp_path)

    assert list(df.columns) == list(EXPECTED_COLUMNS)
    assert len(df) == 2
    assert df["isFraud"].tolist() == [0, 1]


def test_load_raw_data_missing_columns(tmp_path: Path) -> None:
    incomplete = _sample_frame()
    incomplete = incomplete.drop(columns=["newbalanceDest", "isFlaggedFraud"])
    _write_csv(tmp_path, "incomplete.csv", incomplete)

    with pytest.raises(ValueError, match="Missing required columns"):
        load_raw_data(tmp_path)


def test_resolve_raw_csv_no_csv(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No CSV files found"):
        resolve_raw_csv(tmp_path)


def test_load_raw_data_no_csv(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No CSV files found"):
        load_raw_data(tmp_path)


def test_resolve_raw_csv_multiple_csvs(tmp_path: Path) -> None:
    frame = _sample_frame()
    _write_csv(tmp_path, "a.csv", frame)
    _write_csv(tmp_path, "b.csv", frame)

    with pytest.raises(ValueError, match="Multiple CSV files found"):
        resolve_raw_csv(tmp_path)


def test_validate_paysim_schema_ok() -> None:
    validate_paysim_schema(_sample_frame())


def test_load_raw_data_explicit_csv_path(tmp_path: Path) -> None:
    path = _write_csv(tmp_path, "explicit.csv", _sample_frame())
    # Ambiguous directory should be ignored when csv_path is provided.
    _write_csv(tmp_path, "other.csv", _sample_frame())

    df = load_raw_data(tmp_path, csv_path=path)
    assert len(df) == 2
