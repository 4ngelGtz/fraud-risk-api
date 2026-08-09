"""Raw PaySim dataset loading and schema validation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

EXPECTED_COLUMNS: tuple[str, ...] = (
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
    "isFlaggedFraud",
)


def _default_raw_dir() -> Path:
    """Resolve ``data/raw`` relative to the repository root."""
    # src/fraud_risk/data.py -> parents[2] == repo root
    return Path(__file__).resolve().parents[2] / "data" / "raw"


def resolve_raw_csv(raw_dir: Path | str | None = None) -> Path:
    """Return the single PaySim CSV path under ``raw_dir``.

    Raises
    ------
    FileNotFoundError
        If the directory is missing or contains no CSV files.
    ValueError
        If more than one CSV file is present.
    """
    directory = Path(raw_dir) if raw_dir is not None else _default_raw_dir()
    if not directory.is_dir():
        raise FileNotFoundError(
            f"Raw data directory not found: {directory}. "
            "Create data/raw/ and place the PaySim CSV there (see data/README.md)."
        )

    csv_files = sorted(directory.glob("*.csv"))
    if len(csv_files) == 0:
        raise FileNotFoundError(
            f"No CSV files found in {directory}. "
            "Download the PaySim dataset manually and place exactly one .csv file there "
            "(see data/README.md)."
        )
    if len(csv_files) > 1:
        names = ", ".join(path.name for path in csv_files)
        raise ValueError(
            f"Multiple CSV files found in {directory}: {names}. "
            "Keep exactly one PaySim CSV (or pass an explicit path) to resolve the ambiguity."
        )
    return csv_files[0]


def validate_paysim_schema(df: pd.DataFrame) -> None:
    """Ensure ``df`` contains all expected PaySim columns."""
    missing = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            "PaySim schema validation failed. Missing required columns: "
            f"{missing}. Expected columns: {list(EXPECTED_COLUMNS)}."
        )


def load_raw_data(
    raw_dir: Path | str | None = None,
    *,
    csv_path: Path | str | None = None,
) -> pd.DataFrame:
    """Load and validate the raw PaySim CSV.

    Parameters
    ----------
    raw_dir:
        Directory containing exactly one ``.csv`` file. Defaults to ``<repo>/data/raw``.
    csv_path:
        Optional explicit CSV path. When set, ``raw_dir`` discovery is skipped.

    Returns
    -------
    pandas.DataFrame
        Raw PaySim table with expected columns present.
    """
    path = Path(csv_path) if csv_path is not None else resolve_raw_csv(raw_dir)
    if not path.is_file():
        raise FileNotFoundError(
            f"CSV file not found: {path}. Place the PaySim dataset under data/raw/ "
            "(see data/README.md)."
        )

    df = pd.read_csv(path)
    validate_paysim_schema(df)
    return df
