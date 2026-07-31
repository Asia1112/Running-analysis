"""CSV schema validation and persistence helpers."""

from pathlib import Path

import pandas as pd

from .config import DEFAULT_CONFIG, ValidationConfig


def validate_extracted_schema(data: pd.DataFrame, config: ValidationConfig = DEFAULT_CONFIG) -> None:
    """Raise a clear error when required extracted-data columns are absent."""
    missing = [column for column in config.required_columns if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def write_csv(data: pd.DataFrame, path: str | Path) -> Path:
    """Write a DataFrame as UTF-8 CSV and return the resolved output path."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output, index=False, encoding="utf-8")
    return output


def read_csv(path: str | Path) -> pd.DataFrame:
    """Read a UTF-8 CSV into a DataFrame."""
    return pd.read_csv(path)
