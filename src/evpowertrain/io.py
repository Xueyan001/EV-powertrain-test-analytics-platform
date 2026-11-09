
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = [
    "timestamp",
    "soc_pct",
    "voltage_v",
    "current_a",
    "power_kw",
    "battery_temp_c",
    "ambient_temp_c",
]

NUMERIC_COLUMNS = [
    "soc_pct",
    "voltage_v",
    "current_a",
    "power_kw",
    "battery_temp_c",
    "ambient_temp_c",
]


def load_test_data(file_path: str | Path) -> pd.DataFrame:
    """Load and validate DC fast charging test data."""

    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)

    missing = set(REQUIRED_COLUMNS) - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="raise")

    return df
