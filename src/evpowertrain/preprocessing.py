
import pandas as pd


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess DC fast charging test data."""

    # 1. Keep the original data unchanged
    cleaned = df.copy()

    if cleaned.empty:
        raise ValueError("Input data is empty")

    # 2. Sort by timestamp
    cleaned = cleaned.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    # 3. Mark missing sensor measurements
    cleaned["voltage_missing_flag"] = (
        cleaned["voltage_v"].isna().astype(int)
    )

    cleaned["power_missing_flag"] = (
        cleaned["power_kw"].isna().astype(int)
    )

    # 4. Interpolate missing voltage values
    cleaned["voltage_v"] = (
        cleaned["voltage_v"].interpolate(
            method="linear",
            limit_direction="both"
        )
    )

    # 5. Recalculate missing charging power
    calculated_power = (
        cleaned["voltage_v"]
        * cleaned["current_a"]
        / 1000
    )

    cleaned["power_kw"] = (
        cleaned["power_kw"].fillna(calculated_power)
    )

    # 6. Calculate elapsed time in seconds
    cleaned["elapsed_seconds"] = (
        cleaned["timestamp"] - cleaned["timestamp"].iloc[0]
    ).dt.total_seconds()

    return cleaned
