
import pandas as pd


def calculate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate DC charging engineering metrics."""

    data = df.copy()

    if data.empty:
        raise ValueError("Input data is empty")

    required = [
        "timestamp",
        "voltage_v",
        "current_a",
        "power_kw",
        "soc_pct",
        "battery_temp_c",
    ]

    missing = set(required) - set(data.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    # 1. Calculate sampling time intervals
    delta_t = data["timestamp"].diff().dt.total_seconds()

    if (delta_t.iloc[1:] <= 0).any():
        raise ValueError("Timestamps must be strictly increasing")

    data["time_step_s"] = delta_t.fillna(0.0)

    # 2. Calculate DC power from voltage and current
    data["power_calculated_kw"] = (
        data["voltage_v"] * data["current_a"] / 1000
    )

    # Compare reported and calculated power
    data["power_difference_kw"] = (
        data["power_kw"] - data["power_calculated_kw"]
    )

    # 3. Calculate energy using trapezoidal integration
    previous_power = data["power_kw"].shift(1)

    data["energy_step_kwh"] = (
        (data["power_kw"] + previous_power)
        / 2
        * data["time_step_s"]
        / 3600
    ).fillna(0.0)

    data["cumulative_energy_kwh"] = (
        data["energy_step_kwh"].cumsum()
    )

    # 4. Calculate change rates per minute
    delta_min = delta_t / 60

    data["current_rate_a_per_min"] = (
        data["current_a"].diff() / delta_min
    )

    data["temp_rate_c_per_min"] = (
        data["battery_temp_c"].diff() / delta_min
    )

    data["soc_rate_pct_per_min"] = (
        data["soc_pct"].diff() / delta_min
    )

    # First sample has no previous sample
    rate_columns = [
        "current_rate_a_per_min",
        "temp_rate_c_per_min",
        "soc_rate_pct_per_min",
    ]

    data[rate_columns] = data[rate_columns].fillna(0.0)

    return data
