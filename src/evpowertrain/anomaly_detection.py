
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Detect DC charging anomalies using ML and engineering rules."""

    data = df.copy()

    # 1. Required engineering features
    features = [
        "current_rate_a_per_min",
        "temp_rate_c_per_min",
        "current_a",
        "battery_temp_c",
    ]

    required = features + [
        "voltage_missing_flag",
        "power_missing_flag",
    ]

    missing = set(required) - set(data.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if len(data) < 20:
        raise ValueError("At least 20 samples are required")

    # 2. Calculate local deviations
    current_baseline = (
        data["current_a"]
        .rolling(window=9, center=True, min_periods=1)
        .median()
    )

    temperature_baseline = (
        data["battery_temp_c"]
        .rolling(window=9, center=True, min_periods=1)
        .median()
    )

    data["current_deviation_a"] = (
        data["current_a"] - current_baseline
    )

    data["temperature_deviation_c"] = (
        data["battery_temp_c"] - temperature_baseline
    )

    # 3. Machine learning anomaly detection
    ml_features = [
        "current_deviation_a",
        "temperature_deviation_c",
        "current_rate_a_per_min",
        "temp_rate_c_per_min",
    ]

    X = data[ml_features].replace(
        [np.inf, -np.inf], np.nan
    )

    if X.isna().any().any():
        raise ValueError("ML features contain invalid values")

    model = IsolationForest(
        n_estimators=200,
        contamination=0.05,
        random_state=42,
    )

    predictions = model.fit_predict(X)

    data["ml_anomaly"] = (predictions == -1).astype(int)
    data["ml_score"] = model.decision_function(X)

    # 4. Engineering rules
    data["rule_current_drop"] = (
        data["current_deviation_a"] < -70
    ).astype(int)

    data["rule_temperature_spike"] = (
        data["temperature_deviation_c"] > 10
    ).astype(int)

    data["rule_sensor_error"] = (
        (
            data["voltage_missing_flag"] == 1
        )
        | (
            data["power_missing_flag"] == 1
        )
    ).astype(int)

    rule_columns = [
        "rule_current_drop",
        "rule_temperature_spike",
        "rule_sensor_error",
    ]

    data["rule_anomaly"] = (
        data[rule_columns].max(axis=1)
    )

    # 5. Combine ML and engineering rules
    data["predicted_anomaly"] = (
        data[["ml_anomaly", "rule_anomaly"]]
        .max(axis=1)
    )

    return data
