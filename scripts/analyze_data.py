
from pathlib import Path

from evpowertrain.io import load_test_data
from evpowertrain.metrics import calculate_metrics


input_path = Path(
    "data/processed/dc_fast_charging_clean.csv"
)

output_path = Path(
    "data/processed/dc_fast_charging_features.csv"
)

# 1. Load preprocessed data
df = load_test_data(input_path)

# 2. Calculate engineering metrics
result = calculate_metrics(df)

# 3. Save results
output_path.parent.mkdir(parents=True, exist_ok=True)
result.to_csv(output_path, index=False)

# 4. Print summary
print("Engineering analysis completed!")

print(f"Total samples: {len(result)}")

print(
    "Charging duration (min):",
    round(result["time_step_s"].sum() / 60, 2)
)

print(
    "Total DC energy (kWh):",
    round(result["cumulative_energy_kwh"].iloc[-1], 2)
)

print(
    "Maximum power (kW):",
    round(result["power_kw"].max(), 2)
)

print(
    "Maximum battery temperature (C):",
    round(result["battery_temp_c"].max(), 2)
)

print("\nAnomaly-related engineering features:")

print(
    result.loc[
        result["anomaly_type"] != "normal",
        [
            "timestamp",
            "anomaly_type",
            "current_rate_a_per_min",
            "temp_rate_c_per_min",
        ],
    ].to_string(index=False)
)

print(f"\nSaved to: {output_path}")
