
from pathlib import Path

import numpy as np
import pandas as pd


# 1. Simulation configuration
rng = np.random.default_rng(42)

SAMPLES = 120
STEP_SECONDS = 30
BATTERY_CAPACITY_KWH = 60.0
CHARGING_EFFICIENCY = 0.95

soc = 20.0
records = []

start_time = pd.Timestamp("2026-11-08 10:00:00")


# 2. Generate normal charging behavior
for i in range(SAMPLES):

    timestamp = start_time + pd.Timedelta(
        seconds=i * STEP_SECONDS
    )

    # Battery voltage increases with SOC
    voltage = 370 + 0.7 * soc

    # Current tapers at high SOC
    if soc < 80:
        current = 200.0
    else:
        current = max(35.0, 200 - 11 * (soc - 80))

    current = max(0.0, current + rng.normal(0, 2))

    battery_temp = (
        25 + 0.1 * (soc - 20) + rng.normal(0, 0.25)
    )

    anomaly_type = "normal"

    # 3. Inject current-drop anomaly
    if 32 <= i <= 34:
        current *= 0.25
        anomaly_type = "current_drop"

    # 4. Inject temperature sensor anomaly
    if 75 <= i <= 76:
        battery_temp += 18
        anomaly_type = "temperature_spike"

    #stop charging when the battery is full
    if soc >= 100.0:
        current = 0.0
        
    # Electrical charging power in kW
    power_kw = voltage * current / 1000

    records.append({
        "timestamp": timestamp,
        "soc_pct": soc,
        "voltage_v": voltage,
        "current_a": current,
        "power_kw": power_kw,
        "battery_temp_c": battery_temp,
        "ambient_temp_c": 22.0,
        "is_anomaly": int(anomaly_type != "normal"),
        "anomaly_type": anomaly_type,
    })

    # Update SOC using charged energy
    energy_kwh = (
        power_kw * STEP_SECONDS / 3600
        * CHARGING_EFFICIENCY
    )

    soc += energy_kwh / BATTERY_CAPACITY_KWH * 100
    soc = min(soc, 100.0)


# 5. Create DataFrame
df = pd.DataFrame(records)

# Inject missing sensor data
df.loc[100, ["voltage_v", "power_kw"]] = np.nan
df.loc[100, "is_anomaly"] = 1
df.loc[100, "anomaly_type"] = "sensor_error"


# 6. Save synthetic CSV file
output_path = Path("data/raw/dc_fast_charging.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(output_path, index=False)

print("Synthetic DC charging data generated!")
print(f"Saved to: {output_path}")
print(f"Total samples: {len(df)}")
print("\nAnomaly distribution:")
print(df["anomaly_type"].value_counts())
print("\nFirst five rows:")
print(df.head())
