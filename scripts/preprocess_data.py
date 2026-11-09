
from pathlib import Path

from evpowertrain.io import load_test_data
from evpowertrain.preprocessing import preprocess_data


input_path = Path("data/raw/dc_fast_charging.csv")
output_path = Path("data/processed/dc_fast_charging_clean.csv")

# Load raw data
df = load_test_data(input_path)

# Preprocess data
cleaned_df = preprocess_data(df)

# Save processed data
output_path.parent.mkdir(parents=True, exist_ok=True)
cleaned_df.to_csv(output_path, index=False)

print("Preprocessing completed!")
print(f"Input samples: {len(df)}")
print(f"Output samples: {len(cleaned_df)}")

print("\nMissing-value flags:")
print(
    cleaned_df[
        ["voltage_missing_flag", "power_missing_flag"]
    ].sum()
)

print("\nRemaining missing values:")
print(cleaned_df.isna().sum())

print(f"\nSaved to: {output_path}")
