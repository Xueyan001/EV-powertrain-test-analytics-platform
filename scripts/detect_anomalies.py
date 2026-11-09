
from pathlib import Path

from evpowertrain.io import load_test_data
from evpowertrain.anomaly_detection import detect_anomalies


input_path = Path(
    "data/processed/dc_fast_charging_features.csv"
)

output_path = Path(
    "data/processed/dc_fast_charging_anomalies.csv"
)

# 1. Load engineering features
df = load_test_data(input_path)

# 2. Run anomaly detection
result = detect_anomalies(df)

# 3. Save detection results
output_path.parent.mkdir(parents=True, exist_ok=True)
result.to_csv(output_path, index=False)

# 4. Summarize model predictions
print("Anomaly detection completed!")
print(f"Total samples: {len(result)}")

print("ML anomalies:", result["ml_anomaly"].sum())
print("Rule anomalies:", result["rule_anomaly"].sum())
print("Combined anomalies:", result["predicted_anomaly"].sum())

# 5. Compare predictions with synthetic ground truth
actual = result["is_anomaly"] == 1
predicted = result["predicted_anomaly"] == 1

tp = int((actual & predicted).sum())
fp = int((~actual & predicted).sum())
fn = int((actual & ~predicted).sum())

precision = tp / (tp + fp) if tp + fp else 0
recall = tp / (tp + fn) if tp + fn else 0

print("\nEvaluation:")
print(f"True Positives: {tp}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")

print("\nDetected anomalies:")
print(
    result.loc[
        predicted,
        [
            "timestamp",
            "anomaly_type",
            "ml_anomaly",
            "rule_anomaly",
            "predicted_anomaly",
        ],
    ].to_string(index=False)
)

print(f"\nSaved to: {output_path}")
