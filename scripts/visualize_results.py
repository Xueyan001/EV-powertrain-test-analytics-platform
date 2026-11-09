
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from evpowertrain.io import load_test_data


# 1. Load anomaly detection results
df = load_test_data(
    "data/processed/dc_fast_charging_anomalies.csv"
)

# 2. Identify correct detections and false alarms
true_positive = df[
    (df["predicted_anomaly"] == 1)
    & (df["is_anomaly"] == 1)
]

false_positive = df[
    (df["predicted_anomaly"] == 1)
    & (df["is_anomaly"] == 0)
]

# 3. Create engineering plots
fig = make_subplots(
    rows=3,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.08,
    subplot_titles=[
        "DC Charging Current",
        "Battery Temperature",
        "Battery SOC",
    ],
)

signals = [
    ("current_a", "Current (A)"),
    ("battery_temp_c", "Temperature (C)"),
    ("soc_pct", "SOC (%)"),
]

for row, (column, label) in enumerate(signals, start=1):

    # Normal signal curve
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df[column],
            mode="lines",
            name=label,
            line=dict(color="royalblue"),
            showlegend=(row == 1),
        ),
        row=row,
        col=1,
    )

    # Correct anomaly detections
    fig.add_trace(
        go.Scatter(
            x=true_positive["timestamp"],
            y=true_positive[column],
            mode="markers",
            name="True Positive",
            marker=dict(color="red", size=10),
            showlegend=(row == 1),
        ),
        row=row,
        col=1,
    )

    # False positive detections
    fig.add_trace(
        go.Scatter(
            x=false_positive["timestamp"],
            y=false_positive[column],
            mode="markers",
            name="False Positive",
            marker=dict(
                color="orange",
                size=12,
                symbol="x",
            ),
            showlegend=(row == 1),
        ),
        row=row,
        col=1,
    )

    fig.update_yaxes(title_text=label, row=row, col=1)

# 4. Configure layout
fig.update_layout(
    title="DC Fast Charging - Anomaly Detection",
    height=850,
    template="plotly_white",
    hovermode="x unified",
)

fig.update_xaxes(
    title_text="Time",
    row=3,
    col=1,
)

# 5. Export interactive HTML
output_path = Path("outputs/charging_anomaly_report.html")
output_path.parent.mkdir(parents=True, exist_ok=True)

fig.write_html(output_path, include_plotlyjs=True)

print("Visualization completed!")
print(f"True positives: {len(true_positive)}")
print(f"False positives: {len(false_positive)}")
print(f"Saved to: {output_path}")
