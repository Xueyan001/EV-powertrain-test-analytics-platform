
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from evpowertrain.io import load_test_data
from evpowertrain.preprocessing import preprocess_data
from evpowertrain.metrics import calculate_metrics
from evpowertrain.anomaly_detection import detect_anomalies


# -------------------------------------------------
# 1. Page configuration
# -------------------------------------------------
st.set_page_config(
    page_title="EV Charging Test Analytics",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ DC Fast Charging Test Analytics")
st.caption(
    "EV Charging Data Analysis & Anomaly Detection Platform"
)


# -------------------------------------------------
# 2. Load CSV data
# -------------------------------------------------
st.sidebar.header("Data Source")

uploaded_file = st.sidebar.file_uploader(
    "Upload DC charging CSV",
    type=["csv"],
)

project_root = Path(__file__).resolve().parent

default_path = (
    project_root / "data/raw/dc_fast_charging.csv"
)

try:
    if uploaded_file is not None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / "uploaded.csv"
            tmp_path.write_bytes(uploaded_file.getvalue())
            raw_df = load_test_data(tmp_path)

        st.sidebar.success("Uploaded CSV loaded")
    else:
        raw_df = load_test_data(default_path)
        st.sidebar.info("Using synthetic sample data")

    # -------------------------------------------------
    # 3. Run the existing processing pipeline
    # -------------------------------------------------
    clean_df = preprocess_data(raw_df)

    essential_columns = [
        "voltage_v",
        "current_a",
        "power_kw",
        "soc_pct",
        "battery_temp_c",
    ]

    if clean_df[essential_columns].isna().any().any():
        raise ValueError(
            "Unresolved missing values remain after preprocessing."
        )

    features_df = calculate_metrics(clean_df)
    result = detect_anomalies(features_df)

except (OSError, ValueError, KeyError) as error:
    st.error(f"Data processing failed: {error}")
    st.stop()


# -------------------------------------------------
# 4. Engineering KPIs
# -------------------------------------------------
st.subheader("Engineering Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Maximum Power",
        f"{result['power_kw'].max():.1f} kW",
    )

with col2:
    st.metric(
        "DC Energy",
        f"{result['cumulative_energy_kwh'].iloc[-1]:.2f} kWh",
    )

with col3:
    st.metric(
        "Maximum Battery Temp.",
        f"{result['battery_temp_c'].max():.1f} °C",
    )

with col4:
    st.metric(
        "Detected Samples",
        int(result["predicted_anomaly"].sum()),
    )


# -------------------------------------------------
# 5. Interactive charging curves
# -------------------------------------------------
st.subheader("Charging Curves & Anomalies")

fig = make_subplots(
    rows=4,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.06,
    subplot_titles=[
        "Charging Current",
        "Battery Temperature",
        "DC Charging Power",
        "Battery SOC",
    ],
)

signals = [
    ("current_a", "Current (A)"),
    ("battery_temp_c", "Temperature (°C)"),
    ("power_kw", "Power (kW)"),
    ("soc_pct", "SOC (%)"),
]

anomalies = result[result["predicted_anomaly"] == 1]

for row, (column, label) in enumerate(signals, start=1):

    fig.add_trace(
        go.Scatter(
            x=result["timestamp"],
            y=result[column],
            mode="lines",
            name="Measured / Synthetic Signal",
            line=dict(color="royalblue"),
            showlegend=(row == 1),
        ),
        row=row,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=anomalies["timestamp"],
            y=anomalies[column],
            mode="markers",
            name="Predicted Anomaly",
            marker=dict(color="red", size=9),
            showlegend=(row == 1),
        ),
        row=row,
        col=1,
    )

    fig.update_yaxes(
        title_text=label,
        row=row,
        col=1,
    )

fig.update_layout(
    height=950,
    hovermode="x unified",
    template="plotly_white",
    margin=dict(t=60, b=40),
)

fig.update_xaxes(title_text="Time", row=4, col=1)

st.plotly_chart(fig, width="stretch")


# -------------------------------------------------
# 6. Detection summary
# -------------------------------------------------
st.subheader("Anomaly Detection Summary")

a, b, c = st.columns(3)

a.metric(
    "ML Detected",
    int(result["ml_anomaly"].sum()),
)

b.metric(
    "Rule Detected",
    int(result["rule_anomaly"].sum()),
)

c.metric(
    "Combined Detected",
    int(result["predicted_anomaly"].sum()),
)

# Evaluate only when synthetic labels are available
if "is_anomaly" in result.columns:

    actual = result["is_anomaly"] == 1
    predicted = result["predicted_anomaly"] == 1

    tp = int((actual & predicted).sum())
    fp = int((~actual & predicted).sum())
    fn = int((actual & ~predicted).sum())

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    st.write(
        f"**Ground-truth evaluation:** "
        f"Precision = {precision:.1%} | "
        f"Recall = {recall:.1%} | "
        f"TP = {tp}, FP = {fp}, FN = {fn}"
    )


# -------------------------------------------------
# 7. Anomaly details
# -------------------------------------------------
st.subheader("Detected Anomaly Details")

display_columns = [
    "timestamp",
    "soc_pct",
    "current_a",
    "battery_temp_c",
    "ml_anomaly",
    "rule_anomaly",
    "predicted_anomaly",
]

if "anomaly_type" in result.columns:
    display_columns.insert(1, "anomaly_type")

st.dataframe(
    anomalies[display_columns],
    width="stretch",
)


# -------------------------------------------------
# 8. Download reports
# -------------------------------------------------
st.subheader("Export Results")

csv_bytes = result.to_csv(index=False).encode("utf-8-sig")

st.download_button(
    label="Download Analysis CSV",
    data=csv_bytes,
    file_name="dc_charging_analysis_results.csv",
    mime="text/csv",
)

html_report = fig.to_html(
    full_html=True,
    include_plotlyjs=True,
)

st.download_button(
    label="Download Interactive Chart (HTML)",
    data=html_report,
    file_name="dc_charging_charts.html",
    mime="text/html",
)
