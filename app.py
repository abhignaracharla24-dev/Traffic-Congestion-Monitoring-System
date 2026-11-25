
import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from pathlib import Path

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="FlowOps | City Traffic Operations",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset" / "traffic_congestion_advanced_dataset.csv"
MODEL_PATH = BASE_DIR / "models" / "congestion_model.joblib"
METADATA_PATH = BASE_DIR / "models" / "model_metadata.joblib"

# --------------------------------------------------
# DARK DASHBOARD STYLING
# --------------------------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0b1120;
        color: #e5edf7;
    }

    [data-testid="stSidebar"] {
        background-color: #111827;
    }

    [data-testid="stSidebar"] * {
        color: #e5edf7;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }

    h1, h2, h3, h4 {
        color: #f8fafc !important;
    }

    p, label, .stCaption {
        color: #cbd5e1;
    }

    [data-testid="stMetric"] {
        background-color: #151f32;
        border: 1px solid #2b3b52;
        border-radius: 14px;
        padding: 20px;
    }

    [data-testid="stMetricLabel"] {
        color: #cbd5e1 !important;
    }

    [data-testid="stMetricValue"] {
        color: #f8fafc !important;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        color: #e5edf7 !important;
    }

    .stTextInput input,
    .stNumberInput input,
    .stSelectbox div[data-baseweb="select"] {
        background-color: #172338;
        color: #f8fafc;
    }

    .stButton button,
    .stFormSubmitButton button {
        background-color: #2563eb;
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
    }

    .stButton button:hover,
    .stFormSubmitButton button:hover {
        background-color: #1d4ed8;
        color: white;
        border: none;
    }

    [data-testid="stAlert"] {
        border-radius: 10px;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid #2b3b52;
        border-radius: 10px;
    }

    hr {
        border-color: #26354b;
    }

    .traffic-hero {
        position: relative;
        overflow: hidden;
        min-height: 300px;
        padding: 28px 32px;
        border: 1px solid #253b53;
        border-radius: 20px;
        background: linear-gradient(115deg, #101b2d 0%, #102b38 54%, #123a39 100%);
        margin: 10px 0 22px;
    }
    .traffic-hero h2 { margin: 0 0 8px; font-size: 30px; }
    .traffic-hero p { max-width: 590px; color: #b9c9d9; }
    .live-pill { color:#86efac; font-size:12px; font-weight:700; letter-spacing:1.4px; }
    .road-scene { position:absolute; right:2%; bottom:0; width:49%; height:190px; opacity:.95; }
    .road { position:absolute; bottom:19px; left:0; width:100%; height:104px; background:#202d3b; border-top:2px solid #405264; border-bottom:2px solid #405264; transform:perspective(300px) rotateX(5deg); }
    .lane-line { position:absolute; top:50%; left:0; width:200%; border-top:2px dashed #b6c5d3; opacity:.5; animation: laneMove 3s linear infinite; }
    .car { position:absolute; z-index:2; width:35px; height:17px; border-radius:6px 9px 4px 4px; box-shadow:0 0 14px currentColor; animation: drive 12s linear infinite; }
    .car:before { content:''; position:absolute; left:7px; top:-5px; width:18px; height:7px; border-radius:5px 5px 0 0; background:inherit; }
    .car:after { content:'●  ●'; position:absolute; top:11px; left:5px; color:#dbeafe; font-size:9px; letter-spacing:7px; }
    .car-a { top:55px; color:#34d399; background:#34d399; }
    .car-b { top:84px; color:#60a5fa; background:#60a5fa; animation-delay:-4s; animation-duration:15s; }
    .car-c { top:112px; color:#fbbf24; background:#fbbf24; animation-delay:-8s; animation-duration:18s; }
    .signal { position:absolute; right:12%; top:5px; z-index:3; width:15px; height:44px; border-radius:9px; background:#101820; padding:4px; box-shadow:0 0 0 2px #405264; }
    .signal i { display:block; width:7px; height:7px; margin:2px auto; border-radius:50%; background:#273442; }
    .signal i:last-child { background:#34d399; box-shadow:0 0 9px #34d399; }
    @keyframes drive { from { left:-50px; } to { left:105%; } }
    @keyframes laneMove { from { transform:translateX(0); } to { transform:translateX(-50%); } }
    .value-card { background:#121f30; border:1px solid #2b3b52; border-radius:14px; padding:18px 20px; min-height:122px; }
    .value-card strong { color:#f8fafc; font-size:15px; }
    .value-card span { color:#9fb1c6; font-size:13px; display:block; margin-top:8px; }
    .corridor-map { background:#101b2a; border:1px solid #2b3b52; border-radius:16px; padding:18px; }
    .corridor-row { display:grid; grid-template-columns:105px 1fr 82px; align-items:center; gap:12px; margin:12px 0; color:#dbe7f2; font-size:12px; }
    .corridor-track { height:25px; position:relative; overflow:hidden; border-top:2px solid #526174; border-bottom:2px solid #526174; background:repeating-linear-gradient(90deg,#263445 0 34px,#324256 34px 38px); }
    .corridor-track:after { content:''; position:absolute; top:11px; left:0; width:200%; border-top:1px dashed #cbd5e1; opacity:.55; animation:laneMove 3s linear infinite; }
    .map-car { position:absolute; z-index:2; top:5px; left:-28px; height:13px; width:23px; border-radius:5px 7px 3px 3px; animation:drive 10s linear infinite; }
    .map-car:before { content:''; position:absolute; top:-4px; left:5px; width:13px; height:5px; background:inherit; border-radius:4px 4px 0 0; }
    .map-car.two { animation-delay:-3s; animation-duration:13s; }
    .map-car.three { animation-delay:-7s; animation-duration:16s; }
    .severity-tag { text-align:right; font-weight:700; }
    .incident-card { background:#151f32; border:1px solid #34445b; border-left:4px solid #f97316; border-radius:13px; padding:18px; min-height:170px; }
    .incident-card small { color:#9fb1c6; letter-spacing:1px; }
    .incident-card strong { display:block; color:#f8fafc; font-size:19px; margin:8px 0; }
    .incident-card p { color:#cbd5e1; font-size:13px; }
    @media (max-width: 760px) { .road-scene { position:relative; right:auto; width:100%; height:150px; margin-top:5px; } .traffic-hero { padding:22px; } }
    </style>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------
# LOAD DATA AND MODEL
# --------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH, keep_default_na=False)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_metadata():
    return joblib.load(METADATA_PATH)


if not DATA_PATH.exists():
    st.error(f"Dataset not found: {DATA_PATH}")
    st.info(
        "Check that the dataset exists inside the dataset folder "
        "with the expected filename."
    )
    st.stop()

if not MODEL_PATH.exists():
    st.error(f"Trained model not found: {MODEL_PATH}")
    st.info("Check the models folder or run python train_model.py.")
    st.stop()

if not METADATA_PATH.exists():
    st.error(f"Model metadata not found: {METADATA_PATH}")
    st.info("Check the models folder or run python train_model.py.")
    st.stop()

try:
    df = load_data()
    model = load_model()
    metadata = load_metadata()
except Exception as exc:
    st.error("The dataset or model could not be loaded.")
    st.exception(exc)
    st.stop()

# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------
def chart_theme(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0b1120",
        plot_bgcolor="#111827",
        font_color="#e2e8f0",
        margin=dict(l=20, r=20, t=55, b=20),
        legend_title_text=""
    )
    return fig


def metric_count(column, value):
    if column in df.columns:
        return int(
            (
                df[column]
                .astype(str)
                .str.strip()
                .str.casefold()
                == str(value).strip().casefold()
            ).sum()
        )
    return 0


congestion_colors = {
    "Normal": "#22c55e",
    "Moderate": "#eab308",
    "High": "#f97316",
    "Severe": "#ef4444"
}

# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------
st.sidebar.title("🚦 TRAFFIC INTELLIGENCE")
st.sidebar.caption("SMART CITY MONITORING PLATFORM")
st.sidebar.divider()

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Command Center",
        "Traffic Analytics",
        "Congestion Prediction",
        "Anomaly Monitoring",
        "Dataset Explorer",
        "Model Performance",
        "THE ASPECTS"
    ]
)

st.sidebar.divider()
st.sidebar.caption("ML ENGINE: RANDOM FOREST")
st.sidebar.caption("DATA MODE: DATASET-BASED")

# --------------------------------------------------
# 1. COMMAND CENTER
# --------------------------------------------------
if page == "Command Center":
    total = len(df)
    severe = metric_count("Congestion_Level", "Severe")
    high = metric_count("Congestion_Level", "High")
    anomalies = metric_count("Anomaly_Label", "Yes")

    st.markdown(
        """
        <section class="traffic-hero">
          <div class="live-pill">● TRAFFIC OPERATIONS · DEMO FEED</div>
          <div class="live-pill">FLOWOPS · SIMULATED TRAFFIC</div>
          <h2>City traffic operations, in one view.</h2>
          <p>Help city teams spot corridor pressure, review risk, and coordinate a response from one operations workspace.</p>
          <div class="road-scene" aria-label="Animated vehicles moving along a road">
            <div class="signal"><i></i><i></i><i></i></div>
            <div class="road"><div class="lane-line"></div><div class="car car-a"></div><div class="car car-b"></div><div class="car car-c"></div></div>
          </div>
        </section>
        """,
        unsafe_allow_html=True
    )

    st.caption("Prototype view · Metrics and charts use the supplied dataset; no live sensor feed is connected.")

    hotspots = df.copy()
    if "Congestion_Level" in hotspots.columns:
        hotspots = hotspots[hotspots["Congestion_Level"].astype(str).str.strip().str.casefold().isin(["high", "severe"])]
    corridor_count = int(df["Road_ID"].nunique()) if "Road_ID" in df.columns else 0
    hotspot_count = len(hotspots)
    average_speed = pd.to_numeric(df.get("Average_Speed_kmph", pd.Series(dtype=float)), errors="coerce").mean()
    predicted_hotspots = None
    try:
        feature_columns = metadata["feature_columns"]
        if all(feature in df.columns for feature in feature_columns):
            batch_predictions = model.predict(df[feature_columns])
            predicted_hotspots = int(pd.Series(batch_predictions).astype(str).isin(["High", "Severe"]).sum())
    except Exception:
        predicted_hotspots = None

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("CORRIDORS IN DATASET", f"{corridor_count:,}")
    c2.metric("HIGH-RISK RECORDS", f"{hotspot_count:,}", help="Historical High or Severe congestion records in this dataset")
    c3.metric("MODEL-FLAGGED HOTSPOTS", f"{predicted_hotspots:,}" if predicted_hotspots is not None else "Unavailable", help="Batch model classifications over historical dataset rows; not a future forecast")
    c4.metric("AVERAGE RESPONSE TIME", "Not tracked", help="The dataset does not contain operator response timestamps")
    st.caption(f"Historical dataset: {total:,} records · {anomalies:,} anomaly labels · Mean recorded speed {average_speed:.1f} km/h." if pd.notna(average_speed) else f"Historical dataset: {total:,} records · {anomalies:,} anomaly labels.")

    map_left, map_right = st.columns([1.65, 1])
    with map_left:
        st.subheader("Corridor network")
        if "Road_ID" in df.columns and "Congestion_Level" in df.columns:
            severity_rank = {"Normal": 0, "Moderate": 1, "High": 2, "Severe": 3}
            corridor_levels = df.assign(
                _rank=df["Congestion_Level"].astype(str).map(severity_rank).fillna(0)
            ).groupby("Road_ID").agg(_rank=("_rank", "max"), level=("Congestion_Level", "last")).sort_values("_rank", ascending=False).head(4)
            level_color = {"Normal": "#22c55e", "Moderate": "#eab308", "High": "#f97316", "Severe": "#ef4444"}
            rows = []
            for road_id, row in corridor_levels.iterrows():
                level = str(row["level"])
                color = level_color.get(level, "#60a5fa")
                cars = ''.join(f'<i class="map-car {cls}" style="background:{color}"></i>' for cls in ["", "two", "three"])
                rows.append(f'<div class="corridor-row"><span>{road_id} Corridor</span><div class="corridor-track">{cars}</div><span class="severity-tag" style="color:{color}">{level}</span></div>')
            st.markdown('<div class="corridor-map">' + ''.join(rows) + '<small style="color:#9fb1c6">Animated vehicles are illustrative. Colors show the highest historical label per road.</small></div>', unsafe_allow_html=True)
        else:
            st.info("Road and congestion fields are needed to draw the corridor map.")
    with map_right:
        st.subheader("Priority incident")
        if hotspot_count:
            sample = hotspots.iloc[0]
            road = sample.get("Road_ID", "Road")
            junction = sample.get("Junction_ID", "")
            level = str(sample.get("Congestion_Level", "High"))
            action = "Review signal timing and verify queue conditions." if level == "Severe" else "Review signal timing and monitor the next interval."
            st.markdown(f'<div class="incident-card"><small>DATASET-DERIVED · OPERATOR REVIEW</small><strong>{road} Corridor · {level}</strong><p>{("Junction " + str(junction) + " · ") if junction else ""}{action}</p></div>', unsafe_allow_html=True)
        else:
            st.info("No High or Severe incident appears in this dataset.")
        st.caption("No live alerting, dispatch, or signal control is connected.")

    st.subheader("Operator action queue")
    alert_left, alert_right = st.columns([1.3, 1])
    with alert_left:
        if hotspot_count and "Road_ID" in hotspots.columns:
            display_cols = [col for col in ["Road_ID", "Junction_ID", "Congestion_Level", "Average_Speed_kmph", "Average_Waiting_Time_sec", "Incident_Type"] if col in hotspots.columns]
            st.dataframe(hotspots[display_cols].head(5), use_container_width=True, hide_index=True)
        else:
            st.success("No High or Severe records found in the supplied dataset.")
    with alert_right:
        if hotspot_count:
            sample = hotspots.iloc[0]
            road = sample.get("Road_ID", "Selected corridor")
            level = sample.get("Congestion_Level", "High")
            junction = sample.get("Junction_ID", "")
            st.warning(f"**Review {road}{' · ' + str(junction) if junction else ''}**  \nDataset severity: **{level}**. Validate conditions with an operator before dispatching a response.")
        else:
            st.info("No dataset-derived corridor alert is available.")
        st.caption("Suggested review only · this prototype does not dispatch teams or change signals.")

    st.subheader("How an operator uses the service")
    step_cols = st.columns(4)
    for col, title, detail in zip(step_cols, ["01 · Detect", "02 · Assess", "03 · Respond", "04 · Measure"], ["Surface busy corridors", "Review severity and model output", "Coordinate a field or signal response", "Track trends in the next report"]):
        with col:
            st.markdown(f'<div class="value-card"><strong>{title}</strong><span>{detail}</span></div>', unsafe_allow_html=True)

    st.divider()

    st.subheader("Recent congestion trend")
    if "Timestamp" in df.columns and "Congestion_Level" in df.columns:
        trend = df.copy()
        trend["_timestamp"] = pd.to_datetime(trend["Timestamp"], errors="coerce")
        trend = trend.dropna(subset=["_timestamp"])
        if not trend.empty:
            trend["Date"] = trend["_timestamp"].dt.date
            trend["High or Severe"] = trend["Congestion_Level"].astype(str).isin(["High", "Severe"]).astype(int)
            trend = trend.groupby("Date", as_index=False)["High or Severe"].sum().tail(7)
            fig = px.line(trend, x="Date", y="High or Severe", markers=True, title="High or Severe records · last 7 observed dates")
            st.plotly_chart(chart_theme(fig), use_container_width=True)
            st.caption("Historical dataset trend; this is not a live weekly forecast.")

    left, right = st.columns([1.15, 1])

    with left:
        st.subheader("Congestion Distribution")

        if "Congestion_Level" in df.columns:
            counts = (
                df["Congestion_Level"]
                .value_counts()
                .rename_axis("Congestion Level")
                .reset_index(name="Records")
            )

            fig = px.pie(
                counts,
                names="Congestion Level",
                values="Records",
                hole=0.62,
                color="Congestion Level",
                color_discrete_map=congestion_colors
            )

            st.plotly_chart(
                chart_theme(fig),
                use_container_width=True
            )

    with right:
        st.subheader("Traffic Severity Overview")

        if "Congestion_Level" in df.columns:
            counts = df["Congestion_Level"].value_counts().reset_index()
            counts.columns = ["Congestion Level", "Records"]

            fig = px.bar(
                counts,
                x="Congestion Level",
                y="Records",
                color="Congestion Level",
                color_discrete_map=congestion_colors,
                text="Records"
            )

            st.plotly_chart(
                chart_theme(fig),
                use_container_width=True
            )

    st.divider()
    st.subheader("The product opportunity")
    st.caption("A business model built around operational outcomes, with this dashboard as the decision workspace.")
    v1, v2, v3 = st.columns(3)
    with v1:
        st.markdown('<div class="value-card"><strong>Who pays</strong><span>City transport departments, road operators, and large campuses managing busy corridors.</span></div>', unsafe_allow_html=True)
    with v2:
        st.markdown('<div class="value-card"><strong>What they get</strong><span>Congestion visibility, risk alerts, prediction tools, and evidence to prioritize response teams.</span></div>', unsafe_allow_html=True)
    with v3:
        st.markdown('<div class="value-card"><strong>How it can earn</strong><span>Recurring subscription per monitored corridor, with optional deployment, integrations, and support.</span></div>', unsafe_allow_html=True)

    with st.expander("Example customer report"):
        st.write(f"This dataset contains **{total:,} records** across **{corridor_count:,} roads**. It includes **{high + severe:,} High or Severe congestion records** and **{anomalies:,} anomaly labels**. These are historical dataset counts, not current city conditions or measured savings.")

    st.subheader("Recent Dataset Records")
    st.dataframe(
        df.head(10),
        use_container_width=True,
        hide_index=True
    )

# --------------------------------------------------
# 2. TRAFFIC ANALYTICS
# --------------------------------------------------
elif page == "Traffic Analytics":
    st.title("Traffic Analytics")
    st.caption("Explore when congestion builds, which corridors carry the most pressure, and how traffic measures move together.")

    if "Congestion_Level" not in df.columns:
        st.warning("The dataset has no Congestion_Level column.")
    else:
        analytics = df.copy()
        time_col = "Timestamp" if "Timestamp" in analytics.columns else None
        if time_col:
            analytics["_analytics_time"] = pd.to_datetime(analytics[time_col], errors="coerce")

        st.markdown("#### Refine the view")
        f1, f2, f3 = st.columns([1, 1.2, 1])
        levels = sorted(analytics["Congestion_Level"].astype(str).unique().tolist())
        with f1:
            selected_levels = st.multiselect("Congestion level", levels, default=levels, key="analytics_levels")
        with f2:
            roads = sorted(analytics["Road_ID"].astype(str).unique().tolist()) if "Road_ID" in analytics.columns else []
            selected_roads = st.multiselect("Corridors", roads, default=roads, key="analytics_roads") if roads else []
        with f3:
            weather_values = sorted(analytics["Weather"].astype(str).unique().tolist()) if "Weather" in analytics.columns else []
            selected_weather = st.multiselect("Weather", weather_values, default=weather_values, key="analytics_weather") if weather_values else []

        filtered = analytics[analytics["Congestion_Level"].astype(str).isin(selected_levels)]
        if roads:
            filtered = filtered[filtered["Road_ID"].astype(str).isin(selected_roads)]
        if weather_values:
            filtered = filtered[filtered["Weather"].astype(str).isin(selected_weather)]

        numeric_candidates = [c for c in ["Average_Speed_kmph", "Traffic_Density_vehicles_km", "Road_Occupancy_pct", "Average_Waiting_Time_sec", "Travel_Time_Ratio", "Congestion_Score"] if c in filtered.columns]
        speed = pd.to_numeric(filtered.get("Average_Speed_kmph", pd.Series(dtype=float)), errors="coerce")
        waiting = pd.to_numeric(filtered.get("Average_Waiting_Time_sec", pd.Series(dtype=float)), errors="coerce")
        score = pd.to_numeric(filtered.get("Congestion_Score", pd.Series(dtype=float)), errors="coerce")
        anomalies_filtered = int(filtered.get("Anomaly_Label", pd.Series(dtype=str)).astype(str).str.casefold().eq("yes").sum())
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("MATCHING RECORDS", f"{len(filtered):,}")
        m2.metric("MEAN SPEED", f"{speed.mean():.1f} km/h" if speed.notna().any() else "—")
        m3.metric("MEAN WAIT", f"{waiting.mean():.1f} sec" if waiting.notna().any() else "—")
        m4.metric("ANOMALIES", f"{anomalies_filtered:,}")

        if filtered.empty:
            st.info("No records match these filters. Select at least one level, corridor, and weather type.")
        else:
            st.divider()
            chart_left, chart_right = st.columns(2)

            with chart_left:
                st.subheader("Congestion through time")
                if time_col and filtered["_analytics_time"].notna().any():
                    trend = filtered.dropna(subset=["_analytics_time"]).copy()
                    trend["Date"] = trend["_analytics_time"].dt.date
                    trend = trend.groupby(["Date", "Congestion_Level"], as_index=False).size().rename(columns={"size": "Records"})
                    fig = px.area(trend, x="Date", y="Records", color="Congestion_Level", color_discrete_map=congestion_colors, category_orders={"Congestion_Level": ["Normal", "Moderate", "High", "Severe"]})
                    fig.update_layout(hovermode="x unified", legend_title_text="Severity")
                    st.plotly_chart(chart_theme(fig), use_container_width=True)
                else:
                    st.info("Timestamp values are unavailable for a trend chart.")

            with chart_right:
                st.subheader("Corridor pressure ranking")
                if "Road_ID" in filtered.columns:
                    if "Congestion_Score" in filtered.columns:
                        ranked_data = filtered.copy()
                        ranked_data["_score"] = pd.to_numeric(ranked_data["Congestion_Score"], errors="coerce")
                        rank = ranked_data.groupby("Road_ID", as_index=False).agg(Records=("Road_ID", "size"), Mean_score=("_score", "mean")).rename(columns={"Mean_score": "Mean score"})
                        metric = "Mean score"
                    else:
                        rank = filtered.groupby("Road_ID", as_index=False).agg(Records=("Road_ID", "size"))
                        metric = "Records"
                    rank = rank.sort_values(metric, ascending=False).head(10)
                    fig = px.bar(rank.sort_values(metric), x=metric, y="Road_ID", orientation="h", color=metric, color_continuous_scale=["#22c55e", "#eab308", "#f97316", "#ef4444"], text_auto=".1f")
                    fig.update_layout(coloraxis_showscale=False, yaxis_title="Corridor")
                    st.plotly_chart(chart_theme(fig), use_container_width=True)
                else:
                    st.info("Road_ID is unavailable for corridor comparison.")

            lower_left, lower_right = st.columns(2)
            with lower_left:
                st.subheader("Traffic conditions relationship")
                if "Average_Speed_kmph" in filtered.columns and "Average_Waiting_Time_sec" in filtered.columns:
                    scatter = filtered.copy()
                    scatter["Average_Speed_kmph"] = pd.to_numeric(scatter["Average_Speed_kmph"], errors="coerce")
                    scatter["Average_Waiting_Time_sec"] = pd.to_numeric(scatter["Average_Waiting_Time_sec"], errors="coerce")
                    hover_cols = [c for c in ["Road_ID", "Weather", "Incident_Type"] if c in scatter.columns]
                    fig = px.scatter(scatter, x="Average_Speed_kmph", y="Average_Waiting_Time_sec", color="Congestion_Level", color_discrete_map=congestion_colors, hover_data=hover_cols, opacity=.72)
                    fig.update_layout(xaxis_title="Average speed (km/h)", yaxis_title="Average wait (sec)")
                    st.plotly_chart(chart_theme(fig), use_container_width=True)
                else:
                    st.info("Speed and waiting-time columns are needed for this comparison.")

            with lower_right:
                st.subheader("Feature distribution")
                if numeric_candidates:
                    selected_feature = st.selectbox("Measure", numeric_candidates, key="analytics_feature")
                    distribution = filtered.copy()
                    distribution[selected_feature] = pd.to_numeric(distribution[selected_feature], errors="coerce")
                    fig = px.histogram(distribution, x=selected_feature, color="Congestion_Level", color_discrete_map=congestion_colors, marginal="box", barmode="overlay", opacity=.72)
                    st.plotly_chart(chart_theme(fig), use_container_width=True)
                else:
                    st.info("No numeric traffic measures are available.")

            with st.expander(f"Inspect matching records ({len(filtered):,})"):
                visible = filtered.drop(columns=["_analytics_time"], errors="ignore")
                st.dataframe(visible, use_container_width=True, hide_index=True)
                st.download_button("Download filtered data", data=visible.to_csv(index=False).encode("utf-8"), file_name="flowops_traffic_analytics.csv", mime="text/csv")

# --------------------------------------------------
# 3. CONGESTION PREDICTION
# --------------------------------------------------
elif page == "Congestion Prediction":
    st.title("Congestion Prediction Lab")
    st.caption(
        "Submit traffic conditions to obtain an ML prediction."
    )

    try:
        feature_columns = metadata["feature_columns"]
        numeric_cols = metadata["numeric_columns"]
        categorical_cols = metadata["categorical_columns"]

        missing_features = [
            col for col in feature_columns if col not in df.columns
        ]

        if missing_features:
            st.error(
                "The dataset is missing model input features: "
                + ", ".join(missing_features)
            )
            st.stop()

        with st.form("prediction_form"):
            st.subheader("Traffic Input Parameters")
            input_values = {}
            cols = st.columns(2)

            for i, feature in enumerate(feature_columns):
                with cols[i % 2]:
                    if feature in numeric_cols:
                        series = pd.to_numeric(
                            df[feature],
                            errors="coerce"
                        ).dropna()

                        default = (
                            float(series.median())
                            if not series.empty
                            else 0.0
                        )

                        input_values[feature] = st.number_input(
                            feature.replace("_", " "),
                            value=default,
                            key=f"num_{feature}"
                        )

                    else:
                        choices = sorted(
                            df[feature]
                            .astype(str)
                            .replace("", "Unknown")
                            .unique()
                            .tolist()
                        )

                        if not choices:
                            choices = ["Unknown"]

                        input_values[feature] = st.selectbox(
                            feature.replace("_", " "),
                            choices,
                            key=f"cat_{feature}"
                        )

            submitted = st.form_submit_button(
                "PREDICT CONGESTION",
                use_container_width=True
            )

        if submitted:
            input_df = pd.DataFrame(
                [input_values],
                columns=feature_columns
            )

            prediction = model.predict(input_df)[0]

            st.divider()
            st.subheader("Prediction Result")

            color = congestion_colors.get(
                str(prediction),
                "#60a5fa"
            )

            st.markdown(
                f"""
                <div style="
                    background:#151f32;
                    padding:25px;
                    border-radius:14px;
                    border-left:6px solid {color};
                ">
                    <div style="
                        color:#a8b7cc;
                        font-size:14px;
                    ">
                        PREDICTED CONGESTION LEVEL
                    </div>
                    <div style="
                        font-size:34px;
                        font-weight:700;
                        color:{color};
                    ">
                        {prediction}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(input_df)[0]
                classes = model.classes_

                probability_df = pd.DataFrame({
                    "Congestion Level": classes,
                    "Model Score (%)": probabilities * 100
                }).sort_values(
                    "Model Score (%)",
                    ascending=False
                )

                st.subheader("Model Prediction Scores")

                fig = px.bar(
                    probability_df,
                    x="Congestion Level",
                    y="Model Score (%)",
                    color="Congestion Level",
                    color_discrete_map=congestion_colors,
                    text_auto=".1f"
                )

                st.plotly_chart(
                    chart_theme(fig),
                    use_container_width=True
                )

            st.caption(
                "Prediction scores are model outputs, not guaranteed "
                "real-world probabilities."
            )

    except Exception as exc:
        st.error(
            "Prediction could not be completed. Check that the "
            "saved model and metadata match."
        )
        st.exception(exc)

# --------------------------------------------------
# 4. ANOMALY MONITORING
# --------------------------------------------------
elif page == "Anomaly Monitoring":
    st.title("Anomaly Monitoring")
    st.caption(
        "Inspect records labelled as anomalous in the dataset."
    )

    if "Anomaly_Label" not in df.columns:
        st.warning("The dataset has no Anomaly_Label column.")

    else:
        anomaly_df = df[
            df["Anomaly_Label"]
            .astype(str)
            .str.strip()
            .str.lower()
            .eq("yes")
        ]

        m1, m2 = st.columns(2)

        m1.metric("Flagged Records", len(anomaly_df))
        m2.metric(
            "Share of Dataset",
            f"{len(anomaly_df) / max(len(df), 1) * 100:.2f}%"
        )

        st.info(
            "These are existing dataset labels. This page does not "
            "run a separate anomaly-detection algorithm on new traffic data."
        )

        st.subheader("Flagged Traffic Records")

        if anomaly_df.empty:
            st.success("No anomaly-labelled records were found.")
        else:
            st.dataframe(
                anomaly_df,
                use_container_width=True,
                hide_index=True
            )

# --------------------------------------------------
# 5. DATASET EXPLORER
# --------------------------------------------------
elif page == "Dataset Explorer":
    st.title("Dataset Explorer")
    st.caption("Search, filter, and inspect the traffic dataset.")

    search = st.text_input(
        "Search all columns",
        placeholder="Enter a keyword..."
    )

    filtered = df.copy()

    if search.strip():
        mask = filtered.astype(str).apply(
            lambda col: col.str.contains(
                search,
                case=False,
                na=False,
                regex=False
            )
        ).any(axis=1)

        filtered = filtered[mask]

    if "Congestion_Level" in filtered.columns:
        levels = sorted(
            filtered["Congestion_Level"].astype(str).unique()
        )

        chosen = st.multiselect(
            "Filter by congestion level",
            levels,
            default=levels
        )

        filtered = filtered[
            filtered["Congestion_Level"].astype(str).isin(chosen)
        ]

    st.write(f"**{len(filtered):,} records found**")

    st.dataframe(
        filtered,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "Download Filtered CSV",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="filtered_traffic_dataset.csv",
        mime="text/csv"
    )

# --------------------------------------------------
# 6. MODEL PERFORMANCE
# --------------------------------------------------
elif page == "Model Performance":
    st.title("Model Performance")
    st.caption(
        "Review the saved model's held-out evaluation results."
    )

    st.warning(
        "The metrics below are saved training evaluation results. "
        "They are not recalculated from live traffic data."
    )

    st.metric("Test Accuracy", "91.25%")
    st.metric("Weighted F1-score", "91.22%")

    st.subheader("Confusion Matrix")

    confusion = pd.DataFrame(
        [
            [87, 3, 0, 3],
            [3, 65, 2, 0],
            [0, 6, 28, 0],
            [4, 0, 0, 39]
        ],
        index=[
            "Actual High",
            "Actual Moderate",
            "Actual Normal",
            "Actual Severe"
        ],
        columns=[
            "Predicted High",
            "Predicted Moderate",
            "Predicted Normal",
            "Predicted Severe"
        ]
    )

    fig = px.imshow(
        confusion,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Blues"
    )

    st.plotly_chart(
        chart_theme(fig),
        use_container_width=True
    )

    st.subheader("Classification Report")

    report = pd.DataFrame({
        "Class": ["High", "Moderate", "Normal", "Severe"],
        "Precision": [0.93, 0.88, 0.93, 0.93],
        "Recall": [0.94, 0.93, 0.82, 0.91],
        "F1-score": [0.93, 0.90, 0.88, 0.92],
        "Support": [93, 70, 34, 43]
    })

    st.dataframe(
        report,
        use_container_width=True,
        hide_index=True
    )

# --------------------------------------------------
# 7. PRODUCT ASPECTS
# --------------------------------------------------
elif page == "THE ASPECTS":
    st.title("THE ASPECTS")
    st.caption("A product brief for FlowOps: what the service does, what a dependable deployment requires, and how it could become a sustainable business.")

    st.markdown("### FlowOps in one sentence")
    st.info("FlowOps is a proposed traffic operations platform that helps city and road teams understand congestion patterns, review risk, and coordinate decisions across monitored corridors.")
    st.caption("Current status: a dataset-driven prototype. It demonstrates analytics and model workflows; it is not yet connected to live traffic infrastructure.")

    functional_tab, quality_tab, business_tab = st.tabs(["Functional requirements", "Non-functional requirements", "Business model"])

    with functional_tab:
        st.subheader("What the product should let people do")
        functional_items = [
            ("Monitor corridors", "View congestion severity, traffic measures, and recent patterns by road or junction.", "Prototype: corridor overview is derived from historical dataset rows."),
            ("Explore and compare", "Filter by severity, corridor, and conditions; compare time trends and traffic measures.", "Prototype: available in Traffic Analytics and Dataset Explorer."),
            ("Predict congestion", "Enter traffic conditions and receive a congestion class with model scores to support operator review.", "Prototype: manual prediction form; not a live or continuously updating forecast."),
            ("Review unusual records", "Surface anomaly-labelled observations and give operators context for investigation.", "Prototype: displays existing dataset labels; no separate live anomaly pipeline is connected."),
            ("Support response and reporting", "Turn a reviewed risk into an assigned action, then track response and outcome for reporting.", "Future capability: dispatch, signal integrations, ownership, and response-time tracking are not implemented.")
        ]
        for title, description, status in functional_items:
            with st.container(border=True):
                st.markdown(f"**{title}**")
                st.write(description)
                st.caption(status)

    with quality_tab:
        st.subheader("How a production service should behave")
        quality_items = [
            ("Reliable", "Keep dashboards and ingestion available during operating hours; show a clear stale-data state when feeds stop."),
            ("Fast", "Refresh maps, alerts, and filters quickly enough for an operator to use during a developing incident."),
            ("Secure and private", "Use role-based access, protect credentials and traffic data, and keep an auditable record of operator actions."),
            ("Understandable", "Show timestamps, source, confidence or model score, and a plain-language reason behind each alert."),
            ("Accessible", "Use readable contrast, keyboard-friendly controls, clear labels, and colors that are not the only way to convey severity."),
            ("Maintainable and scalable", "Support new corridors and data providers without rebuilding the dashboard; monitor model quality as traffic patterns change.")
        ]
        for title, description in quality_items:
            with st.container(border=True):
                st.markdown(f"**{title}**")
                st.write(description)
        st.warning("These are production requirements and design goals. The current local prototype has not been assessed against service-level targets, security controls, accessibility standards, or live-data reliability.")

    with business_tab:
        st.subheader("Who it serves and how it can earn")
        b1, b2, b3 = st.columns(3)
        with b1:
            st.markdown("#### Buyers")
            st.write("City transport departments, road authorities, toll-road operators, and large campuses managing traffic corridors.")
        with b2:
            st.markdown("#### Value")
            st.write("A shared view of corridor conditions, earlier visibility into risk, and evidence to prioritize operator attention.")
        with b3:
            st.markdown("#### Revenue")
            st.write("Recurring subscription priced by monitored corridor or service area, with optional deployment, data integrations, training, and support.")

        st.subheader("A practical service journey")
        stages = st.columns(4)
        for col, title, detail in zip(stages, ["1 · Pilot", "2 · Connect", "3 · Operate", "4 · Expand"], ["Agree on a small set of corridors and success measures.", "Integrate approved sensor, camera, or transport feeds.", "Provide dashboard access, alerts, training, and support.", "Add corridors and reporting as value is demonstrated."]):
            with col:
                st.markdown(f'<div class="value-card"><strong>{title}</strong><span>{detail}</span></div>', unsafe_allow_html=True)

        st.subheader("Measure value before making savings claims")
        st.write("A customer pilot could track alert review time, operator response time, congestion duration, corridor travel speed, and alert usefulness before and after deployment. Compare similar time periods and account for incidents and weather; only report savings once the data supports them.")
        st.caption("No price, customer result, or ROI is claimed by this prototype. Pricing and outcomes should be validated with pilot customers.")

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.sidebar.divider()
st.sidebar.caption("TRAFFIC CONGESTION MONITORING SYSTEM")
st.sidebar.caption("Dataset-based prototype")
