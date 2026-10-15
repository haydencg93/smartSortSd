"""Streamlit dashboard for Smart Sort transaction data.

Expected transactions table schema (created by ``server.py``):

    CREATE TABLE transactions (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp     TEXT NOT NULL,          -- ISO 8601
        input_method  TEXT NOT NULL,          -- 'barcode' | 'image'
        category      TEXT NOT NULL,          -- 'recycling' | 'trash'
        confidence    REAL,                   -- 0-1, NULL for barcode (deterministic lookup)
        latency_ms    REAL NOT NULL,          -- server-side processing time
        status        TEXT NOT NULL,          -- 'success' | 'retry' (R1.1.6 — not recognized)
        bin_direction TEXT,                   -- 'left' | 'right' | etc.
        image_path    TEXT                    -- path/URL to the captured frame the server saved,
                                               -- NULL for pure barcode lookups with no image capture
    )

Run with:  streamlit run app/dashboard.py
"""

import os
import sqlite3
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw

warnings.filterwarnings("ignore", category=alt.AltairDeprecationWarning)

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = os.environ.get("SMARTSORT_DB", str(PROJECT_ROOT / "waste_sorter.db"))
SLA_SECONDS = 5              # C1.1 — recommendation must display within 5s
ACCURACY_TARGET = 0.75       # C1.2 — held-out test accuracy target

# Reported from the latest offline evaluation; live transactions do not contain
# ground-truth labels for computing model accuracy.
LATEST_HELDOUT_ACCURACY = 0.81
LATEST_HELDOUT_ACCURACY_DATE = "2026-09-01"

MOCK_IMAGE_DIR = str(PROJECT_ROOT / "mock_captures")

# Category → bin direction + display color, in one place (R1.2.2 uses the
# direction; charts use the color). R1.1.1 only requires recycling/trash,
# but everything downstream reads this dict rather than hardcoding category
# names, so adding one later — e.g. compost, hazardous — is a one-line change.
CATEGORIES = {
    "recycling": {"bin_direction": "left", "color": "#2F6B4F"},
    "trash": {"bin_direction": "right", "color": "#8C6A4F"},
    # "compost": {"bin_direction": "center", "color": "#7A8C4F"},  # example: uncomment to extend
}

INPUT_METHODS = {
    "barcode": "#3D6B8C",
    "image": "#C98A3E",
}

STATUS_COLORS = {
    "success": "#8B9388",
    "retry": "#B23A2E",
}

st.set_page_config(
    page_title="Waste sorter dashboard",
    page_icon="♻️",
    layout="wide",
)


def altair_theme():
    return {
        "config": {
            "font": "sans-serif",
            "axis": {
                "labelFont": "sans-serif",
                "titleFont": "sans-serif",
                "labelColor": "#5B6259",
                "titleColor": "#202A22",
                "labelFontSize": 11,
                "titleFontSize": 12,
                "titleFontWeight": 500,
                "titlePadding": 12,
                "labelPadding": 6,
                "gridColor": "#E4E6DE",
                "domainColor": "#C8CBC1",
                "tickColor": "#C8CBC1",
            },
            "legend": {
                "labelFont": "sans-serif",
                "titleFont": "sans-serif",
                "labelFontSize": 11,
                "titleFontSize": 12,
                "titleFontWeight": 500,
                "titlePadding": 8,
                "padding": 8,
            },
            "title": {
                "font": "sans-serif",
                "fontSize": 13,
                "fontWeight": 500,
                "color": "#202A22",
                "anchor": "start",
                "offset": 16,
            },
            "view": {"stroke": "transparent"},
        }
    }


alt.themes.register("waste_sorter", altair_theme)
alt.themes.enable("waste_sorter")


def themed(chart: alt.Chart, height: int = 260) -> alt.Chart:
    """Apply consistent sizing/padding and fix Vega-Lite's tab-hidden sizing bug.

    Charts inside a Streamlit tab that isn't the active one on first render get
    measured at zero width by Vega-Lite, which is what caused labels to render
    "buggy"/overlapping when switching from Overview to Input methods. Forcing
    width="container" + autosize "fit-x" makes Vega-Lite re-measure against the
    actual container the moment the tab becomes visible, instead of caching a
    stale/zero size from its hidden first render.
    """
    return chart.properties(
        width="container",
        height=height,
        autosize=alt.AutoSizeParams(type="fit-x", contains="padding"),
        padding={"left": 6, "right": 28, "top": 12, "bottom": 10},
    )


def hover_line_chart(
    data: pd.DataFrame,
    x_field: str,
    y_field: str,
    color_field: str,
    x_title: str,
    y_title: str,
    color_title: str,
    domain: list,
    range_: list,
    title: str,
) -> alt.Chart:
    """A multi-series line chart whose point markers are hidden by default and
    only appear on the series nearest the cursor — instead of every data point
    being permanently dotted, which gets visually noisy with dense daily data.
    """
    nearest = alt.selection_point(nearest=True, on="mouseover", fields=[x_field], empty=False)

    base = alt.Chart(data).encode(
        x=alt.X(f"{x_field}:T", title=x_title),
        y=alt.Y(f"{y_field}:Q", title=y_title),
        color=alt.Color(f"{color_field}:N", title=color_title, scale=alt.Scale(domain=domain, range=range_)),
    )

    line = base.mark_line(strokeWidth=2)
    points = base.mark_point(size=70, filled=True).encode(
        opacity=alt.condition(nearest, alt.value(1), alt.value(0))
    )
    # Wide invisible hit-area so hovering anywhere near a series (not just
    # exactly on the line's pixels) triggers the point + tooltip.
    hit_area = base.mark_point(size=250, opacity=0).encode(
        tooltip=[
            alt.Tooltip(f"{x_field}:T", title=x_title),
            alt.Tooltip(f"{color_field}:N", title=color_title),
            alt.Tooltip(f"{y_field}:Q", title=y_title),
        ]
    ).add_params(nearest)

    return (line + points + hit_area).properties(title=title)


def _ensure_mock_images() -> dict:
    """Create one placeholder capture per category (idempotent — skips if already on disk).

    Sample images are generated only when the dashboard is using mock data.
    """
    os.makedirs(MOCK_IMAGE_DIR, exist_ok=True)
    paths = {}
    for cat, meta in CATEGORIES.items():
        path = os.path.join(MOCK_IMAGE_DIR, f"{cat}_sample.jpg")
        if not os.path.exists(path):
            img = Image.new("RGB", (480, 360), color=meta["color"])
            draw = ImageDraw.Draw(img)
            draw.text((24, 24), f"{cat.upper()}", fill="white")
            draw.text((24, 60), "placeholder capture", fill="white")
            draw.text((24, 320), "mock data — not a real item photo", fill="white")
            img.save(path, "JPEG", quality=85)
        paths[cat] = path
    return paths


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------

@st.cache_data(ttl=10)  # short TTL so the dashboard stays close to live once real data exists
def load_data() -> tuple[pd.DataFrame, str | None]:
    """Load transactions from the configured SQLite database."""
    if not os.path.exists(DB_PATH):
        return pd.DataFrame(), "Database file does not exist."
    try:
        with sqlite3.connect(DB_PATH, timeout=2) as connection:
            df = pd.read_sql_query("SELECT * FROM transactions", connection)
    except sqlite3.Error as error:
        return pd.DataFrame(), str(error)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df, None


@st.cache_data
def _generate_mock_data(n: int = 600) -> pd.DataFrame:
    """Synthetic data shaped like the real log, for UI development only."""
    rng = np.random.default_rng(seed=42)

    now = datetime.now()
    timestamps = [now - timedelta(minutes=int(m)) for m in rng.exponential(scale=180, size=n).cumsum()]
    timestamps = sorted(timestamps)

    input_method = rng.choice(list(INPUT_METHODS.keys()), size=n, p=[0.3, 0.7])

    # Even weighting across whatever categories are configured above, so
    # adding a 3rd/4th category doesn't require rebalancing probabilities.
    cat_names = list(CATEGORIES.keys())
    category = rng.choice(cat_names, size=n, p=[1 / len(cat_names)] * len(cat_names))

    confidence = np.where(
        input_method == "barcode",
        1.0,  # barcode lookups are deterministic, not a model confidence score
        np.clip(rng.normal(loc=0.84, scale=0.1, size=n), 0.3, 0.99),
    )

    # image classification is slower than barcode lookup
    latency_ms = np.where(
        input_method == "barcode",
        rng.normal(loc=300, scale=80, size=n),
        rng.normal(loc=1800, scale=600, size=n),
    ).clip(min=50)

    # low-confidence image reads occasionally fail and prompt a retry (R1.1.6)
    status = np.where(
        (input_method == "image") & (confidence < 0.55),
        "retry",
        "success",
    )

    bin_direction = np.array([CATEGORIES[c]["bin_direction"] for c in category])

    image_paths_by_category = _ensure_mock_images()
    # NULL image_path for barcode lookups — no camera frame was needed (R1.1.4)
    image_path = np.array(
        [image_paths_by_category[c] if m == "image" else None for c, m in zip(category, input_method)]
    )

    return pd.DataFrame(
        {
            "id": range(1, n + 1),
            "timestamp": timestamps,
            "input_method": input_method,
            "category": category,
            "confidence": confidence,
            "latency_ms": latency_ms,
            "status": status,
            "bin_direction": bin_direction,
            "image_path": image_path,
        }
    )


# ----------------------------------------------------------------------------
# Load + sidebar filters
# ----------------------------------------------------------------------------

st.sidebar.markdown("### Data source")
data_source = st.sidebar.selectbox(
    "Choose data",
    options=["Sample data", "Live database"],
    help="Sample data is generated for preview and is not added to the database.",
)

if data_source == "Sample data":
    df_raw = _generate_mock_data()
    database_error = None
    st.sidebar.caption("Preview mode · sample transactions are not saved.")
else:
    connection_status = st.sidebar.status(
        "Connecting to live database…",
        state="running",
        expanded=False,
    )
    if not os.path.exists(DB_PATH):
        df_raw = pd.DataFrame()
        database_error = None
        connection_status.update(
            label="Waiting to connect · database file not found",
            state="running",
            expanded=True,
        )
    else:
        df_raw, database_error = load_data()
        if database_error:
            connection_status.update(
                label="Waiting for a readable transactions table",
                state="error",
                expanded=True,
            )
        elif df_raw.empty:
            connection_status.update(
                label="Connected · waiting for transactions",
                state="complete",
                expanded=False,
            )
        else:
            connection_status.update(
                label="Connected to live database",
                state="complete",
                expanded=False,
            )
    if st.sidebar.button("Check connection again", icon=":material/refresh:"):
        load_data.clear()
        st.rerun()

if df_raw.empty:
    st.title("Waste sorter dashboard", icon=":material/recycling:")
    if data_source == "Sample data":
        st.info("No sample transactions are available.")
    elif database_error:
        st.error(f"Could not read transactions from the live database: {database_error}")
    elif not os.path.exists(DB_PATH):
        st.info("Waiting for the live database. Choose Sample data to preview the dashboard now.")
    else:
        st.info("Connected to the database. Waiting for the first classified transaction.")
    st.stop()

st.sidebar.markdown("### Filters")

min_date = df_raw["timestamp"].min().date()
max_date = df_raw["timestamp"].max().date()

date_range = st.sidebar.date_input(
    "Date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

selected_categories = st.sidebar.multiselect(
    "Category",
    options=sorted(df_raw["category"].unique()),
    default=sorted(df_raw["category"].unique()),
)

selected_methods = st.sidebar.multiselect(
    "Input method",
    options=sorted(df_raw["input_method"].unique()),
    default=sorted(df_raw["input_method"].unique()),
)

selected_status = st.sidebar.multiselect(
    "Status",
    options=sorted(df_raw["status"].unique()),
    default=sorted(df_raw["status"].unique()),
)

# Apply filters
if len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date = end_date = date_range[0]

mask = (
    (df_raw["timestamp"].dt.date >= start_date)
    & (df_raw["timestamp"].dt.date <= end_date)
    & (df_raw["category"].isin(selected_categories))
    & (df_raw["input_method"].isin(selected_methods))
    & (df_raw["status"].isin(selected_status))
)
df = df_raw.loc[mask].copy()

st.title("Waste sorter dashboard", icon=":material/recycling:")
st.caption(
    "Disposal recommendation activity for Iowa City · "
    f"{data_source} · {len(df):,} transactions in range · "
    f"Updated {datetime.now().strftime('%b %-d, %Y at %-I:%M %p')}"
)

if df.empty:
    st.warning("No transactions match the current filters.")
    st.stop()


# ----------------------------------------------------------------------------
# Top-level KPIs
# ----------------------------------------------------------------------------

total = len(df)
avg_latency_s = df["latency_ms"].mean() / 1000
sla_compliance = (df["latency_ms"] <= SLA_SECONDS * 1000).mean() * 100
retry_rate = (df["status"] == "retry").mean() * 100
category_pcts = (df["category"].value_counts(normalize=True) * 100).to_dict()

metric_definitions = [
    ("Transactions", f"{total:,}", "Current filter range"),
    ("Recycling", f"{category_pcts.get('recycling', 0):.0f}%", "Share of classified items"),
    ("Trash", f"{category_pcts.get('trash', 0):.0f}%", "Share of classified items"),
    ("Average latency", f"{avg_latency_s:.2f}s", f"Target ≤ {SLA_SECONDS}s"),
    ("Within SLA", f"{sla_compliance:.0f}%", f"Recommendations returned within {SLA_SECONDS}s"),
    ("Retry rate", f"{retry_rate:.1f}%", "Unrecognized visual input"),
]
for offset in range(0, len(metric_definitions), 3):
    metric_columns = st.columns(3)
    for column, (label, value, help_text) in zip(
        metric_columns, metric_definitions[offset : offset + 3]
    ):
        column.metric(label, value, help=help_text, border=True)


# ----------------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------------

tab_overview, tab_methods, tab_quality, tab_images, tab_log = st.tabs(
    ["Overview", "Input methods", "Accuracy & latency", "Images", "Raw log"]
)

cat_names = list(CATEGORIES.keys())
cat_colors = [CATEGORIES[c]["color"] for c in cat_names]
method_names = list(INPUT_METHODS.keys())
method_colors = list(INPUT_METHODS.values())

# --- Overview: counts over time + category split ---------------------------
with tab_overview:
    col1, col2 = st.columns([2, 1])

    with col1:
        daily = (
            df.set_index("timestamp")
            .groupby([pd.Grouper(freq="D"), "category"])
            .size()
            .reset_index(name="count")
        )
        chart = hover_line_chart(
            daily, "timestamp", "count", "category",
            "Date", "Items processed", "Category",
            cat_names, cat_colors,
            "Items processed per day, by category",
        )
        st.altair_chart(themed(chart, height=280), width="stretch", key="chart_daily_by_category")

    with col2:
        cat_df = df["category"].value_counts().reset_index()
        cat_df.columns = ["category", "count"]
        base = alt.Chart(cat_df).encode(
            x=alt.X("count:Q", title="Items"),
            y=alt.Y("category:N", title="", sort="-x"),
            color=alt.Color("category:N", scale=alt.Scale(domain=cat_names, range=cat_colors), legend=None),
        )
        bars = base.mark_bar()
        labels = base.mark_text(align="left", dx=6, color="#202A22").encode(text="count:Q")
        chart = (bars + labels).properties(title="Total items by category")
        st.altair_chart(themed(chart, height=280), width="stretch", key="chart_category_totals")

    available_days = sorted(df["timestamp"].dt.date.unique(), reverse=True)
    day_options = ["All days in range"] + [d.strftime("%a, %b %-d, %Y") for d in available_days]
    day_lookup = {d.strftime("%a, %b %-d, %Y"): d for d in available_days}

    hour_col, _ = st.columns([1, 2])
    with hour_col:
        selected_day_label = st.selectbox("Show hourly volume for", options=day_options, key="hour_day_picker")

    if selected_day_label == "All days in range":
        hour_source = df
        hour_title = "Volume by hour of day, summed across the selected date range"
    else:
        picked_day = day_lookup[selected_day_label]
        hour_source = df[df["timestamp"].dt.date == picked_day]
        hour_title = f"Volume by hour of day — {selected_day_label}"

    hourly = hour_source.groupby(hour_source["timestamp"].dt.hour).size().reindex(range(24), fill_value=0).reset_index()
    hourly.columns = ["hour", "count"]
    chart = (
        alt.Chart(hourly)
        .mark_bar(color="#5B6259")
        .encode(
            x=alt.X("hour:O", title="Hour of day (24h)"),
            y=alt.Y("count:Q", title="Items processed"),
            tooltip=[alt.Tooltip("hour:O", title="Hour"), alt.Tooltip("count:Q", title="Items")],
        )
        .properties(title=hour_title)
    )
    st.altair_chart(themed(chart, height=240), width="stretch", key="chart_hourly_volume")
    st.caption(
        "Use a single day to inspect its hourly pattern, or keep all days selected "
        "to see the aggregate volume."
    )

# --- Input methods -----------------------------------------------------------
with tab_methods:
    col1, col2 = st.columns(2)

    with col1:
        method_df = df["input_method"].value_counts().reset_index()
        method_df.columns = ["input_method", "count"]
        base = alt.Chart(method_df).encode(
            x=alt.X("count:Q", title="Transactions"),
            y=alt.Y("input_method:N", title="", sort="-x"),
            color=alt.Color("input_method:N", scale=alt.Scale(domain=method_names, range=method_colors), legend=None),
        )
        chart = (base.mark_bar() + base.mark_text(align="left", dx=6, color="#202A22").encode(text="count:Q")).properties(
            title="Barcode vs. visual identification"
        )
        st.altair_chart(themed(chart, height=200), width="stretch", key="chart_method_totals")
        st.caption("Barcode is attempted first; visual identification is the fallback.")

    with col2:
        status_df = df.groupby(["input_method", "status"]).size().reset_index(name="count")
        chart = (
            alt.Chart(status_df)
            .mark_bar()
            .encode(
                x=alt.X("input_method:N", title="Input method"),
                y=alt.Y("count:Q", title="Transactions"),
                color=alt.Color(
                    "status:N", title="Status", scale=alt.Scale(domain=list(STATUS_COLORS), range=list(STATUS_COLORS.values()))
                ),
                tooltip=["input_method:N", "status:N", "count:Q"],
            )
            .properties(title="Success vs. retry, by method")
        )
        st.altair_chart(themed(chart, height=200), width="stretch", key="chart_status_by_method")

    method_daily = (
        df.set_index("timestamp")
        .groupby([pd.Grouper(freq="D"), "input_method"])
        .size()
        .reset_index(name="count")
    )
    chart = hover_line_chart(
        method_daily, "timestamp", "count", "input_method",
        "Date", "Transactions", "Method",
        method_names, method_colors,
        "Method usage over time",
    )
    st.altair_chart(themed(chart, height=260), width="stretch", key="chart_method_over_time")

# --- Accuracy / confidence / latency -----------------------------------------
with tab_quality:
    image_df = df[df["input_method"] == "image"]

    # --- Extra summary stats: mean alone hides tail latency, and there's no
    # visibility yet into the one number officials probably care about most —
    # the actual measured accuracy from the held-out test set (C1.2).
    p50 = df["latency_ms"].quantile(0.50)
    p95 = df["latency_ms"].quantile(0.95)
    over_sla_count = int((df["latency_ms"] > SLA_SECONDS * 1000).sum())

    st.subheader("Model quality and response time")
    st.caption(
        f"Held-out accuracy was last measured on {LATEST_HELDOUT_ACCURACY_DATE}. "
        "Live confidence is a separate signal."
    )
    quality_metrics = [
        ("Held-out accuracy", f"{LATEST_HELDOUT_ACCURACY:.0%}", f"Target ≥ {ACCURACY_TARGET:.0%}"),
        ("Median latency", f"{p50:.0f}ms", "Typical response time"),
        ("P95 latency", f"{p95:.0f}ms", "Slower 5% of responses"),
        ("Over SLA", f"{over_sla_count:,}", f"Out of {total:,} transactions"),
    ]
    for offset in range(0, len(quality_metrics), 4):
        metric_columns = st.columns(4)
        for column, (label, value, help_text) in zip(
            metric_columns, quality_metrics[offset : offset + 4]
        ):
            column.metric(label, value, help=help_text, border=True)

    if not image_df.empty:
        conf_by_cat = image_df.groupby("category")["confidence"].mean().reset_index()
        chart = (
            alt.Chart(conf_by_cat)
            .mark_bar()
            .encode(
                x=alt.X("confidence:Q", title="Average confidence", scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("category:N", title="", sort="-x"),
                color=alt.Color("category:N", scale=alt.Scale(domain=cat_names, range=cat_colors), legend=None),
                tooltip=["category:N", alt.Tooltip("confidence:Q", title="Avg confidence", format=".2f")],
            )
            .properties(title="Average confidence by category — flags which items the model finds hardest")
        )
        st.altair_chart(themed(chart, height=200), width="stretch", key="chart_confidence_by_category")

    retry_daily = (
        df.set_index("timestamp")
        .groupby(pd.Grouper(freq="D"))
        .apply(lambda g: (g["status"] == "retry").mean() * 100, include_groups=False)
        .reset_index(name="retry_pct")
    )
    chart = (
        alt.Chart(retry_daily)
        .mark_line(strokeWidth=2, color="#B23A2E")
        .encode(
            x=alt.X("timestamp:T", title="Date"),
            y=alt.Y("retry_pct:Q", title="Retry rate (%)"),
            tooltip=[alt.Tooltip("timestamp:T", title="Date"), alt.Tooltip("retry_pct:Q", title="Retry rate", format=".1f")],
        )
        .properties(title="Retry rate over time (R1.1.6) — a rising trend can flag lighting or camera issues")
    )
    st.altair_chart(themed(chart, height=200), width="stretch", key="chart_retry_trend")
    if not image_df.empty:
        chart = (
            alt.Chart(image_df)
            .mark_bar(color="#C98A3E")
            .encode(
                x=alt.X("confidence:Q", bin=alt.Bin(maxbins=20), title="Confidence score"),
                y=alt.Y("count():Q", title="Transactions"),
                tooltip=[alt.Tooltip("count():Q", title="Transactions")],
            )
            .properties(title="Classifier confidence distribution (image identification only)")
        )
        st.altair_chart(themed(chart, height=260), width="stretch", key="chart_confidence_hist")
        st.caption(
            f"Target: ≥{ACCURACY_TARGET:.0%} held-out test accuracy. Live confidence "
            "does not replace the held-out evaluation."
        )
    else:
        st.info("No image-based transactions in this filter range.")

    col1, col2 = st.columns(2)
    with col1:
        base = alt.Chart(df).mark_bar(color="#5B6259").encode(
            x=alt.X("latency_ms:Q", bin=alt.Bin(maxbins=20), title="Latency (ms)"),
            y=alt.Y("count():Q", title="Transactions"),
        )
        rule = (
            alt.Chart(pd.DataFrame({"threshold": [SLA_SECONDS * 1000]}))
            .mark_rule(color="#B23A2E", strokeDash=[4, 4], size=2)
            .encode(x="threshold:Q")
        )
        chart = (base + rule).properties(title=f"Latency distribution (dashed = {SLA_SECONDS}s SLA)")
        st.altair_chart(themed(chart, height=260), width="stretch", key="chart_latency_hist")

    with col2:
        latency_daily = (
            df.set_index("timestamp")
            .groupby([pd.Grouper(freq="D"), "input_method"])["latency_ms"]
            .mean()
            .reset_index()
        )
        chart = (
            alt.Chart(latency_daily)
            .mark_line(point=True, strokeWidth=2)
            .encode(
                x=alt.X("timestamp:T", title="Date"),
                y=alt.Y("latency_ms:Q", title="Avg latency (ms)"),
                color=alt.Color("input_method:N", title="Method", scale=alt.Scale(domain=method_names, range=method_colors)),
                tooltip=[alt.Tooltip("timestamp:T", title="Date"), "input_method:N", alt.Tooltip("latency_ms:Q", title="Avg latency (ms)", format=".0f")],
            )
            .properties(title="Average latency by method, per day")
        )
        st.altair_chart(themed(chart, height=260), width="stretch", key="chart_latency_by_method")

# --- Raw log ------------------------------------------------------------------
with tab_log:
    st.subheader("Transaction log")
    st.dataframe(
        df.sort_values("timestamp", ascending=False),
        width="stretch",
        hide_index=True,
        column_config={
            "id": st.column_config.NumberColumn("ID"),
            "timestamp": st.column_config.DatetimeColumn("Timestamp", format="MMM D, YYYY · h:mm a"),
            "input_method": st.column_config.TextColumn("Method"),
            "category": st.column_config.TextColumn("Category"),
            "confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=1, format="%.2f"),
            "latency_ms": st.column_config.NumberColumn("Latency (ms)", format="%.0f"),
            "status": st.column_config.TextColumn("Status"),
            "bin_direction": st.column_config.TextColumn("Bin cue"),
        },
    )
    st.download_button(
        "Download filtered log as CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="waste_sorter_log.csv",
        mime="text/csv",
    )

st.caption(f"Data source: `{DB_PATH}`")
