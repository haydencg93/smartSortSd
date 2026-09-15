"""
Waste Sorter — Officials Dashboard (skeleton)
================================================

Satisfies R1.3.2 (web-based dashboard to visualize counts over time).

This is a SKELETON: it runs standalone right now using generated mock data,
so you can build/style the dashboard before the ESP32 + classify server
pipeline exists. Every place that needs to change when you wire up the real
SQLite log is marked "# TODO(real-data)".

Expected transactions table schema (created by the classify server):

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

Run with:  streamlit run dashboard.py
"""

import io
import os
import sqlite3
import zipfile
from datetime import datetime, timedelta

import warnings

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw

warnings.filterwarnings("ignore", category=alt.AltairDeprecationWarning)

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------

DB_PATH = "waste_sorter.db"  # TODO(real-data): point at the server's actual DB path
SLA_SECONDS = 5              # C1.1 — recommendation must display within 5s
ACCURACY_TARGET = 0.75       # C1.2 — held-out test accuracy target

# TODO(real-data): the live dashboard has no ground-truth labels, so it can't
# compute held-out accuracy itself — that number only exists when someone runs
# the offline evaluation script against a labeled test set. Point this at that
# script's output (e.g. read the latest value from a small JSON file it
# writes) instead of a hardcoded placeholder once that pipeline exists.
LATEST_HELDOUT_ACCURACY = 0.81
LATEST_HELDOUT_ACCURACY_DATE = "2026-09-01"

MOCK_IMAGE_DIR = "mock_captures"  # TODO(real-data): replace with the server's real capture directory

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

st.set_page_config(page_title="Waste Sorter Dashboard", page_icon="♻️", layout="wide")


# ----------------------------------------------------------------------------
# Styling — civic/municipal palette (fits an Iowa City public-works tool
# rather than a generic SaaS template): warm paper background, moss green
# for recycling, clay brown for trash, Public Sans for type (the U.S. Web
# Design System's official typeface — a deliberate fit for a government
# ops dashboard rather than a default web font).
# ----------------------------------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Public+Sans:wght@400;500;600;700&display=swap');

    html, body, [class*="css"], .stApp {
        font-family: 'Public Sans', sans-serif;
    }

    :root {
        --ws-bg: #F6F5F1;
        --ws-text: #202A22;
        --ws-muted: #5B6259;
        --ws-border: #DCDFD6;
    }

    .stApp { background-color: var(--ws-bg); }

    section[data-testid="stSidebar"] {
        background-color: #EFEEE7;
        border-right: 1px solid var(--ws-border);
    }

    h1, h2, h3 {
        font-family: 'Public Sans', sans-serif;
        font-weight: 600;
        color: var(--ws-text);
        letter-spacing: -0.01em;
    }

    hr { border-color: var(--ws-border); }

    /* Header block */
    .ws-header {
        display: flex;
        align-items: flex-end;
        justify-content: space-between;
        border-bottom: 1px solid var(--ws-border);
        padding-bottom: 14px;
        margin-bottom: 22px;
    }
    .ws-header h1 { margin: 0; font-size: 26px; }
    .ws-header .ws-subtitle { color: var(--ws-muted); font-size: 14px; margin-top: 4px; }
    .ws-header .ws-meta { text-align: right; color: var(--ws-muted); font-size: 13px; line-height: 1.5; }

    /* KPI cards */
    .kpi-row { display: flex; gap: 14px; margin-bottom: 6px; flex-wrap: wrap; }
    .kpi-card {
        flex: 1 1 160px;
        background: #FFFFFF;
        border: 1px solid var(--ws-border);
        border-left: 4px solid #999;
        border-radius: 4px;
        padding: 18px 20px;
    }
    .kpi-label {
        font-size: 12px;
        color: var(--ws-muted);
        margin-bottom: 8px;
        display: flex;
        align-items: center;
    }
    .kpi-value { font-size: 24px; font-weight: 600; color: var(--ws-text); line-height: 1.2; margin-bottom: 6px; }
    .kpi-sub { font-size: 11px; color: var(--ws-muted); }

    /* "?" help badge, replaces st.metric's built-in tooltip since these are custom cards */
    .kpi-help {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 14px;
        height: 14px;
        border-radius: 50%;
        border: 1px solid var(--ws-muted);
        color: var(--ws-muted);
        font-size: 10px;
        margin-left: 6px;
        cursor: help;
        position: relative;
        flex-shrink: 0;
    }
    .kpi-help:hover::after {
        content: attr(data-tip);
        position: absolute;
        bottom: 145%;
        left: 50%;
        transform: translateX(-50%);
        background: #202A22;
        color: #F6F5F1;
        padding: 8px 10px;
        border-radius: 4px;
        font-size: 11px;
        line-height: 1.4;
        white-space: normal;
        width: 200px;
        text-align: left;
        z-index: 10;
        box-shadow: 0 2px 8px rgba(0,0,0,0.18);
    }

    /* Section labels above chart groups */
    .ws-section-label {
        font-size: 13px;
        font-weight: 600;
        color: var(--ws-text);
        margin: 10px 0 4px 0;
    }
    .ws-section-note { font-size: 12px; color: var(--ws-muted); margin: 10px 0 16px 0; }

    button[data-baseweb="tab"] { font-family: 'Public Sans', sans-serif; font-weight: 500; }

    [data-testid="stDataFrame"] { border: 1px solid var(--ws-border); border-radius: 4px; }

    /* Extra breathing room around each chart card so titles/axis labels don't
       crowd the card edge, and so charts have a stable box before a tab is
       first opened (prevents the Vega-Lite "renders squashed until you touch
       the tab" sizing bug). */
    [data-testid="stVegaLiteChart"] {
        padding: 14px 10px 6px 4px;
        min-height: 1px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def altair_theme():
    return {
        "config": {
            "font": "Public Sans",
            "axis": {
                "labelFont": "Public Sans",
                "titleFont": "Public Sans",
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
                "labelFont": "Public Sans",
                "titleFont": "Public Sans",
                "labelFontSize": 11,
                "titleFontSize": 12,
                "titleFontWeight": 500,
                "titlePadding": 8,
                "padding": 8,
            },
            "title": {
                "font": "Public Sans",
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


def kpi_card_html(label: str, value: str, sub: str, color: str, help_text: str = "") -> str:
    help_badge = f'<span class="kpi-help" data-tip="{help_text}">?</span>' if help_text else ""
    return (
        f'<div class="kpi-card" style="border-left-color:{color}">'
        f'<div class="kpi-label">{label}{help_badge}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-sub">{sub}</div>'
        f"</div>"
    )


def _ensure_mock_images() -> dict:
    """Create one placeholder capture per category (idempotent — skips if already on disk).

    TODO(real-data): once the classify server exists, it writes a unique image
    per transaction to MOCK_IMAGE_DIR (or wherever real captures live) and sets
    `image_path` in the transactions table accordingly. This function and the
    "one shared placeholder per category" behavior in _generate_mock_data go
    away entirely at that point — every row will point at its own real photo.
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
def load_data() -> pd.DataFrame:
    """Load transaction log. Falls back to mock data if no DB is present yet."""
    if os.path.exists(DB_PATH):
        # TODO(real-data): this branch activates automatically once the classify
        # server has created waste_sorter.db and started logging rows.
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT * FROM transactions", conn)
        conn.close()
    else:
        df = _generate_mock_data()

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


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

df_raw = load_data()

st.sidebar.markdown("### Filters")

if not os.path.exists(DB_PATH):
    st.sidebar.info("Showing generated mock data — no live database found yet.")

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

st.markdown(
    f"""
    <div class="ws-header">
        <div>
            <h1>Waste Sorter — Operations Dashboard</h1>
            <div class="ws-subtitle">Disposal recommendation activity for Iowa City waste management officials</div>
        </div>
        <div class="ws-meta">
            {len(df):,} transactions in range<br>
            Updated {datetime.now().strftime("%b %-d, %Y · %-I:%M %p")}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
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

sla_color = "#2F6B4F" if sla_compliance >= 90 else ("#B8792F" if sla_compliance >= 75 else "#B23A2E")
retry_color = "#8B9388" if retry_rate < 10 else ("#B8792F" if retry_rate < 20 else "#B23A2E")

cards = [
    kpi_card_html(
        "Total items", f"{total:,}", "in selected range", "#5B6259",
        "Count of all disposal transactions logged for the current filters.",
    )
]
for cat, meta in CATEGORIES.items():
    cards.append(
        kpi_card_html(
            f"{cat.capitalize()}", f"{category_pcts.get(cat, 0):.0f}%", "of total volume", meta["color"],
            f"Share of transactions classified as {cat}.",
        )
    )
cards.append(
    kpi_card_html(
        "Avg latency", f"{avg_latency_s:.2f}s", f"SLA target ≤ {SLA_SECONDS}s (C1.1)", "#3D6B8C",
        f"Average time from item presentation to displayed recommendation. SLA target: ≤{SLA_SECONDS}s per constraint C1.1.",
    )
)
cards.append(
    kpi_card_html(
        "SLA compliance", f"{sla_compliance:.0f}%", "within 5s target", sla_color,
        f"Percentage of transactions that received a recommendation within the {SLA_SECONDS}s SLA (C1.1).",
    )
)
cards.append(
    kpi_card_html(
        "Retry rate", f"{retry_rate:.1f}%", "visual input not recognized (R1.1.6)", retry_color,
        "Percentage of visual-identification attempts that weren't recognized and prompted the user to retry (R1.1.6).",
    )
)

st.markdown(f'<div class="kpi-row">{"".join(cards)}</div>', unsafe_allow_html=True)
st.write("")
st.divider()


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

    hour_col, spacer = st.columns([1, 2])
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
    st.markdown(
        '<div class="ws-section-note">Helps size pickup/service schedules — pick a single day to '
        "check its shape, or leave on \"All days\" to see the aggregate pattern.</div>",
        unsafe_allow_html=True,
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
        st.markdown(
            '<div class="ws-section-note">R1.1.3–R1.1.7: barcode is attempted first; '
            "visual identification is the fallback.</div>",
            unsafe_allow_html=True,
        )

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
    p99 = df["latency_ms"].quantile(0.99)
    max_latency = df["latency_ms"].max()
    over_sla_count = int((df["latency_ms"] > SLA_SECONDS * 1000).sum())

    heldout_color = "#2F6B4F" if LATEST_HELDOUT_ACCURACY >= ACCURACY_TARGET else "#B23A2E"

    stat_cards = [
        kpi_card_html(
            "Held-out accuracy", f"{LATEST_HELDOUT_ACCURACY:.0%}", f"target ≥{ACCURACY_TARGET:.0%} (C1.2)", heldout_color,
            f"Accuracy measured against a labeled held-out test set of Iowa City items, "
            f"last run {LATEST_HELDOUT_ACCURACY_DATE}. This is separate from live confidence — "
            "it's the actual number C1.2 is judged against.",
        ),
        kpi_card_html(
            "Median latency", f"{p50:.0f}ms", "typical case", "#5B6259",
            "50th percentile latency — half of all transactions were faster than this.",
        ),
        kpi_card_html(
            "P95 latency", f"{p95:.0f}ms", "slower 5% of cases", "#3D6B8C",
            "95th percentile latency. Averages can hide a slow tail — this shows what the "
            "worst-typical case looks like, which matters more for an SLA than the mean.",
        ),
        kpi_card_html(
            "Over SLA", f"{over_sla_count:,}", f"of {total:,} transactions", "#B8792F" if over_sla_count else "#8B9388",
            f"Count of transactions that took longer than the {SLA_SECONDS}s SLA (C1.1) to return a recommendation.",
        ),
    ]
    st.markdown(f'<div class="kpi-row">{"".join(stat_cards)}</div>', unsafe_allow_html=True)
    st.write("")

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
    st.divider()

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
        st.markdown(
            f'<div class="ws-section-note">C1.2 target: ≥{ACCURACY_TARGET:.0%} held-out test accuracy. '
            "Live confidence is a proxy signal, not a substitute for the held-out accuracy evaluation.</div>",
            unsafe_allow_html=True,
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
    st.markdown('<div class="ws-section-label">Transaction log</div>', unsafe_allow_html=True)
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

st.divider()
st.caption(
    "TODO(real-data): once the classify server is deployed and writing to "
    f"`{DB_PATH}`, this dashboard switches from mock data to the live log automatically."
)