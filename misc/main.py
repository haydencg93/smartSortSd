import streamlit as st
import pandas as pd
import plotly.express as px

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Smart Sort Dashboard", layout="wide")
st.title("♻️ Smart Sort Sustainability Dashboard")
st.markdown("Monitor waste diversion statistics, track contamination, and view real-time disposal events.")

# --- DATA LOADING ---
@st.cache_data
def load_data():
    # In production, replace with reading edge/cloud log files: pd.read_csv("disposal_logs.csv")
    data = {
        'timestamp': pd.date_range(start='2026-08-01 08:00:00', periods=1200, freq='35min'),
        'detected_item': (['plastic', 'glass', 'cardboard', 'trash', 'metal', 'paper'] * 200),
        'confidence_score': ([0.88, 0.92, 0.75, 0.99, 0.85, 0.91] * 200),
    }
    df = pd.DataFrame(data)
    
    # Iowa City logic mapping
    iowa_city_rules = {
        'cardboard': 'Recycling', 
        'metal': 'Recycling', 
        'paper': 'Recycling', 
        'plastic': 'Recycling', 
        'glass': 'Trash', 
        'trash': 'Trash'
    }
    df['disposal_result'] = df['detected_item'].map(iowa_city_rules)
    df['is_mistake'] = df['detected_item'].apply(lambda x: True if x == 'glass' else False)
    
    return df

df = load_data()

# --- TOP LEVEL METRICS ---
st.header("Overall Environmental Impact & Statistics")
col1, col2, col3, col4 = st.columns(4)

total_items = len(df)
recycled_items = len(df[df['disposal_result'] == 'Recycling'])
diversion_rate = (recycled_items / total_items) * 100 if total_items > 0 else 0

most_frequent = df['detected_item'].value_counts().idxmax()
most_common_mistake = df[df['is_mistake']]['detected_item'].value_counts().idxmax()

col1.metric("Total Transactions", f"{total_items:,}")
col2.metric("Waste Diversion Rate", f"{diversion_rate:.1f}%")
col3.metric("Most Frequent Item", most_frequent.capitalize())
col4.metric("Top Contamination Risk", most_common_mistake.capitalize(), help="Items commonly mistaken as recyclable by users in Iowa City.")

# --- VISUALIZATIONS ---
st.markdown("---")
col_chart1, col_chart2 = st.columns(2)

with col_chart1:
    st.subheader("Disposal Counts Over Time")
    df_chart = df.copy()
    df_chart['date'] = df_chart['timestamp'].dt.date
    time_data = df_chart.groupby(['date', 'disposal_result']).size().reset_index(name='counts')
    fig_time = px.line(
        time_data, 
        x='date', 
        y='counts', 
        color='disposal_result', 
        color_discrete_map={'Recycling': '#2ca02c', 'Trash': '#7f7f7f'}
    )
    st.plotly_chart(fig_time, use_container_width=True)

with col_chart2:
    st.subheader("Waste Diversion Breakdown")
    pie_data = df['disposal_result'].value_counts().reset_index()
    pie_data.columns = ['Result', 'Count']
    fig_pie = px.pie(
        pie_data, 
        values='Count', 
        names='Result', 
        color='Result', 
        color_discrete_map={'Recycling': '#2ca02c', 'Trash': '#7f7f7f'}
    )
    st.plotly_chart(fig_pie, use_container_width=True)

# --- TRANSACTION LOG & FILTERS ---
st.markdown("---")
st.subheader("Real-Time Transaction Log")

# State initialization for pagination
if "rows_to_show" not in st.session_state:
    st.session_state.rows_to_show = 50

# --- COLUMN FILTERS ---
with st.expander("🔍 Filter Log Records", expanded=True):
    f_col1, f_col2, f_col3, f_col4 = st.columns(4)
    
    # 1. Timestamp Filter (Date Range)
    min_date = df['timestamp'].min().date()
    max_date = df['timestamp'].max().date()
    with f_col1:
        date_range = st.date_input(
            "Timestamp Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        
    # 2. Detected Item Filter
    all_items = sorted(df['detected_item'].unique().tolist())
    with f_col2:
        selected_items = st.multiselect("Detected Item", options=all_items, default=all_items)
        
    # 3. Disposal Result Filter
    all_results = sorted(df['disposal_result'].unique().tolist())
    with f_col3:
        selected_results = st.multiselect("Disposal Result", options=all_results, default=all_results)
        
    # 4. Confidence Score Filter
    with f_col4:
        confidence_range = st.slider(
            "Confidence Score Range",
            min_value=0.0,
            max_value=1.0,
            value=(0.0, 1.0),
            step=0.01
        )

# Apply filters
filtered_df = df.copy()

# Date filter handling
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_d, end_d = date_range
    filtered_df = filtered_df[
        (filtered_df['timestamp'].dt.date >= start_d) & 
        (filtered_df['timestamp'].dt.date <= end_d)
    ]
elif isinstance(date_range, tuple) and len(date_range) == 1:
    filtered_df = filtered_df[filtered_df['timestamp'].dt.date == date_range[0]]

# Category and confidence filters
filtered_df = filtered_df[
    (filtered_df['detected_item'].isin(selected_items)) &
    (filtered_df['disposal_result'].isin(selected_results)) &
    (filtered_df['confidence_score'] >= confidence_range[0]) &
    (filtered_df['confidence_score'] <= confidence_range[1])
]

# Sort latest transactions first
display_df = filtered_df[['timestamp', 'detected_item', 'confidence_score', 'disposal_result']].sort_values(by='timestamp', ascending=False)

# --- RECORD COUNTS DISPLAY ---
total_filtered = len(display_df)
current_visible = min(st.session_state.rows_to_show, total_filtered)

st.markdown(
    f"**Showing {current_visible:,} of {total_filtered:,} matching logs** *(Total records in database: {len(df):,})*"
)

# --- TABLE VIEW & PAGINATION ---
st.dataframe(
    display_df.iloc[:st.session_state.rows_to_show],
    use_container_width=True,
    hide_index=True
)

# Pagination Controls
p_col1, p_col2, _ = st.columns([1.5, 1.5, 7])

if current_visible < total_filtered:
    with p_col1:
        if st.button("📥 Load More (50 items)"):
            st.session_state.rows_to_show += 50
            st.rerun()

with p_col2:
    if st.session_state.rows_to_show > 50:
        if st.button("↺ Reset View (50 items)"):
            st.session_state.rows_to_show = 50
            st.rerun()