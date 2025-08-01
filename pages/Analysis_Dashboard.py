import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import datetime
from utils import get_metric_rules
import numpy as np
import base64
import json
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader

st.set_page_config(layout="centered", page_title="Analysis | Byte Consulting")
LOGO_PATH = "assets/logo.png"

# --- AUTHENTICATION SETUP (Must be same as Home.py) ---
import streamlit as st
import yaml
config = st.secrets["config"]
credentials = yaml.safe_load(config['credentials'])
cookie = yaml.safe_load(config['cookie'])

authenticator = stauth.Authenticate(
    credentials,
    cookie["name"],
    cookie["key"],
    cookie["expiry_days"]
)

# Use the authenticator's state from the session
name = st.session_state.get("name")
authentication_status = st.session_state.get("authentication_status")
username = st.session_state.get("username")

# --- MAIN APP LOGIC (WRAPPED IN AUTHENTICATION CHECK) ---
if authentication_status:
    # --- USER-SPECIFIC DIRECTORY & FILE SETUP ---
    DATA_CACHE_DIR = Path(f"data_cache/{username}")
    CUSTOM_METRICS_FILE = DATA_CACHE_DIR / "custom_metrics.json"
    METRIC_PRESETS_FILE = DATA_CACHE_DIR / "metric_presets.json"

    # --- HELPER FUNCTIONS ---
    def load_json_file(file_path, default_data):
        """Generic function to load a JSON file, returning default data on error."""
        if file_path.exists():
            with open(file_path, 'r') as f:
                try:
                    return json.load(f)
                except json.JSONDecodeError:
                    return default_data
        return default_data

    def save_json_file(data_list, file_path):
        """Generic function to save data to a JSON file."""
        with open(file_path, 'w') as f:
            json.dump(data_list, f, indent=4)

    # Pass username to the cached function to ensure user-data isolation
    @st.cache_data(ttl=3600)
    def load_parquet(_username, file_name):
        """Loads a Parquet file from a specific user's data directory."""
        path = Path(f"data_cache/{_username}") / f"{file_name}.parquet"
        if not path.exists():
            st.error(f"Dataset '{file_name}.parquet' not found! Please return to the Home page and re-select.")
            return None
        df = pd.read_parquet(path)
        df['Date'] = pd.to_datetime(df['Date'])
        return df

    def format_value(value, metric_name, is_custom=False):
        if pd.isna(value): return "N/A"
        if is_custom:
            if isinstance(value, (int, np.integer)): return f"{value:,}"
            return f"{value:,.2f}"
        if '%' in metric_name and abs(value) <= 2: return f"{value * 100:.2f}%"
        if '(Rs)' in metric_name or '₹' in metric_name or "Cost" in metric_name: return f"₹ {value:,.2f}"
        if isinstance(value, (int, np.integer)): return f"{value:,}"
        return f"{value:,.2f}"

    def perform_calculation(num1, op, num2):
        if num1 is None or num2 is None or pd.isna(num1) or pd.isna(num2): return np.nan
        if op == '+': return num1 + num2
        if op == '-': return num1 - num2
        if op == '*': return num1 * num2
        if op == '/':
            if num2 == 0: return np.nan
            return num1 / num2
        return np.nan

    @st.cache_data
    def convert_df_to_csv(df):
        return df.to_csv(index=True).encode('utf-8')

    # --- INITIALIZE SESSION STATE & LOAD USER-SPECIFIC CONFIGS ---
    # Custom Metrics
    if 'saved_custom_metrics' not in st.session_state:
        st.session_state.saved_custom_metrics = load_json_file(CUSTOM_METRICS_FILE, [])

    # Metric Presets
    DEFAULT_PRESETS = {
        "Daily": ["Sales (Rs)", "Delivered orders", "Average order value (Rs)"],
        "Weekly": ["Sales (Rs)", "Delivered orders", "Bad orders", "Total complaints"],
        "Monthly": ["Sales (Rs)", "Impressions", "Menu opens", "Cart to orders (%)"]
    }
    if 'metric_presets' not in st.session_state:
        st.session_state.metric_presets = load_json_file(METRIC_PRESETS_FILE, DEFAULT_PRESETS)

    # Check if a dataset was selected on the Home page
    if not st.session_state.get("selected_dataset_name"):
        st.warning("Please select a dataset from the 'Home' page first.")
        st.page_link("Home.py", label="🏠 Go to Home Page")
        st.stop()

    # --- LOAD DATA ---
    df_full = load_parquet(username, st.session_state.get('selected_dataset_name'))
    if df_full is None or df_full.empty:
        st.warning("The selected dataset is empty or could not be loaded. Please choose another on the Home page.")
        st.stop()

    min_date = df_full['Date'].dt.date.min()
    max_date = df_full['Date'].dt.date.max()
    metric_rules = get_metric_rules()
    all_available_metrics = sorted(df_full['Metric'].unique())
    # --- SIDEBAR ---
    with st.sidebar:
        st.write(f'Welcome *{name}*')
        authenticator.logout('Logout', 'sidebar', key='unique_logout_sidebar_dashboard')
        st.title("Navigation")
        st.page_link("Home.py", label="Data Management Hub", icon="🏠")
        st.page_link("pages/Analysis_Dashboard.py", label="Analysis Dashboard", icon="📊")
        st.divider()
        st.header("🔬 Analysis Controls")

        # --- Filter by Hierarchy ---
        st.markdown("#### Filter by Hierarchy")
        selected_brand = st.selectbox("Select Brand", ["All Brands"] + sorted(df_full['Restaurant name'].unique()))
        with st.expander("Advanced Location & Outlet Filters", expanded=False):
            city_df = df_full[
                df_full['Restaurant name'] == selected_brand] if selected_brand != "All Brands" else df_full
            selected_city = st.selectbox("Select City", ["All Cities"] + sorted(city_df['City'].unique()))
            subzone_df = city_df[city_df['City'] == selected_city] if selected_city != "All Cities" else city_df
            selected_subzone = st.selectbox("Select Subzone", ["All Subzones"] + sorted(subzone_df['Subzone'].unique()))
            restaurant_df = subzone_df[
                subzone_df['Subzone'] == selected_subzone] if selected_subzone != "All Subzones" else subzone_df
            if restaurant_df is not None and not restaurant_df.empty:
                restaurant_df.loc[:, 'Res_Display'] = restaurant_df['Restaurant name'] + " (" + restaurant_df[
                    'Restaurant ID'].astype(str) + ")"
                selected_restaurants = st.multiselect("Select Individual Outlets",
                                                      sorted(restaurant_df['Res_Display'].unique()))
            else:
                selected_restaurants = []

        filtered_df = df_full.copy()
        if selected_brand != "All Brands": filtered_df = filtered_df[filtered_df['Restaurant name'] == selected_brand]
        if selected_city != "All Cities": filtered_df = filtered_df[filtered_df['City'] == selected_city]
        if selected_subzone != "All Subzones": filtered_df = filtered_df[filtered_df['Subzone'] == selected_subzone]
        if selected_restaurants:
            selected_ids = [int(res.split('(')[-1][:-1]) for res in selected_restaurants]
            filtered_df = filtered_df[filtered_df['Restaurant ID'].isin(selected_ids)]

        st.markdown("---")

        # --- Analysis Options with Presets ---
        st.markdown("#### Analysis Options")

        def apply_preset():
            preset_name = st.session_state.metric_preset_selector
            if preset_name != "Custom":
                st.session_state.selected_metrics = [
                    m for m in st.session_state.metric_presets.get(preset_name, []) if m in all_available_metrics
                ]

        preset_selection = st.selectbox(
            "Metric Presets",
            ["Custom"] + list(st.session_state.metric_presets.keys()),
            key="metric_preset_selector",
            on_change=apply_preset
        )

        selected_metrics = st.multiselect(
            "Select Metrics to Analyze",
            all_available_metrics,
            key="selected_metrics"
        )

        with st.expander("⚙️ Manage Metric Presets"):
            preset_to_edit = st.selectbox("Choose a preset to modify", list(st.session_state.metric_presets.keys()))
            if preset_to_edit:
                new_metrics_for_preset = st.multiselect(
                    f"Update metrics for '{preset_to_edit}'",
                    all_available_metrics,
                    default=[m for m in st.session_state.metric_presets.get(preset_to_edit, []) if m in all_available_metrics],
                    key=f"preset_edit_{preset_to_edit}"
                )
                if st.button(f"Save '{preset_to_edit}' Preset", use_container_width=True):
                    st.session_state.metric_presets[preset_to_edit] = new_metrics_for_preset
                    save_json_file(st.session_state.metric_presets, METRIC_PRESETS_FILE)
                    st.toast(f"Preset '{preset_to_edit}' updated!", icon="✅")
                    st.rerun()

        st.markdown("---")

        # --- Date & Comparison ---
        st.markdown("#### Date & Comparison")
        is_monthly_view = st.checkbox("Show Monthly Summary", value=False)

        comparison_type = "Monthly"
        if not is_monthly_view:
            comparison_type = st.radio("Select Analysis Mode",
                                       ["Single Date", "Range", "Date vs Date", "Range vs Range"],
                                       index=1, horizontal=True, key="comparison_type_selector")

        p1_start, p1_end, p2_start, p2_end = None, None, None, None

        if comparison_type == "Single Date":
            p1_date = st.date_input("Select Date", max_date, min_value=min_date, max_value=max_date)
            p1_start, p1_end = p1_date, p1_date
        elif comparison_type == "Range":
            c1, c2 = st.columns(2)
            default_start = max_date - datetime.timedelta(days=6)
            p1_start = c1.date_input("Start Date", default_start if default_start >= min_date else min_date,
                                     min_value=min_date, max_value=max_date)
            p1_end = c2.date_input("End Date", max_date, min_value=p1_start, max_value=max_date)
        elif comparison_type == "Date vs Date":
            c1, c2 = st.columns(2)
            default_p1 = max_date - datetime.timedelta(days=7)
            p1_date = c1.date_input("Compare Date", default_p1 if default_p1 >= min_date else min_date,
                                    min_value=min_date, max_value=max_date)
            p2_date = c2.date_input("with Date", max_date, min_value=min_date, max_value=max_date)
            p1_start, p1_end, p2_start, p2_end = p1_date, p1_date, p2_date, p2_date
        elif comparison_type == "Range vs Range":
            st.markdown("**Period 1**")
            c1, c2 = st.columns(2)
            default_p1_end = max_date - datetime.timedelta(days=7)
            default_p1_start = default_p1_end - datetime.timedelta(days=6)
            p1_start = c1.date_input("Period 1 Start", default_p1_start if default_p1_start >= min_date else min_date,
                                     min_value=min_date, max_value=max_date)
            p1_end = c2.date_input("Period 1 End", default_p1_end if default_p1_end >= min_date else min_date,
                                   min_value=p1_start, max_value=max_date)
            st.markdown("**Period 2**")
            c3, c4 = st.columns(2)
            default_p2_start = max_date - datetime.timedelta(days=6)
            p2_start = c3.date_input("Period 2 Start", default_p2_start if default_p2_start >= min_date else min_date,
                                     min_value=min_date, max_value=max_date)
            p2_end = c4.date_input("Period 2 End", max_date, min_value=p2_start, max_value=max_date)

        # --- Analysis Button ---
        if st.button("📊 Generate Analysis", type="primary", use_container_width=True):
            error = False
            if comparison_type != "Monthly":
                if p1_start and p1_end and p1_start > p1_end:
                    st.sidebar.error("Error: Start date cannot be after end date for Period 1."); error = True
                elif p2_start and p2_end and p2_start > p2_end:
                    st.sidebar.error("Error: Start date cannot be after end date for Period 2."); error = True

            if not error:
                st.session_state.analysis_generated = True
                st.session_state.comparison_type = comparison_type
                st.session_state.p1_start, st.session_state.p1_end = p1_start, p1_end
                st.session_state.p2_start, st.session_state.p2_end = p2_start, p2_end
                st.session_state.final_selected_metrics = selected_metrics  # Use a different key to avoid multiselect reset issue
                st.session_state.filtered_df = filtered_df
                st.session_state.filter_selections = {
                    'brand': selected_brand, 'city': selected_city,
                    'subzone': selected_subzone, 'restaurants': selected_restaurants
                }
                for metric in all_available_metrics:
                    default_agg = metric_rules.get(metric, {'agg': 'sum'})['agg'].title()
                    if default_agg == "Mean":
                        default_agg = "Average"
                    st.session_state[f"agg_{metric}"] = default_agg
                st.rerun()

        if st.session_state.get("analysis_generated", False):
            if st.button("🔄 Reset Analysis", use_container_width=True):
                keys_to_delete = [k for k in st.session_state.keys() if
                                  k.startswith('agg_') or k in [
                    'analysis_generated', 'comparison_type',
                    'filter_selections', 'filtered_df',
                    'final_selected_metrics', 'selected_custom_metrics'
                ]]
                for key in keys_to_delete:
                    del st.session_state[key]
                st.rerun()
    # --- UI RENDERING (after sidebar) ---
    if Path(LOGO_PATH).exists():
        logo_html = f"""
        <style>
            .logo-container {{ display: flex; justify-content: center; margin-bottom: 20px; }}
            .logo-img {{ transition: transform 0.4s ease, box-shadow 0.4s ease; border-radius: 10px; cursor: pointer; box-shadow: 0px 8px 15px rgba(0, 0, 0, 0.1); }}
            .logo-img:hover {{ transform: scale(1.03) perspective(1000px) rotateY(-5deg); box-shadow: 0px 12px 25px rgba(242, 92, 54, 0.2); }}
        </style>
        <div class="logo-container"><img src="data:image/png;base64,{base64.b64encode(open(LOGO_PATH, "rb").read()).decode()}" class="logo-img" width="350"></div>"""
        st.markdown(logo_html, unsafe_allow_html=True)

    st.markdown("""<style> h1 {{ color: #3A3A3A; text-align: center; }} h2, h3 {{ color: #F25C36; }} </style>""",
                unsafe_allow_html=True)

    st.title("Analysis Dashboard")
    st.markdown(
        f"<h5 style='text-align: center;'>Analyzing dataset: <strong><code>{st.session_state.get('selected_dataset_name')}.parquet</code></strong></h5>",
        unsafe_allow_html=True)

    if not st.session_state.get("analysis_generated", False):
        st.info("Adjust the controls in the sidebar and click 'Generate Analysis' to begin.")
        st.stop()

    # --- RETRIEVE SESSION STATE DATA ---
    df = st.session_state.filtered_df
    selections = st.session_state.filter_selections
    comparison_type = st.session_state.comparison_type

    title_parts = []
    if selections['restaurants']:
        title_parts.append(f"{len(selections['restaurants'])} selected outlets")
    elif selections['subzone'] != "All Subzones":
        title_parts.append(f"all outlets in {selections['subzone']}")
    elif selections['city'] != "All Cities":
        title_parts.append(f"all outlets in {selections['city']}")
    elif selections['brand'] != "All Brands":
        title_parts.append(f"all '{selections['brand']}' outlets")
    else:
        title_parts.append("All Restaurants")
    st.header(f"Analysis for: {', '.join(title_parts)}")

    # --- MONTHLY ANALYSIS LOGIC ---
    if comparison_type == "Monthly":
        summary_tab, trend_tab = st.tabs(["📊 Performance Summary", "📈 Trend & Contribution"])

        with summary_tab:
            st.subheader("Metric Aggregation Controls")
            num_metrics = len(st.session_state.final_selected_metrics)
            num_cols = min(num_metrics, 4)
            cols = st.columns(num_cols) if num_cols > 0 else []
            for i, metric in enumerate(st.session_state.final_selected_metrics):
                if cols:
                    with cols[i % num_cols]:
                        st.radio(f"**{metric}**", options=['Sum', 'Average'], key=f"agg_{metric}", horizontal=True)

            st.markdown("---")
            st.subheader("Select Custom Metrics to Display")
            selected_custom_metrics = st.multiselect(
                "Choose from your saved custom metrics",
                options=[m['name'] for m in st.session_state.saved_custom_metrics],
                key="selected_custom_metrics"
            )

            # --- Monthly Calculation Logic ---
            metrics_to_display = st.session_state.final_selected_metrics
            custom_metrics_to_display = [m for m in st.session_state.saved_custom_metrics if
                                         m['name'] in selected_custom_metrics]

            # 1. Identify all base metrics needed for calculations
            base_metrics_needed = set(metrics_to_display)
            for cm in custom_metrics_to_display:
                if cm['op1_type'] == 'metric': base_metrics_needed.add(cm['op1_val'])
                if cm['op2_type'] == 'metric': base_metrics_needed.add(cm['op2_val'])

            # 2. Prepare data and get day counts
            monthly_df = df[df['Metric'].isin(base_metrics_needed)].copy()
            monthly_df['Month'] = monthly_df['Date'].dt.to_period('M').astype(str)
            day_counts = monthly_df.groupby('Month')['Date'].nunique()

            # 3. Calculate all needed base values (sum and mean)
            monthly_agg = monthly_df.groupby(['Month', 'Metric'])['Value'].agg(['sum', 'mean']).reset_index()

            # 4. Build the final summary table
            summary_data = []

            # Process standard metrics
            for metric in metrics_to_display:
                user_selection = st.session_state.get(f"agg_{metric}", "Sum")
                if user_selection == "Average":
                    agg_choice = "mean"
                else:
                    agg_choice = "sum"

                metric_data = monthly_agg[monthly_agg['Metric'] == metric].set_index('Month')[agg_choice]
                metric_data.name = metric
                summary_data.append(metric_data)
            # Process selected custom metrics
            # --- Revised code below: handle "number of days" type as well ---
            months = monthly_agg['Month'].unique()

            for cm in custom_metrics_to_display:
                # Get op1_data
                if cm['op1_type'] == 'metric':
                    op1_data = monthly_agg[monthly_agg['Metric'] == cm['op1_val']].set_index('Month')['sum']
                elif cm['op1_type'] == 'number of days':
                    op1_data = pd.Series([day_counts.get(month, np.nan) for month in months], index=months)
                else:
                    op1_data = pd.Series(cm['op1_val'], index=months)

                # Get op2_data
                if cm['op2_type'] == 'metric':
                    op2_data = monthly_agg[monthly_agg['Metric'] == cm['op2_val']].set_index('Month')['sum']
                elif cm['op2_type'] == 'number of days':
                    op2_data = pd.Series([day_counts.get(month, np.nan) for month in months], index=months)
                else:
                    op2_data = pd.Series(cm['op2_val'], index=months)

                # Align data and perform vectorized calculation
                aligned_op1, aligned_op2 = op1_data.align(op2_data, fill_value=np.nan)

                op = cm['op']
                if op == '+':
                    result_series = aligned_op1 + aligned_op2
                elif op == '-':
                    result_series = aligned_op1 - aligned_op2
                elif op == '*':
                    result_series = aligned_op1 * aligned_op2
                elif op == '/':
                    result_series = aligned_op1 / aligned_op2.replace(0, np.nan)
                else:
                    result_series = pd.Series(np.nan, index=aligned_op1.index)

                result_series.name = f"{cm['name']}*"
                summary_data.append(result_series)
        st.markdown("---")
        st.subheader("Monthly Performance Table")
        if summary_data:
            import pandas as pd
            import numpy as np
            import re

            summary_df = pd.concat(summary_data, axis=1).T
            summary_df.columns = [f"{col} ({day_counts.get(col, 0)} days)" for col in summary_df.columns]

            # Format value, handling metrics, custom metrics, and nulls
            def format_value_fixed(value, metric_name):
                if pd.isna(value): return "N/A"
                try:
                    value = float(value)
                except Exception:
                    value = pd.to_numeric(str(value).replace("₹", "").replace(",", "").strip(), errors="coerce")
                if pd.isna(value): return "N/A"
                is_custom = "*" in str(metric_name)
                if is_custom:
                    return f"{value:,.2f}"
                if '%' in metric_name and abs(value) <= 2: return f"{value * 100:.2f}%"
                if '(Rs)' in metric_name or '₹' in metric_name or "Cost" in metric_name: return f"₹ {value:,.2f}"
                if float(value).is_integer(): return f"{int(value):,}"
                return f"{value:,.2f}"

            formatted_rows = []
            for index, row in summary_df.iterrows():
                formatted_row = []
                for col, val in row.items():
                    formatted_row.append(format_value_fixed(val, index))
                formatted_rows.append(formatted_row)

            formatted_df = pd.DataFrame(formatted_rows, index=summary_df.index, columns=summary_df.columns)
            st.dataframe(formatted_df, use_container_width=True)
            csv = convert_df_to_csv(formatted_df)
            st.download_button("📥 Download as CSV", csv, "monthly_analysis.csv", "text/csv")
        else:
            st.warning("No metrics selected for analysis.")

        with trend_tab:
            st.subheader("Monthly Trend Analysis")

            # Combine standard and selected custom metrics for the trend selector
            trend_metric_options = st.session_state.final_selected_metrics + [f"{cm['name']}*" for cm in
                                                                              custom_metrics_to_display]

            trend_metric = st.selectbox("Select a metric to analyze its trend", trend_metric_options,
                                        key="monthly_trend_selector")

            if trend_metric:
                trend_df_data = None
                # Check if it's a custom metric
                if trend_metric.endswith('*'):
                    cm_name = trend_metric[:-1]
                    cm_def = next((m for m in custom_metrics_to_display if m['name'] == cm_name), None)
                    if cm_def:
                        # Find the relevant row in the calculated summary_df
                        if 'summary_df' in locals() and not summary_df.empty:
                            trend_df_data = summary_df.loc[trend_metric].T.reset_index()
                            trend_df_data.columns = ['Month', 'Value']
                            # Clean up month name for plotting
                            trend_df_data['Month'] = trend_df_data['Month'].apply(lambda x: x.split(' ')[0])
                else:
                    # It's a standard metric
                    if 'summary_df' in locals() and not summary_df.empty:
                        trend_df_data = summary_df.loc[trend_metric].T.reset_index()
                        trend_df_data.columns = ['Month', 'Value']
                        trend_df_data['Month'] = trend_df_data['Month'].apply(lambda x: x.split(' ')[0])

                if trend_df_data is not None and not trend_df_data.empty:
                    fig = px.line(trend_df_data, x='Month', y='Value', title=f"<b>Monthly Trend: {trend_metric}</b>",
                                  markers=True)
                    fig.update_layout(hovermode="x unified")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning(
                        f"Could not generate trend data for '{trend_metric}'. Please ensure it's calculated in the summary table first.")

    # --- REGULAR ANALYSIS (NON-MONTHLY) ---
    else:
        p1_start, p1_end, p2_start, p2_end = st.session_state.p1_start, st.session_state.p1_end, st.session_state.p2_start, st.session_state.p2_end
        is_comparison = comparison_type in ["Date vs Date", "Range vs Range"]
        p1_df = df[(df['Date'].dt.date >= p1_start) & (df['Date'].dt.date <= p1_end)]
        p2_df = pd.DataFrame()
        if is_comparison: p2_df = df[(df['Date'].dt.date >= p2_start) & (df['Date'].dt.date <= p2_end)]

        summary_tab, trend_tab = st.tabs(["📊 Performance Summary", "📈 Trend & Contribution"])
        with summary_tab:
            st.subheader("Metric Aggregation Controls")
            num_metrics = len(st.session_state.final_selected_metrics)
            num_cols = min(num_metrics, 4)
            cols = st.columns(num_cols) if num_cols > 0 else []
            for i, metric in enumerate(st.session_state.final_selected_metrics):
                if cols:
                    with cols[i % num_cols]:
                        st.radio(f"**{metric}**", options=['Sum', 'Average'], key=f"agg_{metric}", horizontal=True)

            with st.expander("🔧 Manage Custom Metrics"):
                st.markdown(
                    "Define a new metric using any available metric or a fixed number. Your definitions will be saved to your account.")
                with st.form("custom_metric_form"):
                    new_metric_name = st.text_input("New Metric Name", placeholder="e.g., Ad Cost %")
                    operator = st.selectbox("Operator", ['/', '*', '+', '-'], index=0)
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("**Operand A (Numerator)**")
                        op1_type = st.radio(
                            "Type for A",
                            ["Metric", "Number", "Number of Days"],
                            key="op1_type",
                            horizontal=True
                        )
                        if op1_type == "Metric":
                            op1_val = st.selectbox(
                                "Select Metric A", all_available_metrics, index=None, key="op1_metric_select"
                            )
                        elif op1_type == "Number":
                            op1_val = st.number_input("Enter Number A", value=1.0, format="%.2f", key="op1_num_input")
                        else:  # "Number of Days"
                            op1_val = "num_days"

                    with c2:
                        st.markdown("**Operand B (Denominator)**")
                        op2_type = st.radio(
                            "Type for B",
                            ["Metric", "Number", "Number of Days"],
                            key="op2_type",
                            horizontal=True
                        )
                        if op2_type == "Metric":
                            op2_val = st.selectbox(
                                "Select Metric B", all_available_metrics, index=None, key="op2_metric_select"
                            )
                        elif op2_type == "Number":
                            op2_val = st.number_input("Enter Number B", value=1.0, format="%.2f", key="op2_num_input")
                        else:  # "Number of Days"
                            op2_val = "num_days"

                    submitted = st.form_submit_button("💾 Save Custom Metric", use_container_width=True, type="primary")
                    if submitted:
                        if new_metric_name and op1_val is not None and op2_val is not None:
                            if new_metric_name in [m['name'] for m in st.session_state.saved_custom_metrics]:
                                st.error("A custom metric with this name already exists.")
                            else:
                                new_def = {'name': new_metric_name, 'op': operator,
                                           'op1_type': op1_type.lower(), 'op1_val': op1_val,
                                           'op2_type': op2_type.lower(), 'op2_val': op2_val}
                                st.session_state.saved_custom_metrics.append(new_def)
                                save_json_file(st.session_state.saved_custom_metrics, CUSTOM_METRICS_FILE)
                                st.toast(f"Saved custom metric '{new_metric_name}'!", icon="✅")
                                st.rerun()
                        else:
                            st.warning("Please fill out all fields to create and save a custom metric.")

                st.markdown("---")
                st.write("**Your Saved Metrics:**")
                if not st.session_state.saved_custom_metrics: st.info("No custom metrics saved yet.")
                for i, metric_def in enumerate(st.session_state.saved_custom_metrics):
                    col1, col2 = st.columns([4, 1])
                    formula = f"({metric_def['op1_val']}) {metric_def['op']} ({metric_def['op2_val']})"
                    with col1:
                        st.text(f"• {metric_def['name']} = {formula}")
                    with col2:
                        if st.button("❌ Remove", key=f"remove_metric_{i}", use_container_width=True):
                            st.session_state.saved_custom_metrics.pop(i)
                            save_json_file(st.session_state.saved_custom_metrics, CUSTOM_METRICS_FILE)
                            st.rerun()
            st.markdown("---")
            st.subheader("Select Custom Metrics to Display")
            selected_custom_metrics = st.multiselect(
                "Choose from your saved custom metrics",
                options=[m['name'] for m in st.session_state.saved_custom_metrics],
                key="selected_custom_metrics_standard"
            )

            st.markdown("---")
            st.subheader("Performance Summary Table")
            table_data, calculated_values = [], {'Period 1': {}, 'Period 2': {}}

            # --- Calculation for Standard Analysis ---
            custom_metrics_to_display = [m for m in st.session_state.saved_custom_metrics if
                                         m['name'] in selected_custom_metrics]
            metrics_to_calc = set(st.session_state.final_selected_metrics)
            for cm in custom_metrics_to_display:
                if cm['op1_type'] == 'metric': metrics_to_calc.add(cm['op1_val'])
                if cm['op2_type'] == 'metric': metrics_to_calc.add(cm['op2_val'])

            for metric in metrics_to_calc:
                agg_method = st.session_state.get(f"agg_{metric}", "Sum")
                value1 = p1_df[p1_df['Metric'] == metric]['Value'].sum() if agg_method == 'Sum' else (
                    p1_nonzero['Value'].mean() if not (
                        p1_nonzero := p1_df[(p1_df['Metric'] == metric) & (p1_df['Value'] != 0)]).empty else 0)
                calculated_values['Period 1'][metric] = 0 if pd.isna(value1) else value1
                if is_comparison:
                    value2 = p2_df[p2_df['Metric'] == metric]['Value'].sum() if agg_method == 'Sum' else (
                        p2_nonzero['Value'].mean() if not (
                            p2_nonzero := p2_df[(p2_df['Metric'] == metric) & (p2_df['Value'] != 0)]).empty else 0)
                    calculated_values['Period 2'][metric] = 0 if pd.isna(value2) else value2

            for metric in st.session_state.final_selected_metrics:
                value1 = calculated_values['Period 1'].get(metric, 0)
                row_data = {"Metric": f"{metric} ({st.session_state.get(f'agg_{metric}', 'Sum')})"}
                if is_comparison:
                    value2 = calculated_values['Period 2'].get(metric, 0)
                    rule = metric_rules.get(metric, {'agg': 'sum', 'is_good_when_low': False})
                    delta = value2 - value1
                    p_change = (delta / abs(value1)) * 100 if value1 != 0 else float('inf') if delta > 0 else 0
                    status = "⚪️"
                    if delta != 0:
                        status = "🟢" if (rule['is_good_when_low'] and delta < 0) or (
                                    not rule['is_good_when_low'] and delta > 0) else "🔴"
                    p1_label = f"Period 1 ({p1_start} to {p1_end})" if p1_start != p1_end else f"Date ({p1_start})"
                    p2_label = f"Period 2 ({p2_start} to {p2_end})" if p2_start != p2_end else f"Date ({p2_start})"
                    row_data.update({"Status": status, p1_label: format_value(value1, metric),
                                     p2_label: format_value(value2, metric), "Change": format_value(delta, metric),
                                     "% Change": f"{p_change:+.1f}%" if value1 != 0 else "N/A"})
                else:
                    row_data["Value"] = format_value(value1, metric)
                table_data.append(row_data)

            # ---- CUSTOM METRIC CALCULATION -- REVISED ----
            period1_days = (p1_end - p1_start).days + 1 if p1_start and p1_end else None
            period2_days = (p2_end - p2_start).days + 1 if is_comparison and p2_start and p2_end else None

            for custom_metric in custom_metrics_to_display:
                # Operand 1 - Period 1
                if custom_metric['op1_type'] == 'metric':
                    op1_val1 = calculated_values['Period 1'].get(custom_metric['op1_val'])
                elif custom_metric['op1_type'] == 'number of days':
                    op1_val1 = period1_days
                else:
                    op1_val1 = custom_metric['op1_val']
                # Operand 2 - Period 1
                if custom_metric['op2_type'] == 'metric':
                    op2_val1 = calculated_values['Period 1'].get(custom_metric['op2_val'])
                elif custom_metric['op2_type'] == 'number of days':
                    op2_val1 = period1_days
                else:
                    op2_val1 = custom_metric['op2_val']

                value1 = perform_calculation(op1_val1, custom_metric['op'], op2_val1)
                row_data = {"Metric": f"{custom_metric['name']}*"}
                if is_comparison:
                    # Operand 1 - Period 2
                    if custom_metric['op1_type'] == 'metric':
                        op1_val2 = calculated_values['Period 2'].get(custom_metric['op1_val'])
                    elif custom_metric['op1_type'] == 'number of days':
                        op1_val2 = period2_days
                    else:
                        op1_val2 = custom_metric['op1_val']
                    # Operand 2 - Period 2
                    if custom_metric['op2_type'] == 'metric':
                        op2_val2 = calculated_values['Period 2'].get(custom_metric['op2_val'])
                    elif custom_metric['op2_type'] == 'number of days':
                        op2_val2 = period2_days
                    else:
                        op2_val2 = custom_metric['op2_val']

                    value2 = perform_calculation(op1_val2, custom_metric['op'], op2_val2)
                    delta = value2 - value1
                    p_change = (delta / abs(value1)) * 100 if value1 != 0 else float('inf') if delta > 0 else 0
                    row_data.update(
                        {"Status": "⚪️", p1_label: format_value(value1, custom_metric['name'], is_custom=True),
                         p2_label: format_value(value2, custom_metric['name'], is_custom=True),
                         "Change": format_value(delta, custom_metric['name'], is_custom=True),
                         "% Change": f"{p_change:+.1f}%" if value1 != 0 else "N/A"})
                else:
                    row_data["Value"] = format_value(value1, custom_metric['name'], is_custom=True)
                table_data.append(row_data)

            summary_df = pd.DataFrame(table_data)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
            if not summary_df.empty:
                csv = convert_df_to_csv(summary_df.set_index('Metric'))
                st.download_button("📥 Download as CSV", csv, "analysis_summary.csv", "text/csv")
        with trend_tab:
            st.subheader("Metric Trend Analysis")
            st.info("Custom metrics are not available for trend analysis as they are aggregate calculations.")
            trend_metric = st.selectbox("Select a metric to analyze its trend", st.session_state.final_selected_metrics)
            if trend_metric:
                metric_p1_trend = p1_df[p1_df['Metric'] == trend_metric].groupby('Date')['Value'].sum().reset_index()
                fig = go.Figure()
                p1_name = f'Period 1 ({p1_start} to {p1_end})' if p1_start != p1_end else f'Date: {p1_start}'
                fig.add_trace(go.Scatter(x=metric_p1_trend['Date'], y=metric_p1_trend['Value'], mode='lines+markers',
                                         name=p1_name, line=dict(color='royalblue')))
                if is_comparison:
                    metric_p2_trend = p2_df[p2_df['Metric'] == trend_metric].groupby('Date')[
                        'Value'].sum().reset_index()
                    p2_name = f'Period 2 ({p2_start} to {p2_end})' if p2_start != p2_end else f'Date: {p2_start}'
                    fig.add_trace(
                        go.Scatter(x=metric_p2_trend['Date'], y=metric_p2_trend['Value'], mode='lines+markers',
                                   name=p2_name, line=dict(color='darkorange')))
                fig.update_layout(title_text=f"<b>Daily Trend: {trend_metric}</b>", hovermode="x unified",
                                  legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                st.plotly_chart(fig, use_container_width=True)

                if is_comparison:
                    st.subheader(f"Contribution to Change in {trend_metric}")
                    st.caption("This shows which items contributed most to the change between the two periods.")
                    breakdown_dim = None
                    if selections['brand'] == 'All Brands':
                        breakdown_dim = 'Restaurant name'
                    elif selections['city'] == 'All Cities':
                        breakdown_dim = 'City'
                    elif selections['subzone'] == 'All Subzones':
                        breakdown_dim = 'Subzone'
                    elif len(selections['restaurants']) > 1:
                        breakdown_dim = 'Res_Display'

                    if breakdown_dim:
                        p1_contrib = p1_df[p1_df['Metric'] == trend_metric].groupby(breakdown_dim)['Value'].sum()
                        p2_contrib = p2_df[p2_df['Metric'] == trend_metric].groupby(breakdown_dim)['Value'].sum()
                        contrib_df = pd.DataFrame({'Period 1': p1_contrib, 'Period 2': p2_contrib}).fillna(0)
                        contrib_df['Change'] = contrib_df['Period 2'] - contrib_df['Period 1']
                        contrib_df = contrib_df[contrib_df['Change'] != 0].sort_values('Change', ascending=False).head(
                            20)
                        if not contrib_df.empty:
                            contrib_fig = px.bar(contrib_df, x=contrib_df.index, y='Change', color='Change',
                                                 color_continuous_scale='RdYlGn',
                                                 labels={'x': breakdown_dim.replace('_', ' ').title(),
                                                         'Change': 'Contribution to Change'},
                                                 title=f"Top Contributors to Change in '{trend_metric}'")
                            contrib_fig.update_layout(coloraxis_showscale=False)
                            st.plotly_chart(contrib_fig, use_container_width=True)
                        else:
                            st.info(
                                f"No change detected in '{trend_metric}' at the {breakdown_dim.replace('_', ' ').title()} level.")
                    else:
                        st.info(
                            "Contribution analysis is available when aggregating multiple items (e.g., All Brands, All Cities, or multiple outlets).")
                else:
                    st.info(
                        "Contribution analysis is not applicable for a single period. Please select a comparison mode like 'Date vs Date' or 'Range vs Range'.")

# This part runs if the user is not logged in
elif authentication_status is False:
    st.error('Username/password is incorrect')
elif authentication_status is None:
    st.warning('You must be logged in to access this page. Please go to the Home page to log in.')
    st.page_link("Home.py", label="🏠 Go to Home Page")
