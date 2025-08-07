import streamlit as st
import pandas as pd
from pathlib import Path
import datetime
import base64
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader

st.set_page_config(layout="centered", page_title="Data Management | Byte Consulting")
LOGO_PATH = "assets/logo.png"

# --- AUTHENTICATION SETUP ---
import streamlit as st
import yaml
config = st.secrets["config"]
credentials = yaml.safe_load(config['credentials'])     # <-- THIS MUST BE HERE
cookie = yaml.safe_load(config['cookie'])

authenticator = stauth.Authenticate(
    credentials,
    cookie['name'],
    cookie['key'],
    cookie['expiry_days']
)

# --- LOGIN WIDGET ---
# Step 1: Render the login form. This function now returns None.
authenticator.login()

# Step 2: Retrieve the user's info from st.session_state, which the widget populates.
name = st.session_state.get("name")
authentication_status = st.session_state.get("authentication_status")
username = st.session_state.get("username")

# --- MAIN APP LOGIC (WRAPPED IN AUTHENTICATION CHECK) ---
if authentication_status:
    # --- USER-SPECIFIC DIRECTORY SETUP ---
    DATA_CACHE_DIR = Path(f"data_cache/{username}")
    DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)


    # --- HELPER FUNCTIONS ---
    def get_saved_datasets():
        """Finds all saved .parquet files in the user's directory."""
        return sorted([f.stem for f in DATA_CACHE_DIR.glob("*.parquet")], reverse=True)


    def process_and_save_csv(uploaded_file, dataset_name, action='overwrite'):
        """Reads, processes, and saves a CSV to a user-specific Parquet file."""
        save_path = DATA_CACHE_DIR / f"{dataset_name}.parquet"
        try:
            new_df = pd.read_csv(uploaded_file)
            expected_cols = ['Restaurant ID', 'Restaurant name', 'Subzone', 'City', 'Overview', 'Metric', 'Attribute',
                             'Value']
            if not all(col in new_df.columns for col in expected_cols):
                st.error(f"CSV format error! Missing one or more required columns: {expected_cols}")
                return False
            new_df['Date'] = pd.to_datetime(new_df['Attribute'], errors='coerce')
            if new_df['Date'].isnull().any():
                st.error(
                    "Date format error! Could not parse some dates. Please use YYYY-MM-DD, DD-MM-YYYY, or DD-Mon-YYYY.")
                return False
            new_df['Value'] = pd.to_numeric(new_df['Value'], errors='coerce')
            new_df.dropna(subset=['Value', 'Date'], inplace=True)

            if action == 'append' and save_path.exists():
                existing_df = pd.read_parquet(save_path)
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                combined_df.drop_duplicates(subset=['Restaurant ID', 'Metric', 'Date'], keep='last', inplace=True)
                final_df = combined_df
                success_message = f"✅ Successfully appended data to **`{dataset_name}.parquet`**"
            else:
                final_df = new_df
                success_message = f"✅ Successfully created new dataset: **`{dataset_name}.parquet`**"
            final_df.to_parquet(save_path)
            st.toast(success_message, icon="🎉")
            return True
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            return False


    def set_selected_dataset():
        """Callback to update session state when the selectbox changes."""
        st.session_state.selected_dataset_name = st.session_state.dataset_selector


    # --- SIDEBAR ---
    with st.sidebar:
        st.write(f'Welcome *{name}*')
        authenticator.logout('Logout', 'sidebar')
        st.title("Navigation")
        st.page_link("home.py", label="Data Management Hub", icon="🏠")
        st.page_link("pages/Analysis_Dashboard.py", label="Analysis Dashboard", icon="📊")

    # --- MAIN PAGE UI ---
    if Path(LOGO_PATH).exists():
        logo_html = f"""
        <style>
            .logo-container {{
                display: flex;
                justify-content: center;
                margin-bottom: 5px; /* reduced space below the logo */
                margin-top: -60px;  /* move logo upwards to reduce white space */
            }}
            .logo-img {{
                transition: transform 0.4s ease, box-shadow 0.4s ease;
                border-radius: 10px;
                cursor: pointer;
                box-shadow: 0px 8px 15px rgba(0, 0, 0, 0.1);
            }}
            .logo-img:hover {{
                transform: scale(1.03) perspective(1000px) rotateY(-5deg);
                box-shadow: 0px 12px 25px rgba(242, 92, 54, 0.2);
            }}
            /* Optional: Reduce space above Streamlit main content section */
            section.main > div:first-child {{ padding-top: 0rem !important; }}
        </style>
        <div class="logo-container">
            <img src="data:image/png;base64,{base64.b64encode(open(LOGO_PATH, "rb").read()).decode()}" class="logo-img" width="350">
        </div>
        """
        st.markdown(logo_html, unsafe_allow_html=True)

    st.markdown("""<style>h1 {{ color: #3A3A3A; text-align: center; }} h2, h3 {{ color: #F25C36; }} </style>""",
                unsafe_allow_html=True)

    st.title("Aggregator Growth Analytics Hub")
    st.header("1. Data Management")
    st.markdown("Select an existing dataset for analysis, or upload a new file below.")

    st.subheader("Select Dataset for Analysis")
    saved_datasets = get_saved_datasets()

    if not saved_datasets:
        st.warning("No datasets found for your account. Please upload a new CSV file below to begin.")
    else:
        try:
            # Set default index for selectbox
            current_index = saved_datasets.index(st.session_state.get('selected_dataset_name'))
        except (ValueError, TypeError):
            current_index = 0

        st.selectbox(
            "Choose one of your datasets:",
            saved_datasets,
            index=current_index,
            key="dataset_selector",
            on_change=set_selected_dataset,
            help="This dataset will be used across all analysis pages."
        )
        # Set initial session state for selected dataset
        if 'selected_dataset_name' not in st.session_state and saved_datasets:
            st.session_state.selected_dataset_name = saved_datasets[0]

        if st.session_state.get("selected_dataset_name"):
            st.markdown(
                f"Selected: **`{st.session_state.selected_dataset_name}.parquet`**. Go to the **Analysis Dashboard** to explore.")

    st.markdown("---")
    st.subheader("Upload New CSV File")

    with st.expander("Click here to upload", expanded=False):
        with st.form(key="Click here to upload", clear_on_submit=True):
            uploaded_file = st.file_uploader("Upload your CSV data file", type=['csv'])
            action_choice = st.radio(
                "What would you like to do?",
                ("Create a new dataset", "Append to an existing dataset"),
                horizontal=True, key="action_choice", disabled=(len(saved_datasets) == 0)
            )
            dataset_name = ""
            action = ""
            if action_choice == "Create a new dataset":
                dataset_name = st.text_input("Enter a name for the new dataset",
                                             f"dataset_{datetime.date.today().strftime('%Y_%m_%d')}")
                action = 'overwrite'
            elif action_choice == "Append to an existing dataset":
                if saved_datasets:
                    dataset_name = st.selectbox("Choose a dataset to append to:", saved_datasets)
                    action = 'append'
                else:
                    st.info("No datasets exist yet. Please create a new one first.")

            submitted = st.form_submit_button("Process and Save File", type="primary", use_container_width=True)
            if submitted:
                if uploaded_file and dataset_name:
                    with st.spinner(f"Processing your file..."):
                        if process_and_save_csv(uploaded_file, dataset_name, action=action):
                            st.rerun()
                else:
                    st.warning("Please provide a file and specify a dataset name or selection.")

# This part runs if the user is NOT logged in
elif authentication_status is False:
    st.error('Username/password is incorrect')
elif authentication_status is None:
    st.warning('Please enter your username and password to access the app.')
