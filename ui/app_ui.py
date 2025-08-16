import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="Chronos Forecast Comparator",
    page_icon="⏱️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- API Endpoint ---
API_URL = "http://127.0.0.1:8000/forecast/"

# --- Helper Functions ---

def call_forecast_api(df, date_col, value_col):
    """Sends a dataframe and column names to the backend API."""
    csv_buffer = df.to_csv(index=False).encode('utf-8')
    
    files = {'file': ('data.csv', csv_buffer, 'text/csv')}
    data = {'date_col': date_col, 'value_col': value_col}
    
    try:
        response = requests.post(API_URL, files=files, data=data, timeout=600)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {e}")
        try:
            st.error(f"Response content: {response.content.decode()}")
        except (AttributeError, UnicodeDecodeError):
            st.error("Could not decode response content.")
        return None

# --- UI Layout ---

st.sidebar.title("⚙️ Controls")
st.sidebar.markdown("Select a dataset and click 'Generate Forecast' to see the results.")

if 'api_response' not in st.session_state:
    st.session_state['api_response'] = None
if 'df' not in st.session_state:
    st.session_state['df'] = None

# Sample data selection
sample_data_options = {
    "NIFTY 50 Daily Prices": "data/nifty_50_daily.csv",
    "Reliance Industries Daily Prices": "data/reliance_daily.csv"
}
selected_sample = st.sidebar.selectbox("Choose a sample dataset:", options=list(sample_data_options.keys()))

# File uploader for custom data
uploaded_file = st.sidebar.file_uploader(
    "Or upload your own CSV file",
    type="csv",
    help="Your CSV should have a date column and a value column."
)

if uploaded_file is not None:
    try:
        st.session_state['df'] = pd.read_csv(uploaded_file)
        st.sidebar.success("Custom file uploaded!")
    except Exception as e:
        st.sidebar.error(f"Error reading file: {e}")
        st.session_state['df'] = None
elif st.sidebar.button("Load Sample Data"):
    st.session_state['df'] = pd.read_csv(sample_data_options[selected_sample])

# Main page for displaying results
st.title("⏱️ Time Series Forecasting Comparison")
st.markdown("Comparing **Amazon Chronos (GenAI)** vs. **Traditional Models (ARIMA, ETS)**")

if st.session_state['df'] is not None:
    df = st.session_state['df']
    st.subheader("Data Preview & Configuration")
    st.dataframe(df.head())

    col1, col2 = st.columns(2)
    with col1:
        date_col = st.selectbox("Select your Date Column:", df.columns)
    with col2:
        value_col = st.selectbox("Select your Value Column:", df.columns, index=1 if len(df.columns) > 1 else 0)
    
    if st.button("Generate Forecast", type="primary"):
        with st.spinner('Backend is processing... This may take a few minutes.'):
            start_time = time.time()
            st.session_state['api_response'] = call_forecast_api(df, date_col, value_col)
            end_time = time.time()
            st.info(f"Processing took {end_time - start_time:.2f} seconds.")

if st.session_state['api_response']:
    response_data = st.session_state['api_response']
    
    st.subheader("📊 Performance Metrics")
    st.markdown("Lower is better for all error metrics.")

    # --- FIX: Add a third column for ETS metrics ---
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Chronos (GenAI)")
        chronos_metrics = response_data['chronos_forecast']['metrics']
        st.metric(label="MAE", value=f"{chronos_metrics['mae']:.2f}")
        st.metric(label="RMSE", value=f"{chronos_metrics['rmse']:.2f}")

    with c2:
        st.subheader("ARIMA")
        arima_metrics = response_data['arima_forecast']['metrics']
        st.metric(label="MAE", value=f"{arima_metrics['mae']:.2f}")
        st.metric(label="RMSE", value=f"{arima_metrics['rmse']:.2f}")
    
    with c3:
        st.subheader("ETS")
        ets_metrics = response_data['ets_forecast']['metrics']
        st.metric(label="MAE", value=f"{ets_metrics['mae']:.2f}")
        st.metric(label="RMSE", value=f"{ets_metrics['rmse']:.2f}")
    # --- END OF FIX ---
        
    st.subheader("📈 Forecast Visualization")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=response_data['historical_dates'], y=response_data['historical_values'], mode='lines', name='Historical Data', line=dict(color='royalblue')))
    fig.add_trace(go.Scatter(x=response_data['forecast_dates'], y=response_data['actual_values'], mode='lines', name='Actual Values (Holdout)', line=dict(color='black', width=3)))
    fig.add_trace(go.Scatter(x=response_data['forecast_dates'], y=response_data['arima_forecast']['forecast_values'], mode='lines', name='ARIMA Forecast', line=dict(color='orange', dash='dot')))
    fig.add_trace(go.Scatter(x=response_data['forecast_dates'], y=response_data['chronos_forecast']['forecast_values'], mode='lines', name='Chronos Forecast', line=dict(color='lightgreen', width=2, dash='dash')))
    
    # --- FIX: Add the ETS forecast to the plot ---
    fig.add_trace(go.Scatter(x=response_data['forecast_dates'], y=response_data['ets_forecast']['forecast_values'], mode='lines', name='ETS Forecast', line=dict(color='mediumpurple', dash='longdash')))
    # --- END OF FIX ---

    fig.update_layout(title="Model Forecasts vs. Historical Data", xaxis_title="Date", yaxis_title="Value", legend_title="Series", height=600)
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Select a dataset and click 'Generate Forecast' to get started.")
