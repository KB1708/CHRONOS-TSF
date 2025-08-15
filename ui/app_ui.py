# ui/app_ui.py

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

def call_forecast_api(df):
    """Sends a dataframe to the backend API and returns the JSON response."""
    # Convert dataframe to a CSV in-memory file
    csv_buffer = df.to_csv(index=False).encode('utf-8')
    
    files = {'file': ('data.csv', csv_buffer, 'text/csv')}
    
    try:
        response = requests.post(API_URL, files=files, timeout=600)
        response.raise_for_status()  # Raise an exception for bad status codes
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {e}")
        st.error(f"Response content: {response.content.decode()}")
        return None

# --- UI Layout ---

# Sidebar for controls
st.sidebar.title("⚙️ Controls")
st.sidebar.markdown("Select a dataset and click 'Generate Forecast' to see the results.")

# Use session state to store the API response
if 'api_response' not in st.session_state:
    st.session_state['api_response'] = None

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
    help="Your CSV should have 'Date' and 'Close' columns."
)

# Forecast button
if st.sidebar.button("Generate Forecast", type="primary"):
    df = None
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.sidebar.success("Custom file uploaded successfully!")
        except Exception as e:
            st.sidebar.error(f"Error reading file: {e}")
    else:
        # Load the selected sample data
        df = pd.read_csv(sample_data_options[selected_sample])

    if df is not None:
        with st.spinner('Backend is processing... This may take a few minutes for the first run.'):
            start_time = time.time()
            st.session_state['api_response'] = call_forecast_api(df)
            end_time = time.time()
            st.sidebar.info(f"Processing took {end_time - start_time:.2f} seconds.")

# Main page for displaying results
st.title("⏱️ Time Series Forecasting Comparison")
st.markdown("Comparing **Amazon Chronos (Generative AI)** vs. **Traditional ARIMA**")

if st.session_state['api_response']:
    response_data = st.session_state['api_response']
    
    # --- Metrics Display ---
    st.subheader("📊 Performance Metrics")
    st.markdown("Lower is better for both **Mean Absolute Error (MAE)** and **Root Mean Squared Error (RMSE)**.")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Chronos (GenAI)")
        chronos_metrics = response_data['chronos_forecast']['metrics']
        st.metric(label="MAE", value=f"{chronos_metrics['mae']:.2f}")
        st.metric(label="RMSE", value=f"{chronos_metrics['rmse']:.2f}")

    with c2:
        st.subheader("ARIMA (Traditional)")
        arima_metrics = response_data['arima_forecast']['metrics']
        st.metric(label="MAE", value=f"{arima_metrics['mae']:.2f}")
        st.metric(label="RMSE", value=f"{arima_metrics['rmse']:.2f}")
        
    st.info("Note: Metrics are calculated on the forecasted period against the actual holdout data.")

    # --- Chart Display ---
    st.subheader("📈 Forecast Visualization")
    
    # Create the plotly figure
    fig = go.Figure()

    # Add historical data
    fig.add_trace(go.Scatter(
        x=response_data['historical_dates'],
        y=response_data['historical_values'],
        mode='lines',
        name='Historical Data',
        line=dict(color='royalblue')
    ))

    # Add actual data for the forecast period (the ground truth)
    fig.add_trace(go.Scatter(
        x=response_data['forecast_dates'],
        y=response_data['chronos_forecast']['forecast_values'],  # Using one of the forecasts to get the y-axis truth
        mode='lines',
        name='Actual Values (Holdout)',
        line=dict(color='royalblue', dash='dot')
    ))

    # Add ARIMA forecast
    fig.add_trace(go.Scatter(
        x=response_data['forecast_dates'],
        y=response_data['arima_forecast']['forecast_values'],
        mode='lines',
        name='ARIMA Forecast',
        line=dict(color='orange')
    ))

    # Add Chronos forecast
    fig.add_trace(go.Scatter(
        x=response_data['forecast_dates'],
        y=response_data['chronos_forecast']['forecast_values'],
        mode='lines',
        name='Chronos Forecast',
        line=dict(color='lightgreen', width=3) # Make Chronos line stand out
    ))

    # Update layout for a professional look
    fig.update_layout(
        title="Model Forecasts vs. Historical Data",
        xaxis_title="Date",
        yaxis_title="Value",
        legend_title="Series",
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Select a dataset and click 'Generate Forecast' to get started.")