# ui/app_ui.py

import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
import plotly.express as px
import time

# Page Configuration
st.set_page_config(page_title="Chronos Forecast Comparator", page_icon="⏱️", layout="wide")

API_URL = "http://127.0.0.1:8000/forecast/"

def call_forecast_api(df):
    csv_buffer = df.to_csv(index=False).encode('utf-8')
    files = {'file': ('data.csv', csv_buffer, 'text/csv')}
    try:
        response = requests.post(API_URL, files=files, timeout=600)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {e}")
        return None

# Initialize session state
if 'api_response' not in st.session_state:
    st.session_state.api_response = None
if 'original_df' not in st.session_state:
    st.session_state.original_df = None

# Sidebar
with st.sidebar:
    st.title("⚙️ Controls")
    sample_options = {
        "NIFTY 50": "data/nifty_50_daily.csv",
        "Reliance": "data/reliance_daily.csv"
    }
    selected_sample = st.selectbox("Choose a sample dataset:", options=list(sample_options.keys()))
    uploaded_file = st.file_uploader("Or upload your own CSV file", type="csv")

    if st.button("Generate Forecast", type="primary"):
        df = pd.read_csv(uploaded_file) if uploaded_file else pd.read_csv(sample_options[selected_sample])
        if "Date" not in df.columns or "Close" not in df.columns:
            st.sidebar.error("CSV must contain 'Date' and 'Close' columns.")
        else:
            df['item_id'] = 'series_1'
            st.session_state.original_df = df.copy()
            with st.spinner('Training models and generating forecast...'):
                st.session_state.api_response = call_forecast_api(df)

# Main Page
st.title("⏱️ Time Series Forecasting Comparison")

if st.session_state.api_response:
    response = st.session_state.api_response

    # --- Forecast Visualization ---
    st.subheader("Forecast Visualization")
    fig_forecast = go.Figure()

    all_dates = response['historical_dates'] + response['forecast_dates']
    all_values = response['historical_values'] + response['actual_values']
    
    fig_forecast.add_trace(go.Scatter(
        x=all_dates, y=all_values, mode='lines', name='Historical & Actual Data',
        line=dict(color='royalblue', width=2)
    ))
    
    fig_forecast.add_trace(go.Scatter(
        x=response['forecast_dates'], y=response['arima_forecast']['forecast_values'],
        mode='lines', name='ARIMA Forecast', line=dict(color='yellow', dash='dash')
    ))
    fig_forecast.add_trace(go.Scatter(
        x=response['forecast_dates'], y=response['chronos_forecast']['forecast_values'],
        mode='lines', name='Chronos Forecast', line=dict(color='lightgreen', dash='dash')
    ))
    
    # --- THIS IS THE FIX ---
    # Replaced add_vline with the more robust add_shape to avoid the TypeError
    split_date = response['historical_dates'][-1]
    fig_forecast.add_shape(
        type="line", x0=split_date, y0=0, x1=split_date, y1=1,
        yref="paper", line=dict(color="red", width=2, dash="dash")
    )
    fig_forecast.add_annotation(
        x=split_date, y=1, yref="paper",
        text="Train/Test Split", showarrow=False, yshift=10
    )
    # --- END OF FIX ---

    fig_forecast.update_layout(title="Forecast vs. Actual Holdout Data", height=600, legend_title="Series")
    st.plotly_chart(fig_forecast, use_container_width=True)

    # --- Error Metrics Visualization ---
    st.subheader("Error Metrics Comparison")
    metrics_data = [
        {'Metric': 'MAE', 'Model': 'Chronos', 'Value': response['chronos_forecast']['metrics']['mae']},
        {'Metric': 'MAE', 'Model': 'ARIMA', 'Value': response['arima_forecast']['metrics']['mae']},
        {'Metric': 'RMSE', 'Model': 'Chronos', 'Value': response['chronos_forecast']['metrics']['rmse']},
        {'Metric': 'RMSE', 'Model': 'ARIMA', 'Value': response['arima_forecast']['metrics']['rmse']}
    ]
    metrics_df = pd.DataFrame(metrics_data)

    fig_metrics = px.bar(
        metrics_df, x="Metric", y="Value", color="Model",
        barmode="group", text_auto='.2f', title="Error Metrics (Lower is Better)"
    )
    st.plotly_chart(fig_metrics, use_container_width=True)

else:
    st.info("Select a dataset and click 'Generate Forecast' to get started.")