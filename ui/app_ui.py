# ui/app_ui.py

import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
import plotly.express as px
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
    csv_buffer = df.to_csv(index=False).encode('utf-8')
    files = {'file': ('data.csv', csv_buffer, 'text/csv')}
    
    try:
        response = requests.post(API_URL, files=files, timeout=600)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {e}")
        try:
            st.error(f"Response content: {response.content.decode()}")
        except:
            st.error("Could not decode error response from server.")
        return None

# --- Session State Initialization ---
if 'api_response' not in st.session_state:
    st.session_state.api_response = None
# --- FIX: Add original_df to session state to prevent EmptyDataError ---
if 'original_df' not in st.session_state:
    st.session_state.original_df = None

# --- UI Layout ---
st.sidebar.title("⚙️ Controls")
st.sidebar.markdown("Select a dataset and click 'Generate Forecast' to see the results.")

# Sample data selection
sample_data_options = {
    "NIFTY 50 Daily Prices": "data/nifty_50_daily.csv",
    "Reliance Industries Daily Prices": "data/reliance_daily.csv"
}
selected_sample = st.sidebar.selectbox("Choose a sample dataset:", options=list(sample_data_options.keys()))

uploaded_file = st.sidebar.file_uploader("Or upload your own CSV file", type="csv")

if st.sidebar.button("Generate Forecast", type="primary"):
    df = None
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_csv(sample_data_options[selected_sample])

    if "Date" not in df.columns or "Close" not in df.columns:
        st.sidebar.error("CSV must contain 'Date' and 'Close' columns.")
    else:
        # --- FIX: Save the loaded dataframe to session state ---
        st.session_state.original_df = df.copy()
        
        with st.spinner('Backend is processing... This may take a few minutes.'):
            start_time = time.time()
            st.session_state.api_response = call_forecast_api(df)
            end_time = time.time()
            st.sidebar.info(f"Processing took {end_time - start_time:.2f} seconds.")

st.title("⏱️ Time Series Forecasting Comparison")
st.markdown("Comparing **Amazon Chronos (Generative AI)** vs. **Traditional ARIMA**")

if st.session_state.api_response:
    response_data = st.session_state.api_response
    
    st.subheader("📊 Performance Metrics")
    st.markdown("Lower values are better for both **Mean Absolute Error (MAE)** and **Root Mean Squared Error (RMSE)**.")

    chronos_metrics = response_data['chronos_forecast']['metrics']
    arima_metrics = response_data['arima_forecast']['metrics']

    # --- NEW VISUALIZATION: Group by model ---
    metrics_data = [
        {'Model': 'Chronos', 'Metric': 'MAE', 'Value': chronos_metrics['mae']},
        {'Model': 'Chronos', 'Metric': 'RMSE', 'Value': chronos_metrics['rmse']},
        {'Model': 'ARIMA', 'Metric': 'MAE', 'Value': arima_metrics['mae']},
        {'Model': 'ARIMA', 'Metric': 'RMSE', 'Value': arima_metrics['rmse']}
    ]
    metrics_df = pd.DataFrame(metrics_data)
    
    fig_metrics = px.bar(
        metrics_df,
        x="Model",
        y="Value",
        color="Metric",
        barmode="group",
        text_auto='.2f',
        title="Performance Metrics Comparison"
    )
    st.plotly_chart(fig_metrics, use_container_width=True)
    # --- End of New Visualization ---

    st.subheader("📈 Forecast Visualization")
    fig_forecast = go.Figure()

    fig_forecast.add_trace(go.Scatter(
        x=response_data['historical_dates'],
        y=response_data['historical_values'],
        mode='lines', name='Historical Data', line=dict(color='royalblue')
    ))

    # --- FIX: Read the full dataframe from session state ---
    if st.session_state.original_df is not None:
        df_full = st.session_state.original_df
        actual_values = df_full['Close'].iloc[-len(response_data['forecast_dates']):].values
        fig_forecast.add_trace(go.Scatter(
            x=response_data['forecast_dates'], y=actual_values,
            mode='lines', name='Actual Values (Holdout)', line=dict(color='black', width=3)
        ))
    
    fig_forecast.add_trace(go.Scatter(
        x=response_data['forecast_dates'], y=response_data['arima_forecast']['forecast_values'],
        mode='lines', name='ARIMA Forecast', line=dict(color='orange', dash='dash')
    ))

    fig_forecast.add_trace(go.Scatter(
        x=response_data['forecast_dates'], y=response_data['chronos_forecast']['forecast_values'],
        mode='lines', name='Chronos Forecast', line=dict(color='lightgreen', width=3, dash='dash')
    ))

    fig_forecast.update_layout(
        title="Model Forecasts vs. Historical Data",
        xaxis_title="Date", yaxis_title="Value", legend_title="Series", height=600
    )
    st.plotly_chart(fig_forecast, use_container_width=True)

else:
    st.info("Select a dataset and click 'Generate Forecast' to get started.")