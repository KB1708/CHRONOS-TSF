# app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime
import warnings
import shutil
import uuid
import logging

from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings('ignore')

# --- Page Configuration ---
st.set_page_config(
    page_title="Time Series Forecasting Comparison Portal",
    page_icon="📈",
    layout="wide",
)

#======================================================================
# --- BACKEND LOGIC (INTEGRATED INTO A SINGLE SCRIPT) ---
#======================================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_metrics(actual, predicted):
    """Calculate all forecasting metrics"""
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    # Handle division by zero for MAPE
    mape = np.mean(np.abs((actual - predicted) / np.where(actual == 0, 1, actual))) * 100
    return {'mae': mae, 'rmse': rmse, 'mape': mape}

@st.cache_data(show_spinner=False)
def run_all_models(_df, forecast_horizon):
    """
    Main logic function: Takes a prepared dataframe, splits data, 
    trains all models using AutoGluon, and returns a results dictionary.
    The underscore in _df signals to Streamlit's caching that this input is mutable.
    """
    predictor_path = f"./autogluon_models_temp_{uuid.uuid4()}"
    logger.info(f"Creating temporary model path: {predictor_path}")

    try:
        _df["item_id"] = "series_1"
        
        # Split data: Use all but the forecast_horizon for training
        train_df = _df.iloc[:-forecast_horizon]
        test_df = _df.iloc[-forecast_horizon:]

        train_ts_df = TimeSeriesDataFrame.from_data_frame(
            train_df, id_column="item_id", timestamp_column="Date"
        )
        
        # Define the models we want AutoGluon to train
        hyperparameters = {"Chronos": {}, "ARIMA": {}, "ETS": {}}

        predictor = TimeSeriesPredictor(
            prediction_length=forecast_horizon,
            path=predictor_path,
            target="Value",
            eval_metric="MASE",
            freq="D" # Assuming daily frequency, common for financial data
        )

        logger.info("Training Chronos, ARIMA, and ETS models...")
        predictor.fit(
            train_ts_df, 
            hyperparameters=hyperparameters,
            enable_ensemble=False
        )

        # Programmatically find the full model names
        try:
            chronos_model_name = next(name for name in predictor.model_names() if name.startswith("Chronos"))
        except StopIteration: chronos_model_name = "Chronos"
        
        # Make separate prediction calls for each model
        logger.info("Generating forecasts...")
        chronos_preds = predictor.predict(train_ts_df, model=chronos_model_name)["mean"].values
        arima_preds = predictor.predict(train_ts_df, model="ARIMA")["mean"].values
        ets_preds = predictor.predict(train_ts_df, model="ETS")["mean"].values
        
        # Calculate metrics against the holdout set
        true_values = test_df["Value"].values
        metrics = {
            'Chronos': calculate_metrics(true_values, chronos_preds),
            'ARIMA': calculate_metrics(true_values, arima_preds),
            'ETS': calculate_metrics(true_values, ets_preds)
        }
        
        # Prepare the response dictionary for the UI
        response = {
            "historical_dates": pd.to_datetime(train_df['Date']).dt.strftime('%Y-%m-%d').tolist(),
            "historical_values": train_df['Value'].tolist(),
            "forecast_dates": pd.to_datetime(test_df['Date']).dt.strftime('%Y-%m-%d').tolist(),
            "actual_values": test_df['Value'].tolist(),
            "chronos_forecast": {"forecast_values": chronos_preds.tolist(), "metrics": metrics['Chronos']},
            "arima_forecast": {"forecast_values": arima_preds.tolist(), "metrics": metrics['ARIMA']},
            "ets_forecast": {"forecast_values": ets_preds.tolist(), "metrics": metrics['ETS']}
        }
        return response

    finally:
        # Clean up the temporary model directory
        logger.info(f"Cleaning up temporary model path: {predictor_path}")
        try:
            shutil.rmtree(predictor_path)
        except Exception as e:
            logger.error(f"Error during cleanup of {predictor_path}: {e}")

#======================================================================
# --- STREAMLIT UI (Based on your friend's code) ---
#======================================================================

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = None
if 'df_loaded' not in st.session_state:
    st.session_state.df_loaded = None

# Sidebar
with st.sidebar:
    st.title("⚙️ Configure Your Analysis")
    data_source = st.selectbox("Choose Data Source:", ["Yahoo Finance", "Upload CSV"])
    df_to_process = None

    if data_source == "Yahoo Finance":
        ticker = st.text_input("Enter Stock Ticker:", value="AAPL")
        period = st.selectbox("Select Period:", ["1y", "2y", "5y"], index=1)
        if st.button("Load Data"):
            df_to_process = yf.download(ticker, period=period, progress=False).reset_index()
            st.session_state.df_loaded = df_to_process

    else: # Upload CSV
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file:
            df_to_process = pd.read_csv(uploaded_file)
            st.session_state.df_loaded = df_to_process

    if st.session_state.df_loaded is not None:
        st.header("Forecasting Parameters")
        forecast_horizon = st.slider("Forecast Horizon (Days)", 10, 90, 30)
        
        if st.button("🚀 Run Forecasting", type="primary"):
            st.session_state.results = None # Clear old results
            df = st.session_state.df_loaded
            # This will be passed to the main page for column selection
            st.session_state.run_forecast = True
            st.session_state.forecast_horizon = forecast_horizon

# Main Page
st.title("⏱️ Time Series Forecasting Comparison")
st.markdown("Comparing **Amazon Chronos (GenAI)** vs. **Traditional Models (ARIMA, ETS)**")

if st.session_state.df_loaded is not None:
    df = st.session_state.df_loaded
    st.subheader("Data Preview & Configuration")
    st.dataframe(df.head())

    col1, col2 = st.columns(2)
    with col1:
        date_col = st.selectbox("Select your Date Column:", df.columns)
    with col2:
        value_col = st.selectbox("Select your Value Column:", df.columns, index=min(1, len(df.columns)-1))

    # Prepare data with standardized column names
    df_prepared = df[[date_col, value_col]].copy()
    df_prepared.columns = ['Date', 'Value']
    df_prepared['Date'] = pd.to_datetime(df_prepared['Date'])
    df_prepared.dropna(inplace=True)

    if st.session_state.get('run_forecast', False):
        with st.spinner('Training models and generating forecasts... This may take several minutes.'):
            # Directly call the integrated backend logic
            st.session_state.results = run_all_models(df_prepared, st.session_state.forecast_horizon)
        st.session_state.run_forecast = False # Reset the run trigger

if st.session_state.results:
    res = st.session_state.results

    st.subheader("📈 Forecast Visualization")
    fig_main = go.Figure()
    fig_main.add_trace(go.Scatter(x=res['historical_dates'], y=res['historical_values'], name='Historical (Train) Data', line=dict(color='royalblue')))
    fig_main.add_trace(go.Scatter(x=res['forecast_dates'], y=res['actual_values'], name='Actual Values (Test)', line=dict(color='black', width=3)))
    fig_main.add_trace(go.Scatter(x=res['forecast_dates'], y=res['arima_forecast']['forecast_values'], name='ARIMA Forecast', line=dict(color='orange', dash='dash')))
    fig_main.add_trace(go.Scatter(x=res['forecast_dates'], y=res['chronos_forecast']['forecast_values'], name='Chronos Forecast', line=dict(color='lightgreen', dash='dash')))
    fig_main.add_trace(go.Scatter(x=res['forecast_dates'], y=res['ets_forecast']['forecast_values'], name='ETS Forecast', line=dict(color='red', dash='dash')))
    
    split_date = res['historical_dates'][-1]
    fig_main.add_vline(x=split_date, line_width=2, line_dash="dash", line_color="red", annotation_text="Train/Test Split")
    fig_main.update_layout(title="All Models vs. Actual Data", height=500, template="plotly_dark")
    st.plotly_chart(fig_main, use_container_width=True)

    st.subheader("📊 Visual Metrics Comparison")
    metrics_data = {model.replace('_forecast','').capitalize(): data['metrics'] for model, data in res.items() if model.endswith('_forecast')}
    metrics_df = pd.DataFrame(metrics_data).T.reset_index().rename(columns={'index': 'Model'})
    metrics_df_melted = metrics_df.melt(id_vars='Model', var_name='Metric', value_name='Value')

    fig_metrics = px.bar(metrics_df_melted, x="Metric", y="Value", color="Model", barmode="group", text_auto='.2f', title="Error Metrics (Lower is Better)")
    st.plotly_chart(fig_metrics, use_container_width=True)
    
else:
    st.info("Load a dataset and click 'Run Forecasting' in the sidebar to begin.")