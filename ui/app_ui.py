import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import numpy as np

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

def call_forecast_api(df, date_col, value_col, horizon):
    """Sends a dataframe, column names, and horizon to the backend API."""
    csv_buffer = df.to_csv(index=False).encode('utf-8')
    
    files = {'file': ('data.csv', csv_buffer, 'text/csv')}
    data = {'date_col': date_col, 'value_col': value_col, 'forecast_horizon': horizon}
    
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

# Forecast Horizon slider
forecast_horizon = st.sidebar.slider(
    "Forecast Horizon (Months)", 
    min_value=3, 
    max_value=36, 
    value=12, 
    step=3,
    help="How many months into the future do you want to forecast?"
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
            st.session_state['api_response'] = call_forecast_api(df, date_col, value_col, forecast_horizon)
            end_time = time.time()
            st.info(f"Processing took {end_time - start_time:.2f} seconds.")

if st.session_state['api_response']:
    response_data = st.session_state['api_response']
    
    # --- SECTION 1: Performance Metrics ---
    st.subheader("📊 Performance Metrics Comparison")
    st.markdown("Metrics are calculated on the test set. Lower values are better.")

    metrics = {
        'Chronos': response_data['chronos_forecast']['metrics'],
        'ARIMA': response_data['arima_forecast']['metrics'],
        'ETS': response_data['ets_forecast']['metrics']
    }
    
    for model in metrics:
        true = np.array(response_data['actual_values'])
        pred = np.array(response_data[f'{model.lower()}_forecast']['forecast_values'])
        metrics[model]['mape'] = np.mean(np.abs((true - pred) / np.where(true == 0, 1, true))) * 100

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Chronos (GenAI)")
        st.metric(label="MAE", value=f"{metrics['Chronos']['mae']:.2f}")
        st.metric(label="RMSE", value=f"{metrics['Chronos']['rmse']:.2f}")
        st.metric(label="MAPE", value=f"{metrics['Chronos']['mape']:.2f}%")

    with col2:
        st.subheader("ARIMA")
        st.metric(label="MAE", value=f"{metrics['ARIMA']['mae']:.2f}")
        st.metric(label="RMSE", value=f"{metrics['ARIMA']['rmse']:.2f}")
        st.metric(label="MAPE", value=f"{metrics['ARIMA']['mape']:.2f}%")
    
    with col3:
        st.subheader("ETS")
        st.metric(label="MAE", value=f"{metrics['ETS']['mae']:.2f}")
        st.metric(label="RMSE", value=f"{metrics['ETS']['rmse']:.2f}")
        st.metric(label="MAPE", value=f"{metrics['ETS']['mape']:.2f}%")
        
    # --- SECTION 2: Model Interpretation ---
    st.subheader("💡 Model Interpretation")
    st.info("""
    **How to choose the best model for your use case:**

    - **Lowest RMSE (Root Mean Square Error):** Choose this model if you want to minimize the risk of large, unexpected errors. It's the best choice for reliability and risk management.
    - **Lowest MAE (Mean Absolute Error):** Choose this model if the business cost of an error is directly proportional to its size.
    - **Lowest MAPE (Mean Absolute Percentage Error):** Choose this model if you need to communicate the forecast's accuracy in simple, relative terms. It's the most intuitive metric for business presentations.
    """)

    # --- SECTION 3: Forecast Visualization ---
    st.subheader("📈 Forecast Visualization")
    
    # --- Main Combined Plot ---
    fig_main = go.Figure()
    fig_main.add_trace(go.Scatter(x=response_data['historical_dates'], y=response_data['historical_values'], mode='lines', name='Historical Data', line=dict(color='royalblue', width=2)))
    fig_main.add_trace(go.Scatter(x=response_data['forecast_dates'], y=response_data['actual_values'], mode='lines', name='Actual Values (Holdout)', line=dict(color='grey', width=2, dash='dash')))
    fig_main.add_trace(go.Scatter(x=response_data['forecast_dates'], y=response_data['arima_forecast']['forecast_values'], mode='lines', name='ARIMA Forecast', line=dict(color='orange', width=2.5)))
    fig_main.add_trace(go.Scatter(x=response_data['forecast_dates'], y=response_data['chronos_forecast']['forecast_values'], mode='lines', name='Chronos Forecast', line=dict(color='firebrick', width=2.5)))
    fig_main.add_trace(go.Scatter(x=response_data['forecast_dates'], y=response_data['ets_forecast']['forecast_values'], mode='lines', name='ETS Forecast', line=dict(color='cyan', width=2.5)))
    
    split_date = response_data['historical_dates'][-1]
    fig_main.add_shape(type="line", x0=split_date, y0=0, x1=split_date, y1=1, yref="paper", line=dict(color="red", width=2, dash="dash"))
    fig_main.add_annotation(x=split_date, y=1, yref="paper", text="Train/Test Split", showarrow=False, yshift=10)

    fig_main.update_layout(title="All Models vs. Actual Data", xaxis_title="Date", yaxis_title="Value", height=500, legend_title="Series", template="plotly_dark")
    st.plotly_chart(fig_main, use_container_width=True)

    # --- NEW: Individual Comparison Subplots ---
    st.subheader("Individual Model Performance")
    
    fig_subplots = make_subplots(
        rows=1, cols=3, 
        shared_yaxes=True, 
        subplot_titles=("Chronos vs. Actual", "ARIMA vs. Actual", "ETS vs. Actual")
    )

    models_to_plot = [
        ('Chronos', response_data['chronos_forecast']['forecast_values'], 'firebrick'),
        ('ARIMA', response_data['arima_forecast']['forecast_values'], 'orange'),
        ('ETS', response_data['ets_forecast']['forecast_values'], 'cyan')
    ]

    for i, (name, forecast, color) in enumerate(models_to_plot):
        col = i + 1
        # Add recent history for context
        fig_subplots.add_trace(go.Scatter(
            x=response_data['historical_dates'][-30:], 
            y=response_data['historical_values'][-30:],
            mode='lines', name='History', line=dict(color='royalblue'), showlegend=(i==0)
        ), row=1, col=col)
        
        # Add actual values
        fig_subplots.add_trace(go.Scatter(
            x=response_data['forecast_dates'], 
            y=response_data['actual_values'],
            mode='lines', name='Actual', line=dict(color='grey', dash='dash'), showlegend=(i==0)
        ), row=1, col=col)
        
        # Add the specific model's forecast
        fig_subplots.add_trace(go.Scatter(
            x=response_data['forecast_dates'], 
            y=forecast,
            mode='lines', name=f'{name} Forecast', line=dict(color=color, width=2.5), showlegend=(i==0)
        ), row=1, col=col)

    fig_subplots.update_layout(height=400, template="plotly_dark")
    st.plotly_chart(fig_subplots, use_container_width=True)
    # --- END OF NEW FEATURE ---

else:
    st.info("Select a dataset and click 'Generate Forecast' to get started.")
