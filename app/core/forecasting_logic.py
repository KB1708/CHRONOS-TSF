import pandas as pd
import numpy as np
import torch
from transformers import ChronosT5ForConditionalGeneration, AutoTokenizer
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import logging

# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- MODEL LOADING ---
# This section handles loading the pre-trained Chronos model.
# It's placed here so the model is loaded only once when the app starts.

def get_chronos_model():
    """
    Loads and returns the Chronos-T5 small model and tokenizer.
    Caches the model in memory for efficiency.
    """
    global chronos_model, chronos_tokenizer
    if 'chronos_model' not in globals():
        logger.info("Loading Chronos-T5-Small model for the first time...")
        # Use a try-except block for robustness against download issues
        try:
            # Check for MPS availability for Apple Silicon, otherwise CUDA or CPU
            device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
            
            chronos_model = ChronosT5ForConditionalGeneration.from_pretrained(
                "amazon/chronos-t5-small",
                device_map=device,
                torch_dtype=torch.bfloat16,
            )
            chronos_tokenizer = AutoTokenizer.from_pretrained(
                "amazon/chronos-t5-small"
            )
            logger.info(f"Chronos model loaded successfully on device: {device}")
        except Exception as e:
            logger.error(f"Failed to load Chronos model: {e}")
            # Depending on the use case, you might want to exit or handle this differently
            raise
    return chronos_model, chronos_tokenizer

# Initialize the model on startup
get_chronos_model()


# --- FORECASTING FUNCTIONS ---

def run_chronos_forecast(context: np.ndarray, forecast_horizon: int) -> np.ndarray:
    """
    Generates a forecast using the pre-trained Chronos model.
    
    Args:
        context (np.ndarray): The historical time series data.
        forecast_horizon (int): The number of future steps to predict.

    Returns:
        np.ndarray: The forecasted values.
    """
    model, tokenizer = get_chronos_model()
    device = model.device

    inputs = tokenizer(
        [context.tolist()],
        padding=True,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    # Generate the forecast
    outputs = model.generate(
        **inputs,
        num_beams=10,
        num_return_sequences=1,
        prediction_length=forecast_horizon,
    )

    # Decode and return the forecast
    forecast = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    # The output is a string of space-separated numbers, convert it to a numpy array
    forecast_values = np.fromstring(forecast[0], sep=' ', dtype=float)
    
    return forecast_values


def run_arima_forecast(train_data: pd.Series, forecast_horizon: int) -> np.ndarray:
    """
    Trains an ARIMA model and generates a forecast.
    Note: For a POC, we use a simple (p,d,q)=(5,1,0) order. A real-world
    application would use auto_arima or other methods to find the best order.
    
    Args:
        train_data (pd.Series): The historical time series data for training.
        forecast_horizon (int): The number of future steps to predict.

    Returns:
        np.ndarray: The forecasted values.
    """
    # A common baseline order for financial data
    order = (5, 1, 0) 
    
    model = ARIMA(train_data, order=order)
    fitted_model = model.fit()
    
    # Generate forecast
    forecast = fitted_model.forecast(steps=forecast_horizon)
    
    return forecast.values


# --- METRICS AND ORCHESTRATION ---

def calculate_metrics(true_values: np.ndarray, predicted_values: np.ndarray) -> dict:
    """
    Calculates MAE and RMSE for a forecast.
    
    Args:
        true_values (np.ndarray): The actual observed values.
        predicted_values (np.ndarray): The values predicted by the model.

    Returns:
        dict: A dictionary containing the MAE and RMSE.
    """
    mae = mean_absolute_error(true_values, predicted_values)
    rmse = np.sqrt(mean_squared_error(true_values, predicted_values))
    return {"mae": mae, "rmse": rmse}


def generate_all_forecasts_and_metrics(df: pd.DataFrame, forecast_horizon: int = 20):
    """
    The main orchestration function. It takes a dataframe, splits it,
    runs both models, calculates metrics, and returns all results.
    
    Args:
        df (pd.DataFrame): DataFrame with 'Date' and 'Close' columns.
        forecast_horizon (int): Number of periods to forecast. Defaults to 20.

    Returns:
        dict: A dictionary containing all the data needed for the API response.
    """
    # Ensure Date column is datetime
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    
    # Use the 'Close' column for forecasting
    time_series = df['Close']
    
    # Split data: train on all data except the last `forecast_horizon` points
    train_series = time_series.iloc[:-forecast_horizon]
    test_series = time_series.iloc[-forecast_horizon:]
    
    # --- Run Models ---
    logger.info("Running Chronos forecast...")
    chronos_preds = run_chronos_forecast(train_series.values, forecast_horizon)
    
    logger.info("Running ARIMA forecast...")
    arima_preds = run_arima_forecast(train_series, forecast_horizon)
    
    # --- Calculate Metrics ---
    logger.info("Calculating metrics...")
    chronos_metrics = calculate_metrics(test_series.values, chronos_preds)
    arima_metrics = calculate_metrics(test_series.values, arima_preds)
    
    # --- Prepare Response ---
    # Convert numpy arrays and pandas series to lists for JSON serialization
    response = {
        "historical_dates": train_series.index.strftime('%Y-%m-%d').tolist(),
        "historical_values": train_series.values.tolist(),
        "forecast_dates": test_series.index.strftime('%Y-%m-%d').tolist(),
        "chronos_forecast": {
            "forecast_values": chronos_preds.tolist(),
            "metrics": chronos_metrics
        },
        "arima_forecast": {
            "forecast_values": arima_preds.tolist(),
            "metrics": arima_metrics
        }
    }
    
    return response