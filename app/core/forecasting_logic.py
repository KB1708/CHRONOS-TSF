import pandas as pd
import logging
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_manual_metrics(true_values, predicted_values):
    """Helper function to calculate metrics using sklearn."""
    mae = mean_absolute_error(true_values, predicted_values)
    rmse = np.sqrt(mean_squared_error(true_values, predicted_values))
    return {"mae": mae, "rmse": rmse}


def generate_all_forecasts_and_metrics(df: pd.DataFrame, date_col: str, target_col: str, forecast_horizon: int = 20):
    """
    The main orchestration function using AutoGluon.
    It takes a dataframe and column names, runs models, and returns results.
    """
    # Use the column names provided by the user
    logger.info(f"Using '{date_col}' as date column and '{target_col}' as target column.")
    df = df[[date_col, target_col]].copy()
    df.columns = ['Date', 'Close'] # Standardize names internally

    # Prepare the initial TimeSeriesDataFrame
    df["item_id"] = "stock_price"
    df["Date"] = pd.to_datetime(df["Date"])
    ts_df = TimeSeriesDataFrame.from_data_frame(
        df, id_column="item_id", timestamp_column="Date"
    )
    
    # Split data into training and test sets
    train_data = ts_df.iloc[:-forecast_horizon]
    test_data = ts_df.iloc[-forecast_horizon:]

    MIN_TRAIN_SIZE = 41
    if len(train_data) < MIN_TRAIN_SIZE:
        error_message = f"Training data is too short. The model requires at least {MIN_TRAIN_SIZE} data points, but only {len(train_data)} were provided. Please use a larger dataset or reduce the forecast horizon."
        logger.error(error_message)
        raise ValueError(error_message)
    
    predictor_path = "./autogluon_models"
    # --- FIX: Add ETS to the hyperparameters ---
    hyperparameters = {"Chronos": {}, "ARIMA": {}, "ETS": {}}
    # --- END OF FIX ---

    predictor = TimeSeriesPredictor(
        prediction_length=forecast_horizon,
        path=predictor_path,
        target="Close",
        eval_metric="MASE",
        freq="D" 
    )

    # Fit the predictor ONLY on the training data
    predictor.fit(
        train_data, 
        hyperparameters=hyperparameters,
        enable_ensemble=False
    )
    
    try:
        chronos_model_name = next(name for name in predictor.model_names() if name.startswith("Chronos"))
    except StopIteration:
        chronos_model_name = "Chronos"

    chronos_predictions = predictor.predict(train_data, model=chronos_model_name)
    arima_predictions = predictor.predict(train_data, model="ARIMA")
    ets_predictions = predictor.predict(train_data, model="ETS") # Predict with ETS
    
    # Extract forecast dates and values
    forecast_dates_ts = chronos_predictions.index.get_level_values('timestamp')
    chronos_preds = chronos_predictions["mean"].values
    arima_preds = arima_predictions["mean"].values
    ets_preds = ets_predictions["mean"].values # Get ETS values

    # Get the TRUE actual values from the holdout set
    true_values = test_data["Close"].values
    
    # Calculate metrics against the true values
    chronos_metrics = calculate_manual_metrics(true_values, chronos_preds)
    arima_metrics = calculate_manual_metrics(true_values, arima_preds)
    ets_metrics = calculate_manual_metrics(true_values, ets_preds) # Calculate ETS metrics
    
    # Prepare the final response object
    response = {
        "historical_dates": pd.to_datetime(train_data.index.get_level_values('timestamp')).strftime('%Y-%m-%d').tolist(),
        "historical_values": train_data['Close'].tolist(),
        "forecast_dates": pd.to_datetime(forecast_dates_ts).strftime('%Y-%m-%d').tolist(),
        "actual_values": true_values.tolist(),
        "chronos_forecast": {
            "forecast_values": chronos_preds.tolist(),
            "metrics": chronos_metrics
        },
        "arima_forecast": {
            "forecast_values": arima_preds.tolist(),
            "metrics": arima_metrics
        },
        # --- FIX: Add ETS results to the response ---
        "ets_forecast": {
            "forecast_values": ets_preds.tolist(),
            "metrics": ets_metrics
        }
        # --- END OF FIX ---
    }
    
    return response
