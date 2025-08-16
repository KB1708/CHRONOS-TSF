import pandas as pd
import logging
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import shutil

# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_manual_metrics(true_values, predicted_values):
    """Helper function to calculate metrics using sklearn."""
    mae = mean_absolute_error(true_values, predicted_values)
    rmse = np.sqrt(mean_squared_error(true_values, predicted_values))
    return {"mae": mae, "rmse": rmse}


def generate_all_forecasts_and_metrics(df: pd.DataFrame, forecast_horizon: int = None):
    df["item_id"] = "series_1"
    
    # Split data: 70% for training, 30% for testing
    split_index = int(len(df) * 0.7)
    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]
    
    prediction_length = len(test_df)

    train_ts_df = TimeSeriesDataFrame.from_data_frame(
        train_df, id_column="item_id", timestamp_column="Date"
    )
    
    predictor_path = "./autogluon_models"
    
    # --- THIS IS THE FIX ---
    # 2. Delete the old model directory before starting a new run.
    # This ensures every forecast is clean and prevents caching errors.
    try:
        shutil.rmtree(predictor_path)
    except FileNotFoundError:
        pass  # Ignore if the folder doesn't exist on the first run
    # --- END OF FIX ---
        
    hyperparameters = {"Chronos": {}, "ARIMA": {}}

    predictor = TimeSeriesPredictor(
        prediction_length=prediction_length,
        path=predictor_path,
        target="Close",
        eval_metric="MASE",
        freq="D"
    )

    predictor.fit(
        train_ts_df, 
        hyperparameters=hyperparameters,
        enable_ensemble=False
    )

    try:
        chronos_model_name = next(name for name in predictor.model_names() if name.startswith("Chronos"))
    except StopIteration:
        chronos_model_name = "Chronos"

    # Make separate, explicit predict calls for each model
    chronos_predictions = predictor.predict(train_ts_df, model=chronos_model_name)
    arima_predictions = predictor.predict(train_ts_df, model="ARIMA")

    chronos_preds = chronos_predictions["mean"].values
    arima_preds = arima_predictions["mean"].values
    
    true_values = test_df["Close"].values
    chronos_metrics = calculate_manual_metrics(true_values, chronos_preds)
    arima_metrics = calculate_manual_metrics(true_values, arima_preds)
    
    response = {
        "historical_dates": pd.to_datetime(train_df['Date']).dt.strftime('%Y-%m-%d').tolist(),
        "historical_values": train_df['Close'].tolist(),
        "forecast_dates": pd.to_datetime(test_df['Date']).dt.strftime('%Y-%m-%d').tolist(),
        "actual_values": test_df['Close'].tolist(),
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