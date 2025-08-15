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


def generate_all_forecasts_and_metrics(df: pd.DataFrame, forecast_horizon: int = 20):
    """
    The main orchestration function using AutoGluon.
    It takes a dataframe, runs Chronos and ARIMA, and returns all results.
    """
    df["item_id"] = "stock_price"
    ts_df = TimeSeriesDataFrame.from_data_frame(
        df, id_column="item_id", timestamp_column="Date"
    )
    
    predictor_path = "./autogluon_models"
    hyperparameters = {"Chronos": {}, "ARIMA": {}}

    predictor = TimeSeriesPredictor(
        prediction_length=forecast_horizon,
        path=predictor_path,
        target="Close",
        eval_metric="MASE",
        freq="D"
    )

    predictor.fit(
        ts_df, 
        hyperparameters=hyperparameters,
        enable_ensemble=False
    )

    try:
        chronos_model_name = next(name for name in predictor.model_names() if name.startswith("Chronos"))
    except StopIteration:
        chronos_model_name = "Chronos"

    chronos_predictions = predictor.predict(ts_df, model=chronos_model_name)
    arima_predictions = predictor.predict(ts_df, model="ARIMA")
    
    # --- FINAL FIXES START HERE ---
    
    # 1. Correctly extract forecast dates and values
    forecast_dates_ts = chronos_predictions.index.get_level_values('timestamp')
    chronos_preds = chronos_predictions["mean"].values
    arima_preds = arima_predictions["mean"].values

    # 2. Get the true actual values for the forecast period to calculate metrics
    true_values = df.iloc[-forecast_horizon:]["Close"].values
    
    # 3. Calculate metrics manually for accuracy
    chronos_metrics = calculate_manual_metrics(true_values, chronos_preds)
    arima_metrics = calculate_manual_metrics(true_values, arima_preds)
    
    # 4. Correctly extract historical data
    train_df = df.iloc[:-forecast_horizon]

    response = {
        "historical_dates": pd.to_datetime(train_df['Date']).dt.strftime('%Y-%m-%d').tolist(),
        "historical_values": train_df['Close'].tolist(),
        "forecast_dates": pd.to_datetime(forecast_dates_ts).strftime('%Y-%m-%d').tolist(), # <-- FIX 1
        "chronos_forecast": {
            "forecast_values": chronos_preds.tolist(),
            "metrics": chronos_metrics  # <-- FIX 2
        },
        "arima_forecast": {
            "forecast_values": arima_preds.tolist(),
            "metrics": arima_metrics  # <-- FIX 2
        }
    }
    
    return response