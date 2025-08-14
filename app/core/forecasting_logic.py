import pandas as pd
import logging
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_all_forecasts_and_metrics(df: pd.DataFrame, forecast_horizon: int = 20):
    """
    The main orchestration function using AutoGluon.
    It takes a dataframe, runs Chronos and ARIMA, and returns all results.

    Args:
        df (pd.DataFrame): DataFrame with 'Date' and 'Close' columns.
        forecast_horizon (int): Number of periods to forecast.

    Returns:
        dict: A dictionary containing all the data needed for the API response.
    """
    # 1. Prepare the data for AutoGluon
    # The 'item_id' is needed by AutoGluon to identify different time series.
    # Since we have only one, we can use a constant value like 'stock_price'.
    df["item_id"] = "stock_price"
    
    # Convert to a TimeSeriesDataFrame, specifying the time and target columns
    ts_df = TimeSeriesDataFrame.from_data_frame(
        df,
        id_column="item_id",
        timestamp_column="Date",
    )
    
    # 2. Initialize and Train the Models
    logger.info("Initializing TimeSeriesPredictor...")
    # We specify a path to save the model artifacts. This is required by AutoGluon.
    predictor_path = "./autogluon_models"
    
    # We explicitly tell AutoGluon to only use Chronos and ARIMA.
    # This keeps the comparison focused and speeds up the process.
    hyperparameters = {
        "Chronos-T5-Small": {},
        "ARIMA": {}
    }

    predictor = TimeSeriesPredictor(
        prediction_length=forecast_horizon,
        path=predictor_path,
        target="Close",
        eval_metric="MASE", # AutoGluon requires an optimization metric, MASE is a good default
    )

    logger.info("Training Chronos and ARIMA models...")
    predictor.fit(
        ts_df,
        hyperparameters=hyperparameters,
    )

    # 3. Generate Forecasts
    logger.info("Generating forecasts...")
    predictions = predictor.predict(ts_df)

    # 4. Evaluate Models and Get Metrics
    logger.info("Evaluating models...")
    leaderboard = predictor.leaderboard(ts_df, silent=True)
    
    # Extract MAE and RMSE for both models from the leaderboard
    def get_metrics(model_name):
        try:
            model_stats = leaderboard[leaderboard['model'] == model_name].iloc[0]
            # AutoGluon gives negative MAE/RMSE, so we take the absolute value.
            return {
                "mae": abs(model_stats.get("MAE", 0.0)),
                "rmse": abs(model_stats.get("RMSE", 0.0))
            }
        except (IndexError, KeyError):
            # Fallback if a model fails or metrics aren't available
            return {"mae": 0.0, "rmse": 0.0}

    chronos_metrics = get_metrics("Chronos-T5-Small")
    arima_metrics = get_metrics("ARIMA")

    # 5. Format the Response
    # The historical data and forecast dates are managed slightly differently
    train_series = df.iloc[:-forecast_horizon]
    test_series = df.iloc[-forecast_horizon:]

    # Extract the forecasted values from the predictions DataFrame
    chronos_preds = predictions[predictions.index.get_level_values('item_id') == 'stock_price']['Chronos-T5-Small']
    arima_preds = predictions[predictions.index.get_level_values('item_id') == 'stock_price']['ARIMA']

    response = {
        "historical_dates": pd.to_datetime(train_series['Date']).dt.strftime('%Y-%m-%d').tolist(),
        "historical_values": train_series['Close'].tolist(),
        "forecast_dates": pd.to_datetime(test_series['Date']).dt.strftime('%Y-%m-%d').tolist(),
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