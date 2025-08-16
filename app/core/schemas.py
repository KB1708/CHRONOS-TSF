from pydantic import BaseModel
from typing import List, Dict, Any

class ForecastMetrics(BaseModel):
    """
    Pydantic model for the performance metrics of a single model.
    """
    mae: float
    rmse: float

class ModelForecast(BaseModel):
    """
    Pydantic model for the complete results of a single model.
    """
    forecast_values: List[float]
    metrics: ForecastMetrics

class ForecastResponse(BaseModel):
    """
    The main response model for the /forecast API endpoint.
    It contains the full results for both models.
    """
    historical_dates: List[Any]
    historical_values: List[float]
    forecast_dates: List[Any]
    actual_values: List[float]
    chronos_forecast: ModelForecast
    arima_forecast: ModelForecast