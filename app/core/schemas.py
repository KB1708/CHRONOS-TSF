from pydantic import BaseModel
from typing import List, Dict

class ModelMetrics(BaseModel):
    """Defines the structure for model performance metrics."""
    mae: float
    rmse: float

class ForecastResult(BaseModel):
    """Defines the structure for a single model's forecast."""
    forecast_values: List[float]
    metrics: ModelMetrics

class ForecastResponse(BaseModel):
    """
    Defines the final structure of the entire API response.
    """
    historical_dates: List[str]
    historical_values: List[float]
    forecast_dates: List[str]
    actual_values: List[float]
    
    chronos_forecast: ForecastResult
    arima_forecast: ForecastResult
    
    # --- FIX: Add the ets_forecast field ---
    ets_forecast: ForecastResult
    # --- END OF FIX ---

