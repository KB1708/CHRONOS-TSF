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
    This schema is the 'gatekeeper' for the data sent to the frontend.
    """
    historical_dates: List[str]
    historical_values: List[float]
    forecast_dates: List[str]
    
    # --- THIS IS THE FIX ---
    # We must explicitly add the 'actual_values' field here.
    actual_values: List[float]
    # --- END OF FIX ---
    
    chronos_forecast: ForecastResult
    arima_forecast: ForecastResult

