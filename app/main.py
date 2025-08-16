import pandas as pd
import io
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .core.forecasting_logic import generate_all_forecasts_and_metrics
from .core.schemas import ForecastResponse

# Initialize the FastAPI app
app = FastAPI(
    title="Time Series Forecasting API",
    description="An API to compare Chronos, ARIMA, and ETS forecasting models.",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", tags=["Root"])
def read_root():
    return {"message": "Welcome to the Forecasting Comparison API!"}

@app.post("/forecast/", response_model=ForecastResponse, tags=["Forecasting"])
async def create_forecast(
    file: UploadFile = File(...),
    date_col: str = Form(...),
    value_col: str = Form(...),
    forecast_horizon: int = Form(...)
):
    """
    This endpoint receives a CSV file, column names, and a forecast horizon,
    runs models, and returns the results.
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a CSV file.")
    
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        if date_col not in df.columns or value_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Provided column names not found. Ensure '{date_col}' and '{value_col}' are in the CSV.")

        results = generate_all_forecasts_and_metrics(
            df, 
            date_col=date_col, 
            target_col=value_col, 
            forecast_horizon=forecast_horizon
        )
        
        return results

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
