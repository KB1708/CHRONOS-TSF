import pandas as pd
import io
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .core.forecasting_logic import generate_all_forecasts_and_metrics
from .core.schemas import ForecastResponse

# Initialize the FastAPI app
app = FastAPI(
    title="Time Series Forecasting API",
    description="An API to compare Chronos and ARIMA forecasting models.",
    version="1.0.0"
)

# Configure CORS (Cross-Origin Resource Sharing)
# This is important to allow our Streamlit frontend to communicate with this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/", tags=["Root"])
def read_root():
    """A simple endpoint to check if the API is running."""
    return {"message": "Welcome to the Forecasting Comparison API!"}

@app.post("/forecast/", response_model=ForecastResponse, tags=["Forecasting"])
async def create_forecast(file: UploadFile = File(...)):
    """
    This endpoint receives a CSV file, runs forecasting models, and returns the results.
    
    - **file**: An uploaded CSV file with 'Date' and 'Close' columns.
    """
    # Ensure the uploaded file is a CSV
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a CSV file.")
    
    try:
        # Read the uploaded file into a pandas DataFrame
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        # Basic validation of the CSV structure
        if 'Date' not in df.columns or 'Close' not in df.columns:
            raise HTTPException(status_code=400, detail="CSV must contain 'Date' and 'Close' columns.")

        # Generate forecasts and metrics
        results = generate_all_forecasts_and_metrics(df)
        
        return results

    except Exception as e:
        # Catch any other potential errors during processing
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")