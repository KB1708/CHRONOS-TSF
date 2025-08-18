# GenAI Time Series Forecasting Portal

This project is a proof-of-concept web application that compares the performance of a **Generative AI model (Amazon Chronos)** against traditional statistical models (**ARIMA, ETS**) for time series forecasting.

---

## Features

- **Interactive UI**: A Streamlit-based user interface for easy interaction.  
- **Multiple Models**: Compares the performance of Chronos, ARIMA, and ETS.  
- **Flexible Data Input**: Users can upload their own CSV files or use pre-loaded sample datasets.  
- **Dynamic Configuration**: Allows users to select date/value columns and set the forecast horizon.  
- **Rich Visualization**: Provides clear graphs of the forecasts, performance metrics, and a summary table.  

---

## Architecture

The application is built with a separate frontend and backend to ensure scalability and stability, which is a best practice for production-level ML systems.  

- **Frontend**: A Streamlit application (`ui/app_ui.py`) that provides the user interface.  
- **Backend**: A FastAPI server (`app/main.py`) that handles the heavy computational work of running the forecasting models.  

---

## Getting Started

Follow these instructions to set up and run the project on your local machine.

### Prerequisites

- Python 3.11  
- Git  

### 1. Clone the Repository

Open your terminal or command prompt and clone the project:


git clone --branch shivadharshini  https://github.com/KB1708/CHRONOS-TSF.git

cd CHRONOS-TSF

### 2. Create and Activate a Virtual Environment

It is highly recommended to use a virtual environment to manage project dependencies.

Create the environment (using Python 3.11)

py -3.11 -m venv venv

Activate the environment
On Windows:

venv\Scripts\activate

On Mac/Linux:

source venv/bin/activate


### 3. Install Dependencies

Install all the required Python libraries from the `requirements.txt` file.

pip install -r requirements.txt


---

## How to Run the Application

This application requires **two separate processes** to be running at the same time in two different terminals.

### 1. Start the Backend Server

Open your **first terminal**, navigate to the project root, and run the following command to start the FastAPI backend server:

uvicorn app.main:app --reload


You should see a message indicating the server is running on:  
👉 http://127.0.0.1:8000  

Leave this terminal running.

### 2. Start the Frontend UI

Open a **second, new terminal**, navigate to the same project root, and activate the virtual environment again:

On Windows:

venv\Scripts\activate

On Mac/Linux:

source venv/bin/activate


Then, run the following command to start the Streamlit frontend:


streamlit run ui/app_ui.py


Your web browser will automatically open a new tab with the application running.

---

## How to Use the App

1. **Select a Dataset**: Use the sidebar to either upload your own CSV file or choose one of the pre-loaded sample datasets.  
2. **Configure Columns**: In the main panel, use the dropdowns to select the correct "Date" and "Value" columns from your data.  
3. **Set Forecast Horizon**: Use the slider in the sidebar to choose how many months into the future you want to forecast.  
4. **Generate Forecast**: Click the **"Generate Forecast"** button to run the analysis.  
5. **View Results**: The results, including metrics and graphs, will be displayed on the main page.  

---
