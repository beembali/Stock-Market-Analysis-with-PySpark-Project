# Stock Market Analysis with PySpark

## Overview
This project implements a complete data pipeline for stock market analysis using PySpark, focusing on price prediction using Linear Regression with technical indicators. The pipeline fetches data from Alpha Vantage API, processes it, performs analysis, and stores results in MySQL.

## Features
- Real-time stock data fetching from Alpha Vantage API
- Technical indicators calculation:
  - Bollinger Bands
  - RSI (Relative Strength Index)
  - Moving Averages (20-day and 100-day)
  - Momentum (Rate of Change)
- Linear Regression model for price prediction
- Interactive visualizations using Matplotlib
- Data persistence in MySQL database

## Technical Stack
- PySpark: Data processing and ML
- Alpha Vantage API: Stock data source
- MySQL: Data storage
- Matplotlib: Visualization
- Python requests: API interaction

## Setup
1. Environment Variables Required:
```python
   ALPHA_API_KEY=your_api_key
   sql_username=your_mysql_username
   sql_password=your_mysql_password
   sql_port=your_mysql_port
   ```

2. Dependencies:
```python
  pyspark
  requests
  matplotlib
  mysql-connector-python
  ```

3. MySQL Setup:
- Create a database named 'stock'
- Ensure MySQL server is running


## Pipeline Steps

1. Data Ingestion: Fetches daily stock data from Alpha Vantage
2. Transformation: Calculates technical indicators
3. Analysis: Implements Linear Regression for price prediction 
4. Visualization: Creates analysis plots
5. Storage: Stores processed data in MySQL

## Usage
``` python 
python main.py
```

## Output
- RMSE and R² metrics for model evaluation
- Feature importance analysis
- Four visualization plots:
    - Actual vs Predicted Prices
    - Prediction Error Over Time
    - Actual vs Predicted Scatter Plot
    - Price with Moving Averages and Bollinger Bands


## Data Sources
- Alpha Vantage API (Time Series Daily)