# Linear-Regression-on-Stock-price-prediction-system-APPL

## Project Overview
This project aims to predict stock prices using Linear Regression. It involves collecting historical stock data, preprocessing it, selecting relevant features, training a machine learning model, and evaluating its performance. The goal is to predict the next day's closing price based on historical data

## Objective
1. Apply Machine Learning Concepts
  - Understand how to apply Linear Regression, a fundamental machine learning algorithm, to a real-world problem.
  - Gain hands-on experience with the machine learning workflow, including data collection, preprocessing, feature selection, model training, and evaluation.

2. Predict Stock Prices
  - Build a model that can predict the next day's closing price of a stock based on historical data (e.g., Open, High, Low, Close, Volume).
  - Explore how well a simple model like Linear Regression performs on time-series financial data.

3. Develop Data Science Skills
  - Data Collection: Learn how to fetch and work with real-world data using APIs like yfinance.
  - Data Preprocessing: Clean and prepare data for modeling, including handling missing values and creating target variables.
  - Feature Selection: Identify the most relevant features for prediction using techniques like correlation analysis.
  - Model Evaluation: Use metrics like Mean Squared Error (MSE) and R-squared to evaluate model performance.

```Python
import yfinance as yf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
```

### Data Collection
```Python
# Fetch historical stock data for a specific ticker (e.g., Apple)
ticker = "AAPL"
data = yf.download(ticker, start="2024-04-01", end="2025-03-31")

# Display the first few rows of the data
print(data.head())
```
######
The use of yfinance library to fetch historical stock data for a specific ticker (e.g., Apple - AAPL). 
The data includes:
Open: The opening price of the stock.
High: The highest price of the stock during the day.
Low: The lowest price of the stock during the day.
Close: The closing price of the stock.
Volume: The number of shares traded during the day.

Explanation:
The yfinance library provides an easy way to access stock market data.
The data is stored in a Pandas DataFrame for further processing.

### Data Preprocessing 
```Python
# Create a new column for the target variable (e.g., predicting the next day's closing price)
data['Next_Close'] = data['Close'].shift(-1)

# Drop rows with NaN values (since the last row won't have a 'Next_Close')
data.dropna(inplace=True)

# Select features and target variable
features = ['Open', 'High', 'Low', 'Close', 'Volume']
X = data[features]
y = data['Next_Close']

# Display the first few rows of the features and target
print(X.head())
print(y.head())
```
######
Create a target variable (Next_Close), which represents the next day's closing price.
Drop rows with missing values (NaN) since the last row won't have a Next_Close value.
Select features (Open, High, Low, Close, Volume) and the target variable.

Explanation:
The target variable (Next_Close) is created by shifting the Close column by one day.
Features like Open, High, Low, Close, and Volume are used to predict the target.

