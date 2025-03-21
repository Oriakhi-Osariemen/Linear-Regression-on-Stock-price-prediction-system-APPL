# Linear-Regression-on-Stock-price-prediction-system-APPL

![Appl Logo](https://github.com/Oriakhi-Osariemen/Linear-Regression-on-Stock-price-prediction-system-APPL/blob/main/download%20(2).jpeg)

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

### Feature Selection 
```Python
# Calculate correlation matrix
corr_matrix = data.corr()

# Visualize the correlation matrix
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
```
![Result](https://github.com/Oriakhi-Osariemen/Linear-Regression-on-Stock-price-prediction-system-APPL/blob/main/Screenshot%202025-03-21%20012257.png)

######
Analyze the correlation between features and the target variable to identify the most relevant features.

Explanation:
The correlation matrix shows how each feature is related to the target (Next_Close).

Features with high correlation (close to 1 or -1) are more important for prediction.

### Train-Test Split
```Python
# Split the data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training set size:", X_train.shape)
print("Testing set size:", X_test.shape)
```

#####
Training set size: (194, 5)
Testing set size: (49, 5)

The dataset is split into training and testing sets to evaluate the model's performance.

Explanation:
The training set (80%) is used to train the model.

The testing set (20%) is used to evaluate the model's performance on unseen data.

### Model Training 
```Python
# Initialize the model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)
```

#####
Mean Squared Error: 6.981434921686678
R-squared: 0.9885746789261214

Evaluate the model using Mean Squared Error (MSE) and R-squared.

Explanation:
Mean Squared Error (MSE): Measures the average squared difference between predicted and actual values. Lower values indicate better performance.

R-squared: Indicates how well the model explains the variance in the target variable. Values closer to 1 indicate a better fit.

#### Result Evaluation 
```Python
# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='Actual Prices')
plt.plot(y_pred, label='Predicted Prices')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.title(f'{ticker} Stock Price Prediction')
plt.legend()
plt.show()
```
![Result1](https://github.com/Oriakhi-Osariemen/Linear-Regression-on-Stock-price-prediction-system-APPL/blob/main/Screenshot%202025-03-21%20014854.png)

#####
Visualization of the actual vs predicted stock prices to assess the model's performance.

Explanation:
The plot shows how closely the predicted prices match the actual prices.

This helps in visually assessing the model's accuracy.
