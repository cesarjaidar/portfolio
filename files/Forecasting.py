#!/usr/bin/env python
# coding: utf-8
Author: Cesar Jaidar
# # DSC680 Week 1-4

# In[32]:


# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Load the dataset
file_path = 'Top10VideoGameStocks.csv'
data = pd.read_csv(file_path)

# Display basic information about the dataset
data_info = data.info()
data_head = data.head()

# Convert the 'Date' column to datetime format and set it as the index
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Check the number of unique companies
unique_companies = data['Company'].nunique()


# In[33]:


# Visualize stock price trends (Close price) for all companies over time
plt.figure(figsize=(15, 8))
sns.lineplot(data=data, x=data.index, y='Close', hue='Company')
plt.title('Stock Price Trends (Close Price) Over Time')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


# In[34]:


# Visualize trading volume trends for all companies over time
plt.figure(figsize=(15, 8))
sns.lineplot(data=data, x=data.index, y='Volume', hue='Company')
plt.title('Trading Volume Trends Over Time')
plt.xlabel('Date')
plt.ylabel('Volume')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


# In[35]:


# Check for duplicate dates per company
duplicates = data.reset_index().duplicated(subset=['Date', 'Company'], keep=False)

# Display duplicates if they exist
duplicate_rows = data.reset_index()[duplicates]

# Reset the index temporarily to ensure proper plotting
data_reset = data.reset_index()

# Plot stock price trends (Close price) for all companies over time
plt.figure(figsize=(15, 8))
sns.lineplot(data=data_reset, x='Date', y='Close', hue='Company')
plt.title('Stock Price Trends (Close Price) Over Time')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()



# In[36]:


# Plot trading volume trends for all companies over time
plt.figure(figsize=(15, 8))
sns.lineplot(data=data_reset, x='Date', y='Volume', hue='Company')
plt.title('Trading Volume Trends Over Time')
plt.xlabel('Date')
plt.ylabel('Volume')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


# ## Time-Series Analysis

# In[37]:


from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error
import numpy as np

# Select a specific company's data for ARIMA modeling (e.g., Sony Interactive Entertainment)
company_name = "Sony Interactive Entertainment"
sony_data = data[data['Company'] == company_name]

# Use the 'Close' price for ARIMA modeling
sony_close_prices = sony_data['Close']

# Plot the data to visualize trends and seasonality
plt.figure(figsize=(12, 6))
plt.plot(sony_close_prices, label=f"{company_name} Close Prices")
plt.title(f"{company_name} Close Prices Over Time")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()
plt.show()




# In[38]:


# Plot ACF and PACF to determine ARIMA parameters
plot_acf(sony_close_prices, lags=40)
plt.title("Autocorrelation Function (ACF)")
plt.show()


# In[39]:


plot_pacf(sony_close_prices, lags=40)
plt.title("Partial Autocorrelation Function (PACF)")
plt.show()



# In[40]:


# Define ARIMA model parameters (p, d, q) based on ACF and PACF insights
p, d, q = 2, 1, 2

# Fit the ARIMA model
arima_model = ARIMA(sony_close_prices, order=(p, d, q))
arima_result = arima_model.fit()

# Summary of the ARIMA model
arima_summary = arima_result.summary()

# Plot the fitted values against the original data
plt.figure(figsize=(12, 6))
plt.plot(sony_close_prices, label="Original")
plt.plot(arima_result.fittedvalues, label="Fitted", color="red")
plt.title(f"{company_name} ARIMA Model Fitted Values")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()
plt.show()

# Calculate and display the model's mean squared error (MSE)
mse = mean_squared_error(sony_close_prices[d:], arima_result.fittedvalues[d:])
mse, arima_summary


# ## ARIMA Analysis
Order: ARIMA(2, 1, 2) was chosen based on ACF and PACF plots.

Model Fit:
AIC (Akaike Information Criterion): 893.639
BIC (Bayesian Information Criterion): 912.107
Coefficients:

AR (autoregressive) and MA (moving average) terms are statistically significant (p < 0.05).

Residual Diagnostics:
Ljung-Box test suggests no significant autocorrelation in residuals (p > 0.05 for L1).
The residuals' Jarque-Bera test indicates non-normality (likely due to heavy tails).

Performance:

Mean Squared Error (MSE): 1.2088, which represents the average squared difference between observed and fitted values.
Visualization:

The plot of fitted values shows a good alignment with the original data, indicating the model captures trends effectively.

# ## Forecasting

# In[41]:


# Forecast future stock prices using the ARIMA model
forecast_steps = 12  # Forecast for the next 12 months (1 year)
forecast = arima_result.get_forecast(steps=forecast_steps)
forecast_mean = forecast.predicted_mean
forecast_ci = forecast.conf_int()

# Prepare forecast table for display
forecast_table = pd.DataFrame({
    "Forecasted Close Price": forecast_mean,
    "Lower Bound": forecast_ci.iloc[:, 0],  # Use the first column as the lower bound
    "Upper Bound": forecast_ci.iloc[:, 1],  # Use the second column as the upper bound
})
forecast_table.index.name = "Date"

# Convert indexes to numerical values for plotting compatibility
forecast_mean_np = forecast_mean.to_numpy()
forecast_ci_lower = forecast_ci.iloc[:, 0].to_numpy()
forecast_ci_upper = forecast_ci.iloc[:, 1].to_numpy()

# Convert the index to a range for plotting
forecast_numeric_index = np.arange(len(forecast_mean))
historical_numeric_index = np.arange(len(sony_close_prices))

# Plot the historical and forecasted data with confidence intervals
plt.figure(figsize=(12, 6))
plt.plot(historical_numeric_index, sony_close_prices, label="Historical Data", color="blue")
plt.plot(forecast_numeric_index + len(historical_numeric_index), forecast_mean_np, label="Forecasted Data", color="red")
plt.fill_between(forecast_numeric_index + len(historical_numeric_index),
                 forecast_ci_lower, forecast_ci_upper,
                 color="pink", alpha=0.3, label="Confidence Interval")
plt.title(f"{company_name} Stock Price Forecast (Next 12 Months)")
plt.xlabel("Time (Index)")
plt.ylabel("Close Price")
plt.legend()
plt.show()

# Print the forecasted values
print("Sony Stock Price Forecast (Next 12 Months):")
print(forecast_table)


# ## Forecast Analysis
The forecast graph illustrates the historical and projected stock prices for Sony Interactive Entertainment over the next 12 months. Below is a detailed analysis:

Historical Performance:

The historical stock prices exhibit significant fluctuations over the observed time period, reflecting major market events and company-specific developments.
Notable peaks and troughs highlight periods of rapid growth and subsequent corrections.

Forecasted Values:

The forecasted stock prices (indicated by the red line) show a relatively stable trend, with minor fluctuations over the next 12 months.
This suggests a period of reduced volatility in Sony's stock price compared to historical trends.

Confidence Intervals:

The shaded pink area represents the 95% confidence interval for the forecast.
The widening of the confidence interval indicates increased uncertainty as we move further into the future.
Despite this uncertainty, the forecast suggests that Sony's stock price is unlikely to deviate drastically from the predicted values.

Key Implications:

Investors can expect Sony's stock price to maintain relative stability in the near term.
The forecast provides a reliable basis for short-term investment decisions, as the predicted values fall within a narrow range of the historical price trend. The widening confidence interval highlights the need to account for potential market disruptions or unexpected events that could impact stock performance.

In conclusion, the forecast suggests a stable outlook for Sony Interactive Entertainment's stock price over the next year, with limited but noticeable uncertainty as the forecast horizon extends. Investors should monitor ongoing market conditions to ensure alignment with these predictions.
# ## Machine Learning - LSTM

# In[42]:


from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Normalize the data (scale values between 0 and 1)
scaler = MinMaxScaler(feature_range=(0, 1))
sony_close_prices_scaled = scaler.fit_transform(sony_close_prices.values.reshape(-1, 1))

# Create a function to prepare the data for LSTM (input-output pairs)
def prepare_lstm_data(data, time_step=60):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:i + time_step, 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

# Adjust the time_step to ensure sufficient data for testing
time_step = 30

# Re-prepare training and testing data for LSTM
def prepare_lstm_data(data, time_step):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:i + time_step, 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

# Generate input-output pairs
X_train, y_train = prepare_lstm_data(train_data, time_step)
X_test, y_test = prepare_lstm_data(test_data, time_step)

# Reshape input data to be 3D [samples, time_steps, features] for LSTM
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Check the updated shapes of the data
X_train.shape, y_train.shape, X_test.shape, y_test.shape


# In[43]:


# Build the LSTM model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1))  # Output layer for stock price prediction

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# Model training is complete. Let me proceed to make predictions and evaluate the model.


# In[44]:


# Make predictions on the test data
predicted_prices = model.predict(X_test)

# Scale the predicted values back to the original price range
predicted_prices_rescaled = scaler.inverse_transform(predicted_prices)
y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

# Plot the actual vs predicted stock prices
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(y_test_rescaled, label="Actual Prices", color="blue")
plt.plot(predicted_prices_rescaled, label="Predicted Prices", color="red")
plt.title("Actual vs Predicted Stock Prices")
plt.xlabel("Time")
plt.ylabel("Stock Price")
plt.legend()
plt.show()

# Evaluate the model performance
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test_rescaled, predicted_prices_rescaled)
print(f"Mean Squared Error: {mse}")


# ## LSTM Analysis
The graph displays the actual vs. predicted stock prices for Sony Interactive Entertainment, generated using an LSTM neural network. Below is the interpretation:

Model Performance:

The model captures the overall trend of the stock prices but struggles to match the short-term fluctuations accurately.
The predicted prices (red line) are smoother and less volatile compared to the actual prices (blue line), indicating the model's inability to react to rapid changes in the stock price.
Underfitting:

The model appears to underfit the data, as the predicted values deviate from the actual prices, particularly in areas with high volatility.
This could be due to a limited number of epochs, insufficient complexity in the model, or suboptimal hyperparameter tuning.

Improvement Areas:

Increase Model Complexity: Add more LSTM layers or units to better capture sequential dependencies.
Longer Training: Train the model for more epochs with a lower learning rate to improve accuracy.
Feature Engineering: Include additional features (e.g., trading volume, market indices) to provide more context for predictions.
Hyperparameter Tuning: Optimize parameters such as the number of layers, units, dropout rate, and batch size to improve the model's performance.

Use Case:

While the LSTM model shows potential for identifying general trends, it may not yet be suitable for precise short-term predictions.

In conclusion, the LSTM model provides a baseline prediction of Sony's stock prices. Enhancing the model's complexity and feature set, along with additional tuning, can improve its accuracy for both short- and long-term forecasting. Let me know if you'd like to proceed with any refinements or further analysis.
# ## Regression Analysis on Stock Performance

# In[45]:


import statsmodels.api as sm
from sklearn.model_selection import train_test_split

# Select relevant columns for regression analysis
regression_data = sony_data[['Open', 'High', 'Low', 'Volume', 'Close']]

# Check for missing values
missing_values = regression_data.isnull().sum()

# Drop any rows with missing values (if necessary)
regression_data = regression_data.dropna()

# Define the independent variables (X) and dependent variable (y)
X = regression_data[['Open', 'High', 'Low', 'Volume']]  # Predictors
y = regression_data['Close']  # Target variable

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Add a constant to the predictors (for intercept in regression model)
X_train_const = sm.add_constant(X_train)
X_test_const = sm.add_constant(X_test)

# Check the data shapes to ensure correctness
X_train_const.shape, y_train.shape, X_test_const.shape, y_test.shape, missing_values


# In[46]:


# Build the OLS regression model
ols_model = sm.OLS(y_train, X_train_const).fit()

# Get a summary of the regression model
ols_summary = ols_model.summary()

# Make predictions on the test set
y_pred = ols_model.predict(X_test_const)

# Evaluate the model performance using Mean Squared Error (MSE)
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)

ols_summary, mse


# ## Regression Analysis
R-squared: Indicates that 99.3% of the variance in the closing stock price (Close) is explained by the predictors (Open, High, Low, Volume).
Adjusted R-squared: 0.993
Adjusted for the number of predictors, confirming the model's high explanatory power.
F-statistic: 8133 (p-value: 6.61e-249)
The overall model is statistically significant.

Coefficients:
Intercept (const):
Coefficient: -0.0986 (p = 0.323)
Not statistically significant (p > 0.05).
Opening Price (Open):
Coefficient: -0.5317 (p < 0.001)
Statistically significant. A decrease in the opening price by 1 unit is associated with a decrease of 0.5317 units in the closing price.
High Price (High):
Coefficient: 0.8923 (p < 0.001)
Statistically significant. A unit increase in the high price is associated with a 0.8923 unit increase in the closing price.
Low Price (Low):
Coefficient: 0.6283 (p < 0.001)
Statistically significant. A unit increase in the low price is associated with a 0.6283 unit increase in the closing price.

Mean Squared Error (MSE): 0.173, Low MSE indicates good predictive performance of the model.

Conclusion:
High and Low Prices are the most significant factors influencing Sony's closing stock price.
Opening Price also significantly affects the closing price, but with a negative relationship.
Trading Volume does not significantly influence the closing price, possibly due to multicollinearity or lack of variability in volume.