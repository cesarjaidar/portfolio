# 📈 Stock Price Analysis and Forecasting of Top Video Game Companies

## 📄 Project Overview  
This project focuses on analyzing and forecasting the stock price trends of the top video game companies using time-series analysis and machine learning models. The goal is to visualize historical trends, detect patterns, and predict future stock prices to provide actionable insights for investors and analysts.  

---

## 🚀 Key Objectives  
- **Visualize stock price and trading volume trends** for major video game companies.  
- **Identify patterns** using ARIMA and LSTM models for time-series forecasting.  
- **Predict future stock prices** and evaluate model performance with metrics like MSE and R².  

---

## 🛠️ Tools and Libraries  
- **Pandas** – Data manipulation and preprocessing  
- **Matplotlib / Seaborn** – Data visualization  
- **Statsmodels** – ARIMA modeling for time-series analysis  
- **TensorFlow / Keras** – LSTM neural networks for stock prediction  
- **Scikit-Learn** – Model evaluation (MSE, R²)  

---

## 📊 Data Source  
- **Dataset**: `Top10VideoGameStocks.csv` from Kaggle & Yahoo Finances
- **Description**: Contains stock price data (open, close, high, low) and trading volumes for leading video game companies.  

---

## 🔍 Key Features  
- **Trend Visualization** – Line plots for stock prices and trading volume over time.  
- **Time-Series Forecasting** – ARIMA and LSTM models are implemented to predict stock prices.  
- **Model Evaluation** – Compare forecasting performance using metrics such as Mean Squared Error (MSE) and R².  

---

## ⚙️ Project Workflow  

1. **Data Loading and Preprocessing**  
   - Import data and handle missing values.  
   - Convert date columns to `datetime` and set them as index.  

2. **Exploratory Data Analysis (EDA)**  
   - Visualize historical stock prices and trading volumes.  
   - Detect duplicates and clean data.  

3. **Time-Series Modeling (ARIMA)**  
   - Use ACF and PACF plots to identify ARIMA parameters (p, d, q).  
   - Fit the ARIMA model and forecast future prices.  

4. **LSTM Neural Network**  
   - Scale data and reshape it for LSTM input.  
   - Train and evaluate LSTM models to capture long-term trends.  

5. **Model Evaluation**  
   - Compare ARIMA and LSTM predictions.  
   - Use metrics like MSE and R² to assess model accuracy.  

---

## 📈 Visuals and Outputs  

- **Stock Price Trends**  
  ![Stock Trends](path/to/stock_trend_plot.png)  

- **Forecasted vs Actual Prices (ARIMA and LSTM)**  
  ![ARIMA vs LSTM](path/to/arima_lstm_comparison.png)  

---

## 🧩 Key Findings  
- **LSTM models** excel in capturing long-term dependencies but require more data and tuning.  
- **ARIMA** is effective for short-term forecasting and easier to interpret.  
- Combining both methods can improve accuracy and stability in predictions.  

---

## 📂 Project Structure  
