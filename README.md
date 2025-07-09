# 📈 Stock Prediction Dashboard using AI & Technical Analysis

A full-featured Python-based dashboard for stock price prediction using both **machine learning (ML)** and **deep learning (DL)** models, enriched with technical indicators and candlestick visualizations.

## 🚀 Features

- 📊 **Technical Indicator Dashboard**: Visualizes Bollinger Bands, Moving Averages, RSI, and MACD.
- 🕯️ **Candlestick Charts**: Styled charts with volume overlays using `mplfinance`.
- 🔍 **Machine Learning Models**:
  - Linear Regression
  - Support Vector Regression (SVR)
  - Random Forest Regressor
  - XGBoost
- 🧠 **Deep Learning (LSTM)**: Bidirectional LSTM with dropout for time-series prediction.
- 🧮 **Feature Engineering**: Includes volatility, returns, RSI, MACD, Bollinger Bands.
- 📈 **Fundamental Analysis Integration**: Pulls live PE, PB, Beta, etc., via `yfinance`.
- 🧪 **Model Performance Metrics**: MAE, RMSE, MSE with error plots.
- 🧠 **Ensemble Predictions**: Blends model outputs for improved accuracy.
- 🔔 **Buy/Sell Signal Generation**: Based on predictions and indicator crossovers.

## 📌 Installation

```bash
git clone https://github.com/your-username/Stock-Prediction-Dashboard-AI.git
cd Stock-Prediction-Dashboard-AI
pip install -r requirements.txt
