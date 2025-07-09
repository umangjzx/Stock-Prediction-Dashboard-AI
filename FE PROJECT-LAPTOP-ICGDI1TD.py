import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import logging
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Bidirectional, Input
import mplfinance as mpf

# Setup logging function
def setup_logging():
    logging.basicConfig(filename='stock_prediction.log', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

# Function to fetch stock data
def fetch_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="2y")
        info = stock.info
        company_details = {
            "Company Name": info.get("longName", "N/A"),
            "Market Cap": info.get("marketCap", "N/A"),
            "52-Week High": info.get("fiftyTwoWeekHigh", "N/A"),
            "52-Week Low": info.get("fiftyTwoWeekLow", "N/A"),
            "Live Price": info.get("currentPrice", "N/A"),
            "Week Open": info.get("regularMarketOpen", "N/A"),
            "Month Open": info.get("open", "N/A"),
            "15-Day Open": info.get("dayHigh", "N/A")
        }
        return df, company_details
    except Exception as e:
        logging.error(f"Error fetching stock data: {e}")
        return None, None

# Function to fetch fundamental data
def fetch_fundamental_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        fundamentals = {
            "PE Ratio": stock.info.get("forwardPE", "N/A"),
            "PEG Ratio": stock.info.get("pegRatio", "N/A"),
            "PB Ratio": stock.info.get("priceToBook", "N/A"),
            "Dividend Yield": stock.info.get("dividendYield", "N/A"),
            "Beta": stock.info.get("beta", "N/A"),
        }
        return fundamentals
    except Exception as e:
        logging.error(f"Error fetching fundamental data: {e}")
        return None

# Feature engineering function
def feature_engineering(df, fundamentals=None):
    df['MA_50'] = df['Close'].rolling(window=min(50, len(df))).mean()
    df['MA_200'] = df['Close'].rolling(window=min(200, len(df))).mean()
    df['Volatility'] = df['Close'].pct_change().rolling(30).std()
    df['Returns'] = df['Close'].pct_change()
    df['RSI'] = compute_rsi(df['Close'])
    df['MACD'], df['Signal'] = compute_macd(df['Close'])
    df['Upper_BB'], df['Lower_BB'] = compute_bollinger_bands(df['Close'])

    # Incorporate fundamental data into the DataFrame
    if fundamentals:
        for key, value in fundamentals.items():
            df[key] = value

    df.dropna(inplace=True)
    return df

# Function to compute RSI
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Function to compute MACD
def compute_macd(series, short_window=12, long_window=26, signal_window=9):
    short_ema = series.ewm(span=short_window, adjust=False).mean()
    long_ema = series.ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, signal

# Function to compute Bollinger Bands
def compute_bollinger_bands(series, window=20, num_std=2):
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return upper_band, lower_band

# Function to visualize technical indicators
def visualize_technical_indicators(df, ticker):
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Close'], label='Close Price', color='blue')
    plt.plot(df.index, df['Upper_BB'], label='Upper Bollinger Band', linestyle='dashed', color='red')
    plt.plot(df.index, df['Lower_BB'], label='Lower Bollinger Band', linestyle='dashed', color='green')
    plt.title(f'{ticker} Bollinger Bands & Close Price')
    plt.legend()
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['MACD'], label='MACD', color='purple')
    plt.plot(df.index, df['Signal'], label='Signal Line', color='orange')
    plt.title(f'{ticker} MACD Indicator')
    plt.legend()
    plt.show()

# Function to plot candlestick chart
def plot_candlestick(df, ticker):
    mpf.plot(df, type='candle', style='charles', title=f'{ticker} Candlestick Chart', volume=True, mav=(50, 200))

# Function to split data
def split_data(df):
    features = ['Open', 'High', 'Low', 'MA_50', 'MA_200', 'Volatility', 'Returns', 'RSI', 'MACD', 'Signal']
    X = df[features]
    y = df['Close']
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Function for hyperparameter tuning
def hyperparameter_tuning(model, params, X, y):
    grid_search = GridSearchCV(model, params, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X, y)
    return grid_search.best_estimator_

# Function for cross-validation
def cross_validate_model(model, X, y):
    scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    return np.mean(np.abs(scores))

# Function to ensemble predictions
def ensemble_predictions(predictions):
    return np.mean(list(predictions.values()), axis=0)

# Function to train models
def train_models(X_train, y_train, X_test):
    models = {
        "Linear Regression": LinearRegression(),
        "SVR": SVR(kernel='rbf', C=1000, gamma=0.1),
        "Random Forest": RandomForestRegressor(n_estimators=300, max_depth=15, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=7)
    }
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        results[name] = predictions
    return results

# Function to train LSTM model
def train_lstm(X_train, y_train, X_test):
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Reshape for LSTM
    X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
    X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

    model = Sequential([
        Input(shape=(1, X_train_scaled.shape[1])),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.2),
        Bidirectional(LSTM(64, return_sequences=False)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train_lstm, y_train, epochs=50, batch_size=32, verbose=1)

    return model.predict(X_test_lstm)

# Function to plot predictions
def plot_predictions(y_test, predictions):
    plt.figure(figsize=(14, 7))
    plt.plot(y_test.index, y_test.values, label='Actual Prices', color='black', linewidth=2)
    for model_name, preds in predictions.items():
        plt.plot(y_test.index, preds, label=model_name)
    plt.title('Model Predictions vs Actual Prices')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()

# Function to plot error histogram
def plot_error_histogram(y_test, predictions):
    for model_name, preds in predictions.items():
        errors = y_test - preds
        plt.figure(figsize=(14, 7))
        sns.histplot(errors, bins=30, kde=True)
        plt.title(f'Prediction Errors for {model_name}')
        plt.xlabel('Error')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.axvline(0, color='red', linestyle='--')
        plt.show()

# Function to infer model performance
def infer_model_performance(y_test, predictions):
    performance_results = {}
    for model_name, preds in predictions.items():
        mae = mean_absolute_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        performance_results[model_name] = {
            "First 5 Predictions": preds[:5],
            "Mean Absolute Error": mae,
            "Mean Squared Error": mse
        }
        print(f"\nModel: {model_name}")
        print(f"First 5 Predictions: {preds[:5]}...")
        print(f"Mean Absolute Error: {mae:.4f}")
        print(f"Mean Squared Error: {mse:.4f}")
    
    # Log the performance results
    for model_name, metrics in performance_results.items():
        logging.info(f"{model_name} - First 5 Predictions: {metrics['First 5 Predictions']}, "
                     f"MAE: {metrics['Mean Absolute Error']:.4f}, "
                     f"MSE: {metrics['Mean Squared Error']:.4f}")
    
    # New function to print inferences
    print_inferences(performance_results)

# Function to print inferences
def print_inferences(performance_results):
    print("\nInference Summary:")
    for model_name, metrics in performance_results.items():
        print(f"{model_name}:")
        print(f"  - First 5 Predictions: {metrics['First 5 Predictions']}")
        print(f"  - Mean Absolute Error: {metrics['Mean Absolute Error']:.4f}")
        print(f"  - Mean Squared Error: {metrics['Mean Squared Error']:.4f}")

# Main function
if __name__ == "__main__":
    setup_logging()
    ticker = input("Enter Stock Ticker: ").strip().upper()
    df, company_details = fetch_stock_data(ticker)

    if df is not None:
        print("Company Details:")
        for key, value in company_details.items():
            print(f"{key}: {value}")

        # Fetch fundamental data
        fundamentals = fetch_fundamental_data(ticker)
        df = feature_engineering(df, fundamentals)
        visualize_technical_indicators(df, ticker)
        plot_candlestick(df, ticker)

        X_train, X_test, y_train, y_test = split_data(df)

        # Hyperparameter tuning for the models
        models = {
            "Linear Regression": LinearRegression(),
            "SVR": SVR(),
            "Random Forest": RandomForestRegressor(),
            "XGBoost": XGBRegressor()
        }

        params = {
            "SVR": {'C': [0.1, 1, 10], 'gamma': [0.01, 0.1, 1]},
            "Random Forest": {'n_estimators': [100, 200], 'max_depth': [10, 15, None]},
            "XGBoost": {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 5, 7]}
        }

        best_models = {}
        for name, model in models.items():
            if name in params:
                best_model = hyperparameter_tuning(model, params[name], X_train, y_train)
                best_models[name] = best_model
                print(f"Best parameters for {name}: {best_model.get_params()}")
            else:
                best_models[name] = model.fit(X_train, y_train)

        # Cross-validation results
        for name, model in best_models.items():
            cv_score = cross_validate_model(model, X_train, y_train)
            print(f"{name} Cross-Validation Score: {-cv_score:.4f}")

        # Collect predictions
        predictions = {name: model.predict(X_test) for name, model in best_models.items()}

        # Train LSTM and add predictions
        predictions["LSTM"] = train_lstm(X_train, y_train, X_test).flatten()

        # Ensemble predictions
        predictions["Ensemble"] = ensemble_predictions(predictions)

        # Print predictions and evaluation metrics
        infer_model_performance(y_test, predictions)

        # Plot predictions
        plot_predictions(y_test, predictions)

        # Plot error histograms
        plot_error_histogram(y_test, predictions)

        # Error analysis
        for model_name, preds in predictions.items():
            errors = y_test - preds
            print(f"{model_name} Error Analysis: Mean Error: {errors.mean()}, Std Error: {errors.std()}")
