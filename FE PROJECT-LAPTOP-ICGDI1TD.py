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
import warnings
warnings.filterwarnings('ignore')

# Setup logging function
def setup_logging():
    logging.basicConfig(filename='stock_prediction.log', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

# Function to fetch stock data
def fetch_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="2y")
        if df.empty:
            print(f"No data found for ticker: {ticker}")
            return None, None
            
        info = stock.info
        company_details = {
            "Company Name": info.get("longName", "N/A"),
            "Market Cap": info.get("marketCap", "N/A"),
            "52-Week High": info.get("fiftyTwoWeekHigh", "N/A"),
            "52-Week Low": info.get("fiftyTwoWeekLow", "N/A"),
            "Live Price": info.get("currentPrice", df['Close'].iloc[-1]),
            "Week Open": info.get("regularMarketOpen", "N/A"),
            "Month Open": info.get("open", "N/A"),
            "15-Day Open": info.get("dayHigh", "N/A")
        }
        return df, company_details
    except Exception as e:
        logging.error(f"Error fetching stock data: {e}")
        print(f"Error fetching stock data: {e}")
        return None, None

# Function to fetch fundamental data
def fetch_fundamental_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        fundamentals = {
            "PE_Ratio": info.get("forwardPE", np.nan),
            "PEG_Ratio": info.get("pegRatio", np.nan),
            "PB_Ratio": info.get("priceToBook", np.nan),
            "Dividend_Yield": info.get("dividendYield", np.nan),
            "Beta": info.get("beta", np.nan),
        }
        # Replace None values with NaN
        for key, value in fundamentals.items():
            if value is None:
                fundamentals[key] = np.nan
        return fundamentals
    except Exception as e:
        logging.error(f"Error fetching fundamental data: {e}")
        print(f"Error fetching fundamental data: {e}")
        return None

# Feature engineering function
def feature_engineering(df, fundamentals=None):
    # Create a copy to avoid modifying original
    df = df.copy()
    
    # Calculate moving averages with proper window sizes
    ma_50_window = min(50, len(df))
    ma_200_window = min(200, len(df))
    
    df['MA_50'] = df['Close'].rolling(window=ma_50_window).mean()
    df['MA_200'] = df['Close'].rolling(window=ma_200_window).mean()
    
    # Calculate volatility
    df['Volatility'] = df['Close'].pct_change().rolling(30).std()
    df['Returns'] = df['Close'].pct_change()
    
    # Technical indicators
    df['RSI'] = compute_rsi(df['Close'])
    df['MACD'], df['Signal'] = compute_macd(df['Close'])
    df['Upper_BB'], df['Lower_BB'] = compute_bollinger_bands(df['Close'])
    
    # Add lagged features
    df['Close_Lag1'] = df['Close'].shift(1)
    df['Close_Lag2'] = df['Close'].shift(2)
    df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
    
    # Incorporate fundamental data into the DataFrame
    if fundamentals:
        for key, value in fundamentals.items():
            if pd.notna(value):
                df[key] = value
            else:
                df[key] = 0  # Fill NaN with 0 for fundamentals
    
    # Drop rows with NaN values
    df = df.dropna()
    
    if len(df) < 50:
        print("Warning: Very few data points after feature engineering. Results may be unreliable.")
    
    return df

# Function to compute RSI
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    # Avoid division by zero
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)  # Fill NaN with neutral RSI value

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
    try:
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        
        # Plot 1: Bollinger Bands
        axes[0].plot(df.index, df['Close'], label='Close Price', color='blue', linewidth=2)
        axes[0].plot(df.index, df['Upper_BB'], label='Upper Bollinger Band', 
                    linestyle='--', color='red', alpha=0.7)
        axes[0].plot(df.index, df['Lower_BB'], label='Lower Bollinger Band', 
                    linestyle='--', color='green', alpha=0.7)
        axes[0].fill_between(df.index, df['Upper_BB'], df['Lower_BB'], alpha=0.1, color='gray')
        axes[0].set_title(f'{ticker} - Bollinger Bands & Close Price')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: MACD
        axes[1].plot(df.index, df['MACD'], label='MACD', color='purple', linewidth=2)
        axes[1].plot(df.index, df['Signal'], label='Signal Line', color='orange', linewidth=2)
        axes[1].bar(df.index, df['MACD'] - df['Signal'], label='MACD Histogram', 
                   color='gray', alpha=0.3)
        axes[1].set_title(f'{ticker} - MACD Indicator')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Additional plot for RSI
        plt.figure(figsize=(15, 5))
        plt.plot(df.index, df['RSI'], label='RSI', color='orange', linewidth=2)
        plt.axhline(y=70, color='r', linestyle='--', alpha=0.7, label='Overbought (70)')
        plt.axhline(y=30, color='g', linestyle='--', alpha=0.7, label='Oversold (30)')
        plt.axhline(y=50, color='gray', linestyle='-', alpha=0.5)
        plt.title(f'{ticker} - RSI Indicator')
        plt.ylabel('RSI')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
    except Exception as e:
        print(f"Error in visualization: {e}")

# Function to plot candlestick chart
def plot_candlestick(df, ticker):
    try:
        # Prepare data for mplfinance
        df_candle = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        
        # Create moving averages for the plot
        mavs = []
        if len(df) >= 50:
            mavs.append(50)
        if len(df) >= 200:
            mavs.append(200)
        
        if mavs:
            mpf.plot(df_candle, type='candle', style='charles', 
                    title=f'{ticker} - Candlestick Chart', 
                    volume=True, mav=tuple(mavs), figsize=(15, 8))
        else:
            mpf.plot(df_candle, type='candle', style='charles', 
                    title=f'{ticker} - Candlestick Chart', 
                    volume=True, figsize=(15, 8))
    except Exception as e:
        print(f"Error plotting candlestick chart: {e}")

# Function to split data
def split_data(df):
    # Define features - use only columns that exist
    base_features = ['Open', 'High', 'Low', 'Volume', 'MA_50', 'MA_200', 
                    'Volatility', 'Returns', 'RSI', 'MACD', 'Signal',
                    'Close_Lag1', 'Close_Lag2', 'Volume_MA']
    
    # Add fundamental features if they exist
    fundamental_features = ['PE_Ratio', 'PEG_Ratio', 'PB_Ratio', 'Dividend_Yield', 'Beta']
    
    # Only use features that exist in the dataframe
    features = [col for col in base_features + fundamental_features if col in df.columns]
    
    print(f"Using features: {features}")
    
    X = df[features]
    y = df['Close']
    
    # Check for any remaining NaN values
    if X.isnull().any().any():
        print("Warning: NaN values found in features. Filling with forward fill method.")
        X = X.fillna(method='ffill').fillna(method='bfill')
    
    return train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

# Function for hyperparameter tuning
def hyperparameter_tuning(model, params, X, y):
    try:
        grid_search = GridSearchCV(model, params, cv=3, scoring='neg_mean_squared_error', 
                                 n_jobs=-1, verbose=0)
        grid_search.fit(X, y)
        return grid_search.best_estimator_
    except Exception as e:
        print(f"Error in hyperparameter tuning: {e}")
        return model.fit(X, y)

# Function for cross-validation
def cross_validate_model(model, X, y):
    try:
        scores = cross_val_score(model, X, y, cv=3, scoring='neg_mean_squared_error')
        return np.mean(np.abs(scores))
    except Exception as e:
        print(f"Error in cross-validation: {e}")
        return 0

# Function to ensemble predictions
def ensemble_predictions(predictions):
    # Remove LSTM and Ensemble from predictions for ensemble calculation
    model_preds = {k: v for k, v in predictions.items() 
                  if k not in ['LSTM', 'Ensemble']}
    if model_preds:
        return np.mean(list(model_preds.values()), axis=0)
    else:
        return np.zeros(len(list(predictions.values())[0]))

# Function to train models
def train_models(X_train, y_train, X_test):
    models = {
        "Linear Regression": LinearRegression(),
        "SVR": SVR(kernel='rbf', C=100, gamma='scale'),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    }
    
    results = {}
    for name, model in models.items():
        try:
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            results[name] = predictions
            print(f"{name} trained successfully")
        except Exception as e:
            print(f"Error training {name}: {e}")
    
    return results

# Function to train LSTM model
def train_lstm(X_train, y_train, X_test):
    try:
        print("Training LSTM model...")
        
        # Scale the data
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Scale target variable
        y_scaler = MinMaxScaler()
        y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
        
        # Reshape for LSTM (samples, timesteps, features)
        X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
        X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))
        
        # Build LSTM model
        model = Sequential([
            Input(shape=(1, X_train_scaled.shape[1])),
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        # Train model
        history = model.fit(X_train_lstm, y_train_scaled, 
                          epochs=50, batch_size=32, 
                          validation_split=0.2, 
                          verbose=0)
        
        # Make predictions
        predictions_scaled = model.predict(X_test_lstm)
        predictions = y_scaler.inverse_transform(predictions_scaled).flatten()
        
        print("LSTM model trained successfully")
        return predictions
        
    except Exception as e:
        print(f"Error training LSTM: {e}")
        # Return a simple prediction based on mean
        return np.full(len(X_test), y_train.mean())

# Function to plot predictions
def plot_predictions(y_test, predictions):
    try:
        plt.figure(figsize=(16, 8))
        
        # Plot actual values
        plt.plot(y_test.index, y_test.values, label='Actual Prices', 
                color='black', linewidth=3, alpha=0.8)
        
        # Plot predictions
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
        for i, (model_name, preds) in enumerate(predictions.items()):
            plt.plot(y_test.index, preds, label=model_name, 
                    color=colors[i % len(colors)], linewidth=2, alpha=0.7)
        
        plt.title('Model Predictions vs Actual Prices', fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Price ($)', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error plotting predictions: {e}")

# Function to plot error histogram
def plot_error_histogram(y_test, predictions):
    try:
        n_models = len(predictions)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, (model_name, preds) in enumerate(predictions.items()):
            if i < len(axes):
                errors = y_test.values - preds
                axes[i].hist(errors, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
                axes[i].axvline(0, color='red', linestyle='--', linewidth=2)
                axes[i].set_title(f'{model_name} - Prediction Errors')
                axes[i].set_xlabel('Error ($)')
                axes[i].set_ylabel('Frequency')
                axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(predictions), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error plotting error histogram: {e}")

# Function to infer model performance
def infer_model_performance(y_test, predictions):
    performance_results = {}
    
    print("\n" + "="*60)
    print("MODEL PERFORMANCE ANALYSIS")
    print("="*60)
    
    for model_name, preds in predictions.items():
        try:
            mae = mean_absolute_error(y_test, preds)
            mse = mean_squared_error(y_test, preds)
            rmse = np.sqrt(mse)
            
            # Calculate percentage error
            mape = np.mean(np.abs((y_test - preds) / y_test)) * 100
            
            performance_results[model_name] = {
                "First 5 Predictions": preds[:5],
                "Mean Absolute Error": mae,
                "Mean Squared Error": mse,
                "Root Mean Squared Error": rmse,
                "Mean Absolute Percentage Error": mape
            }
            
            print(f"\n{model_name}:")
            print(f"  MAE:  ${mae:.2f}")
            print(f"  RMSE: ${rmse:.2f}")
            print(f"  MAPE: {mape:.2f}%")
            print(f"  First 5 Predictions: {[f'${p:.2f}' for p in preds[:5]]}")
            
        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
    
    # Log the performance results
    for model_name, metrics in performance_results.items():
        try:
            logging.info(f"{model_name} - MAE: {metrics['Mean Absolute Error']:.4f}, "
                        f"RMSE: {metrics['Root Mean Squared Error']:.4f}, "
                        f"MAPE: {metrics['Mean Absolute Percentage Error']:.4f}")
        except:
            pass
    
    # Find best performing model
    if performance_results:
        best_model = min(performance_results.items(), 
                        key=lambda x: x[1]['Mean Absolute Error'])
        print(f"\n🏆 Best Performing Model: {best_model[0]} (Lowest MAE: ${best_model[1]['Mean Absolute Error']:.2f})")
    
    return performance_results

# Main function
def main():
    setup_logging()
    
    # Get ticker from user
    ticker = input("Enter Stock Ticker (e.g., AAPL, GOOGL, MSFT): ").strip().upper()
    
    if not ticker:
        print("Please enter a valid ticker symbol")
        return
    
    print(f"\nFetching data for {ticker}...")
    
    # Fetch stock data
    df, company_details = fetch_stock_data(ticker)
    
    if df is None or df.empty:
        print("Failed to fetch stock data. Please check the ticker symbol and try again.")
        return
    
    print(f"\nSuccessfully fetched {len(df)} days of data")
    
    # Display company details
    print("\n" + "="*50)
    print("COMPANY DETAILS")
    print("="*50)
    for key, value in company_details.items():
        print(f"{key:15}: {value}")
    
    # Fetch fundamental data
    print("\nFetching fundamental data...")
    fundamentals = fetch_fundamental_data(ticker)
    
    if fundamentals:
        print("\nFundamental Ratios:")
        for key, value in fundamentals.items():
            if pd.notna(value):
                print(f"{key:15}: {value:.2f}" if isinstance(value, (int, float)) else f"{key:15}: {value}")
    
    # Feature engineering
    print("\nPerforming feature engineering...")
    df = feature_engineering(df, fundamentals)
    
    if len(df) < 100:
        print("Warning: Limited data available. Results may not be reliable.")
    
    # Visualizations
    print("\nGenerating visualizations...")
    visualize_technical_indicators(df, ticker)
    plot_candlestick(df, ticker)
    
    # Split data
    print("\nSplitting data for training and testing...")
    X_train, X_test, y_train, y_test = split_data(df)
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Train models
    print("\nTraining machine learning models...")
    predictions = train_models(X_train, y_train, X_test)
    
    # Train LSTM
    if len(X_train) > 50:  # Only train LSTM if we have enough data
        predictions["LSTM"] = train_lstm(X_train, y_train, X_test)
    
    # Ensemble predictions
    if len(predictions) > 1:
        predictions["Ensemble"] = ensemble_predictions(predictions)
    
    if not predictions:
        print("No models were successfully trained.")
        return
    
    # Evaluate performance
    performance_results = infer_model_performance(y_test, predictions)
    
    # Generate plots
    print("\nGenerating prediction plots...")
    plot_predictions(y_test, predictions)
    plot_error_histogram(y_test, predictions)
    
    # Error analysis
    print("\n" + "="*60)
    print("ERROR ANALYSIS")
    print("="*60)
    
    for model_name, preds in predictions.items():
        try:
            errors = y_test.values - preds
            print(f"\n{model_name}:")
            print(f"  Mean Error: ${errors.mean():.2f}")
            print(f"  Std Error:  ${errors.std():.2f}")
            print(f"  Min Error:  ${errors.min():.2f}")
            print(f"  Max Error:  ${errors.max():.2f}")
        except Exception as e:
            print(f"Error analyzing {model_name}: {e}")
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
