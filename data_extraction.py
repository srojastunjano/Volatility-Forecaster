import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf


def get_realized_variance_yfinance(symbol, period="max"):
    print(f"Fetching intraday data for {symbol} over the last {period}...")
    
    ticker = yf.Ticker(symbol)
    # Switch to daily data for long-term horizons
    df = ticker.history(period=period, interval="1d")
    
    # Calculate daily log returns
    df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Target: 252-day Rolling Realized Variance (Annualized)
    # We square the returns and sum them over a 252-day sliding window
    df['target_rv'] = df['log_return'].pow(2).rolling(window=252).sum()
    
    # IMPORTANT: We must shift the target so we are predicting the FUTURE year
    # based on the CURRENT sequence.
    df['target_rv'] = df['target_rv'].shift(-252)
    
    return df.dropna()

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        window = data[i:(i + seq_length)]
        target = data[i + seq_length]
        X.append(window)
        y.append(target)
    return np.array(X), np.array(y)

def prepare_data(symbol='NVDA', period='max', seq_length=22):

    rv_data = get_realized_variance_yfinance(symbol, period=period)
    rv_values = rv_data[['target_rv']].values

    if rv_data is None or rv_data.empty:
        raise ValueError(
            f"Data extraction failed for {symbol}. "
            f"Check your period argument ('{period}') or your internet connection."
        )
    
    X_raw, y_raw = create_sequences(rv_values, seq_length)

    # (80% Train, 20% Test)
    split_idx = int(len(X_raw) * 0.8)
    
    X_train_raw, X_test_raw = X_raw[:split_idx], X_raw[split_idx:]
    y_train_raw, y_test_raw = y_raw[:split_idx], y_raw[split_idx:]
    
    scaler = MinMaxScaler()
    
    # must flatten X_train to fit the scaler, then reshape back to 3D
    # (samples * seq_length, features)
    X_train_flat = X_train_raw.reshape(-1, 1) 
    scaler.fit(X_train_flat) 
    
    X_train = scaler.transform(X_train_raw.reshape(-1, 1)).reshape(X_train_raw.shape)
    y_train = scaler.transform(y_train_raw.reshape(-1, 1)).reshape(y_train_raw.shape)

    X_test = scaler.transform(X_test_raw.reshape(-1, 1)).reshape(X_test_raw.shape)
    y_test = scaler.transform(y_test_raw.reshape(-1, 1)).reshape(y_test_raw.shape)

    print("\n--- Model Ready Tensors ---")
    print(f"X_train shape: {X_train.shape} -> [samples, seq_length, features]")
    print(f"y_train shape: {y_train.shape} -> [samples, features]")
    print(f"X_test shape:  {X_test.shape}")
    print(f"y_test shape:  {y_test.shape}")

    return X_train, X_test, y_train, y_test, scaler
