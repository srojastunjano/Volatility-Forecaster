import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf


def get_realized_variance_yfinance(symbol, period="600d"):
    print(f"Fetching intraday data for {symbol} over the last {period}...")
    
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval="1h")
    
    if df.empty:
        print(f"No data found for {symbol}.")
        return None

    df = df.reset_index()
    
    time_col = 'Datetime' if 'Datetime' in df.columns else 'Date'
    df = df.rename(columns={time_col: 'timestamp', 'Close': 'close'})
    
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    
    df['date'] = df['timestamp'].dt.date
    df = df.dropna(subset=['log_return'])

    # Calculate True Realized Variance: Sum of squared intraday returns
    daily_rv = df.groupby('date')['log_return'].apply(lambda x: np.sum(x**2)).reset_index()
    daily_rv.rename(columns={'log_return': 'realized_variance'}, inplace=True)
    
    return daily_rv

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        window = data[i:(i + seq_length)]
        target = data[i + seq_length]
        X.append(window)
        y.append(target)
    return np.array(X), np.array(y)

def prepare_data():
    rv_data = get_realized_variance_yfinance('AAPL', period='600d')
    rv_values = rv_data[['realized_variance']].values
    
    SEQUENCE_LENGTH = 10
    
    X_raw, y_raw = create_sequences(rv_values, SEQUENCE_LENGTH)

    # (e.g., 80% Train, 20% Test)
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

    return X_train, X_test, y_train, y_test
