import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL
from sklearn.preprocessing import RobustScaler
from tqdm import tqdm
import h5py

import os

# def get_sp500_tickers(filepath='tickers.txt'):
def get_sp500_tickers(filepath='stock_test.txt'):
    print(f"Reading S&P 500 Ticker List from {filepath}...")
        
    with open(filepath, 'r') as file:
        tickers = [line.strip() for line in file if line.strip()]
    
    print(tickers)
    return [t.replace('.', '-') for t in tickers]

def process_single_stock(symbol, seq_len=21):
    """
    STEP 1: The Global Hybrid Data Pipeline (Local Processing)
    Fetches data, calculates Log-RV, and performs STL decomposition.
    """
    try:

        df = yf.download(symbol, period="max", interval="1d", progress=False)
        
        if len(df) < 500:
            return None, None, None 
            
        # If yfinance returns a MultiIndex column, flattens it
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)

        # Calculate Realized Variance (21-day rolling window)
        df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
        df['rv'] = df['log_return'].pow(2).rolling(window=21).sum()
        
        # The Log-Volatility Transformation
        df['log_rv'] = np.log(df['rv'] + 1e-9)
        df = df.dropna()

        # STL Decomposition
        stl = STL(df['log_rv'], period=21, seasonal=21, robust=True)
        res = stl.fit()
        
        df['trend'] = res.trend
        df['seasonal'] = res.seasonal

        features = df[['log_rv', 'trend', 'seasonal']].values
        
        # Robust Scaling (Per-Stock)
        scaler = RobustScaler()
        scaled_features = scaler.fit_transform(features)
        
        scaled_log_rv = scaled_features[:, 0]
        scaled_trend = scaled_features[:, 1]
        scaled_seasonal = scaled_features[:, 2]

        # Panel-Stacked Sequencing
        X_temporal, X_contextual, y_target = [], [], []
        
        # Slide the window across this specific stock's history
        for i in range(len(scaled_features) - seq_len - 1):
           # past month of Log-RV
            window_rv = scaled_log_rv[i : i + seq_len]
            
            # Trend and Seasonal state at the END of the window
            current_trend = scaled_trend[i + seq_len - 1]
            current_seasonal = scaled_seasonal[i + seq_len - 1]
            
            # The Log-RV of the NEXT day (t + 1)
            target = scaled_log_rv[i + seq_len]
            
            X_temporal.append(window_rv)
            X_contextual.append([current_trend, current_seasonal])
            y_target.append(target)
            
        return np.array(X_temporal), np.array(X_contextual), np.array(y_target), scaler

    except Exception as e:
        return None, None, None


def build_global_tensor(max_stocks=500, seq_len=21):
    """
    Orchestrates the pipeline and stacks the individual stock matrices 
    into the massive Global Tensor for the Hybrid Model.
    """
    tickers = get_sp500_tickers()[:max_stocks]
    
    global_X_temp = []
    global_X_cont = []
    global_y = []
    
    successful_stocks = 0
    
    print(f"\nProcessing {len(tickers)} stocks for Global Panel...")
    
    for ticker in tqdm(tickers, desc="Compiling Global Tensor"):
        X_temp, X_cont, y, scaler = process_single_stock(ticker, seq_len=seq_len)
        
        if X_temp is not None:
            global_X_temp.append(X_temp)
            global_X_cont.append(X_cont)
            global_y.append(y)
            successful_stocks += 1

        # print("successfull stock: ", successful_stocks, global_X_temp, global_X_cont, global_y)     

    global_X_temp = np.concatenate(global_X_temp, axis=0)
    global_X_cont = np.concatenate(global_X_cont, axis=0)
    global_y = np.concatenate(global_y, axis=0)
    # Reshape Temporal X to explicitly include the feature dimension for the Transformer
    # From shape (samples, seq_len) -> (samples, seq_len, 1)
    global_X_temp = np.expand_dims(global_X_temp, axis=-1)
    
    print("\n--- Global Tensor Compilation Complete ---")
    print(f"Total Stocks Successfully Processed: {successful_stocks}")
    print(f"Global Temporal Shape (Transformer Input): {global_X_temp.shape}")
    print(f"Global Contextual Shape (Dense Input):     {global_X_cont.shape}")
    print(f"Global Target Shape:                       {global_y.shape}")
    
    return global_X_temp, global_X_cont, global_y, scaler


X_temporal, X_contextual, y_target, scaler = build_global_tensor(max_stocks=500, seq_len=21)


np.savez_compressed('global_panel_data.npz', 
                    temporal=X_temporal, 
                    contextual=X_contextual, 
                    target=y_target)

# with open('input_data.txt', 'w') as f:
#     f.write("# X_temporal\n")
#     # Flattening for 2D saving if the tensor is > 2D
#     np.savetxt(f, X_temporal.reshape(X_temporal.shape[0], -1))
    
#     f.write("\n# X_contextual\n")
#     np.savetxt(f, X_contextual.reshape(X_contextual.shape[0], -1))
    
#     f.write("\n# y_target\n")
#     np.savetxt(f, y_target.reshape(y_target.shape[0], -1))

# print("Results saved to model_results.txt")


