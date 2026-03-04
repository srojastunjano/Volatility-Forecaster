import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import data_extraction

rv_data = data_extraction.get_realized_variance_yfinance('AAPL', period='60d')
rv_values = rv_data[['realized_variance']].values

scaler = StandardScaler()
rv_scaled = scaler.fit_transform(rv_values)

# efine the sliding window function
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        window = data[i:(i + seq_length)]
        target = data[i + seq_length]
        
        X.append(window)
        y.append(target)
        
    return np.array(X), np.array(y)

SEQUENCE_LENGTH = 10

X, y = create_sequences(rv_scaled, SEQUENCE_LENGTH)

print(f"Original data shape: {rv_scaled.shape}")
print(f"X (Input) shape: {X.shape} -> [samples, sequence_length, num_features]")
print(f"y (Target) shape: {y.shape} -> [samples, target_features]")
print(f"X (Input) shape: {X} -> [samples, sequence_length, num_features]")
print(f"y (Target) shape: {y} -> [samples, target_features]")