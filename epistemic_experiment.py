import numpy as np
import yfinance as yf
from architecture.inference import generate_credal_set
from architecture.transformer import PositionalEncoding
from architecture.evidential_head import EvidentialRegressionHead
from architecture.MCDropout import MCDropout
from architecture.transformer import TransformerEncoderBlock
import tensorflow as tf
from architecture.data_extraction import prepare_data
import pandas as pd

def prepare_alien_data(alien_ticker, original_scaler, window_size, period="5y"):
    """
    Stressor 2: The Domain Shifter
    Fetches OOD data (e.g., Crypto) and maps it through the Stock Scaler.
    """
    print(f"Fetching Alien Domain Data: {alien_ticker}...")

    df = yf.download(alien_ticker, period=period, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
        
    # Example proxy for Realized Variance if you don't have intraday data:
    # (Daily Return)^2 as a rough daily variance proxy for the sake of the experiment
    returns = np.log(df['Adj Close'] / df['Adj Close'].shift(1)).dropna()
    raw_rv = (returns ** 2).values.reshape(-1, 1)
    
    # 2. THE CRITICAL STEP: Scale using the TSLA scaler
    # Do NOT call fit_transform(). Only call transform().
    alien_scaled = original_scaler.transform(raw_rv)
    
    # 3. Create the 3D overlapping windows for the Transformer
    X_alien = []
    y_alien = []
    for i in range(len(alien_scaled) - window_size):
        X_alien.append(alien_scaled[i : i + window_size])
        y_alien.append(alien_scaled[i + window_size])
        
    return np.array(X_alien), np.array(y_alien)

def evaluate_domain(model, scaler, X_data, domain_name, K):
        print(f"Evaluating Domain: {domain_name}...")
        aleatoric_list = []
        epistemic_list = []
        credal_spread_list = []
        
        for i in range(len(X_data)):
            # Force MC Dropout to remain active
            res = generate_credal_set(model, X_data[i:i+1], scaler, K=K)
            
            aleatoric_list.append(np.mean(res['aleatoric']))
            epistemic_list.append(np.mean(res['epistemic_nig']))
            credal_spread_list.append(res['credal_epistemic_var'])
            
        return {
            'Mean_Aleatoric': np.mean(aleatoric_list),
            'Mean_Epistemic': np.mean(epistemic_list),
            'Mean_Credal_Spread': np.mean(credal_spread_list)
        }

def run_ood_domain_test(model, X_baseline, X_alien, scaler, K=100):
    """
    Compares the uncertainty profile of Baseline (TSLA) vs. Alien (BTC).
    """
    print("\n--- Starting Domain Shift Test (OOD Evaluation) ---")
    
    results = {}

    # Evaluate both domains
    results['Baseline (Stocks)'] = evaluate_domain(model, scaler, X_baseline, "Baseline (TSLA),", K)
    results['Alien (Crypto)'] = evaluate_domain(model, scaler, X_alien, "Alien (BTC-USD)", K)
    
    # Print the Disentanglement Comparison
    print("\n--- Disentanglement Results ---")
    for domain, metrics in results.items():
        print(f"\n{domain}:")
        print(f"  Aleatoric (Noise):    {metrics['Mean_Aleatoric']:.6f}")
        print(f"  Epistemic (Ignorance):{metrics['Mean_Epistemic']:.6f}")
        print(f"  Credal Variance:      {metrics['Mean_Credal_Spread']:.6f}")
        
    return results