import numpy as np
import matplotlib.pyplot as plt
from architecture.inference import generate_credal_set

def inject_ood_shocks(X_test, y_test, shock_type='crash', start_idx=50, duration=10, feature_idx=0):
    """
    Injects synthetic Out-of-Distribution (OOD) shocks into the test set tensors.
    
    Parameters:
    - X_test: 3D numpy array [samples, time_steps, features]
    - y_test: 1D or 2D numpy array of targets
    - shock_type: 'crash' (10x multiplier) or 'zombie' (flatline near zero)
    - start_idx: The index in the test set where the shock begins
    - duration: How many consecutive days the shock lasts
    - feature_idx: The index of the RV feature in the 3rd dimension of X_test (usually 0)
    """
    X_ood = np.copy(X_test)
    y_ood = np.copy(y_test)
    
    end_idx = min(start_idx + duration, len(X_ood))
    
    if shock_type == 'crash':
        # Simulate a massive volatility spike entering the Transformer's lookback window
        X_ood[start_idx:end_idx, :, feature_idx] *= 10.0
        y_ood[start_idx:end_idx] *= 10.0
    elif shock_type == 'zombie':
        # Simulate a complete loss of market signal
        X_ood[start_idx:end_idx, :, feature_idx] = 1e-6
        y_ood[start_idx:end_idx] = 1e-6
        
    return X_ood, y_ood


def run_abstention_experiment(model, X_test, y_test, scaler, K=100, shock_type='crash'):
    """
    Runs the model on OOD data, calculates total uncertainty, and generates the Risk-Abstention curve.
    """
    print(f"\n--- Starting Abstention Experiment: OOD Type '{shock_type.upper()}' ---")
    
    X_ood, y_ood = inject_ood_shocks(X_test, y_test, shock_type=shock_type, start_idx=50, duration=10)
    
    prediction_data = []
    
    # Run Inference Loop
    for i in range(len(X_ood)):
        # Call your existing function (assumes it is loaded in your namespace)
        res = generate_credal_set(model, X_ood[i:i+1], scaler, K=K)
        
        actual_scaled = y_ood[i]
        actual_unscaled = scaler.inverse_transform(actual_scaled.reshape(-1, 1))[0,0]
        forecast = res['mean_forecast']
        
        sq_error = (actual_unscaled - forecast)**2
        
        # Calculate Uncertainty Components
        # We combine the NIG evidence and the Credal Variance for a robust Epistemic metric
        epistemic = np.mean(res['epistemic_nig']) + res['credal_epistemic_var']
        aleatoric = np.mean(res['aleatoric'])
        
        # Total Uncertainty is the sum of noise + ignorance
        total_uncertainty = epistemic + aleatoric
        
        prediction_data.append({
            'index': i,
            'sq_error': sq_error,
            'epistemic': epistemic,
            'aleatoric': aleatoric,
            'total_uncertainty': total_uncertainty
        })

    # Calculate the Risk-Abstention Curve (AURAC Logic)
    # Sort predictions by Total Uncertainty (Highest to Lowest)
    sorted_preds = sorted(prediction_data, key=lambda x: x['total_uncertainty'], reverse=True)
    
    # test abstaining from 0% up to 50% of the most uncertain predictions
    abstention_fractions = np.linspace(0, 0.5, 21) 
    mse_curve = []
    
    for frac in abstention_fractions:
        drop_count = int(frac * len(sorted_preds))
        kept_preds = sorted_preds[drop_count:] 
        
        if len(kept_preds) > 0:
            current_mse = np.mean([p['sq_error'] for p in kept_preds])
        else:
            current_mse = 0
        mse_curve.append(current_mse)
        
    print(f"Base MSE (0% Abstention): {mse_curve[0]:.6f}")
    print(f"MSE after rejecting top 10% uncertain: {mse_curve[4]:.6f}")
    print(f"MSE after rejecting top 25% uncertain: {mse_curve[10]:.6f}")
    
    return prediction_data, abstention_fractions, mse_curve