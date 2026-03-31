import numpy as np
from inference import generate_credal_set
from transformer import PositionalEncoding
from evidential_head import EvidentialRegressionHead
from MCDropout import MCDropout
from transformer import TransformerEncoderBlock
import tensorflow as tf
from data_extraction import prepare_data

def perturb_with_noise(X_test, noise_scale=0.01, clip_min=1e-6, seed=None):
    """
    The Perturbation Engine (Stressor 1: Noise)
    Injects synthetic Gaussian noise into the test set to measure Aleatoric uncertainty scaling.
    
    Parameters:
    - X_test: 3D numpy array [samples, time_steps, features]. This should be your scaled input.
    - noise_scale: The standard deviation (sigma) of the Gaussian noise.
    - clip_min: The minimum allowable value (prevents negative variance).
    - seed: For reproducible dissertation results.
    
    Returns:
    - X_noisy: The perturbed 3D tensor.
    """

    if seed is not None:
        np.random.seed(seed)
        
    # Generate Gaussian noise matching the exact shape of X_test
    noise = np.random.normal(loc=0.0, scale=noise_scale, size=X_test.shape)
    
    # Add the noise to the original signal
    X_noisy = X_test + noise
    
    # Domain constraint: Realized Variance cannot be negative. 
    if clip_min is not None:
        X_noisy = np.maximum(X_noisy, clip_min)
        
    return X_noisy


def run_noise_sensitivity_test(model, X_test, scaler, K=100):
    """
    Tests the model across progressively noisier environments to track
    the Disentanglement of Aleatoric vs. Epistemic uncertainty.
    """
    # Define the noise levels to test (0.0 is the baseline clean data)
    noise_levels = [0.0, 0.01, 0.05, 0.10, 0.20, 0.50]
    
    experiment_results = []
    
    print("--- Starting Chaos vs. Ignorance Test (Noise Injection) ---")
    
    for scale in noise_levels:
        print(f"Testing Noise Level (Sigma): {scale}")
        
        X_noisy = perturb_with_noise(X_test, noise_scale=scale, seed=42)
        
        aleatoric_list = []
        epistemic_list = []
        credal_spread_list = []
        
        # Run Inference over the noisy test set
        for i in range(len(X_noisy)):
            # Using your existing generate_credal_set function
            res = generate_credal_set(model, X_noisy[i:i+1], scaler, K=K)
            
            # Extract the raw uncertainty metrics
            aleatoric_list.append(np.mean(res['aleatoric']))
            epistemic_list.append(np.mean(res['epistemic_nig']))
            credal_spread_list.append(res['credal_epistemic_var'])
            
        # Aggregate the results for this noise level
        experiment_results.append({
            'Noise_Scale': scale,
            'Mean_Aleatoric': np.mean(aleatoric_list),
            'Mean_Epistemic': np.mean(epistemic_list),
            'Mean_Credal_Spread': np.mean(credal_spread_list)
        })
        
    
    return experiment_results

K = 100
model_path = "ibdl_volatility_1y_v8.keras"

model = tf.keras.models.load_model(
model_path, 
custom_objects={
    "PositionalEncoding": PositionalEncoding, 
    "TransformerEncoderBlock": TransformerEncoderBlock,
    "EvidentialRegressionHead": EvidentialRegressionHead,
    "MCDropout": MCDropout
},
safe_mode=False,
)   

X_train, X_test, y_train, y_test, scaler = prepare_data("TSLA","max",10)
run_noise_sensitivity_test(model, X_test, scaler, 100)

