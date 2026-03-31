import numpy as np
import tensorflow as tf
from transformer import PositionalEncoding
import data_extraction as data_extraction
from evidential_head import EvidentialRegressionHead
from MCDropout import MCDropout
from transformer import TransformerEncoderBlock
from evidential_loss import EvidentialLoss
from tensorflow.keras import mixed_precision
from panel_data_extraction import build_global_tensor
from scipy import stats
import numpy as np
import scipy.stats as stats
import tensorflow as tf
import os
import sys
import tf_keras as keras

os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=-1'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.modules['tf_keras'] = keras
sys.modules['tf_keras.src'] = keras.src
sys.modules['tf_keras.src.engine'] = keras.src.engine
sys.modules['tf_keras.src.engine.functional'] = keras.src.engine.functional

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"Found {len(gpus)} GPU(s): {gpus}")
    # This specifically checks for the Apple Metal plug-in
    details = tf.config.experimental.get_device_details(gpus[0])
    print(f"Device Name: {details.get('device_name', 'Unknown')}")
else:
    print("No GPU found. Running on CPU.")

print(f"Current Global Policy: {mixed_precision.global_policy().name}")

# To force it back to standard 32-bit:
mixed_precision.set_global_policy('float32')
def evaluate(model, X_temp_test, X_cont_test, y_test, scaler, K=100, confidence_level=0.90):
    """
    Evaluates the model across the entire test set using vectorized matrix operations.
    """
    print(f"\n--- Running Vectorized Evaluation (Target Confidence: {confidence_level*100}%) ---")
    
    # 1. Generate the Credal Set for the ENTIRE test set at once
    res = generate_credal_set(model, X_temp_test, X_cont_test, scaler, K=K, confidence_level=confidence_level)
    
    forecast = res['mean_forecast']
    low, high = res['ihdr_bounds']
    
    # 2. Unscale the actual targets (Vectorized)
    actual_unscaled = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    # 3. Calculate Metrics (Vectorized boolean logic and array math)
    coverages = (actual_unscaled >= low) & (actual_unscaled <= high)
    abs_errors = np.abs(actual_unscaled - forecast)
    sq_errors = (actual_unscaled - forecast)**2

    mae = np.mean(abs_errors)
    mse = np.mean(sq_errors)
    rmse = np.sqrt(mse)
    coverage_pct = np.mean(coverages) * 100
        
    print("\n--- Final Test Set Evaluation ---")
    print(f"Empirical Coverage: {coverage_pct:.2f}% (Expected: {confidence_level*100}%)")
    print(f"Mean Absolute Error (MAE): {mae:.5f}")
    print(f"Mean Squared Error (MSE):  {mse:.5f}")
    print(f"Root Mean Sq Error (RMSE): {rmse:.5f}")

    # 'res' now contains arrays of size (N,) representing the full test set
    return res

def generate_credal_set(model, X_temp, X_cont, target_scaler, K=100, confidence_level=0.90):
    """
    Runs K stochastic forward passes to generate a Credal Set of NIG distributions.
    Properly applies the Law of Total Variance to construct the IHDR bounds.
    """
    print(f"Running {K} stochastic forward passes. This may take a moment...")
    
    # Instead of tf.tile (which can cause OOM on large datasets), we run predict K times.
    # Because your custom MCDropout forces training=True, predict() naturally creates the Credal Set.
    preds_list = []
    for _ in range(K):
        # Batch size prevents OOM, verbose=0 keeps the console clean
        preds = model.predict([X_temp, X_cont], batch_size=4096, verbose=0)
        preds_list.append(preds)
        
    # Stack along a new K dimension: Shape becomes (K, N_samples, 4)
    preds = np.stack(preds_list, axis=0)
    
    # Split the predictions along the last axis and squeeze out the single dimension
    gamma = preds[..., 0]  # Shape: (K, N)
    v = preds[..., 1]      
    alpha = preds[..., 2]  
    beta = preds[..., 3]   

    # 1. Calculate Uncertainties in the SCALED space first
    aleatoric_scaled = beta / (alpha - 1.0)
    epistemic_nig_scaled = beta / (v * (alpha - 1.0))
    
    # Total Evidential Variance for each pass (Aleatoric + Epistemic)
    evidential_var_scaled = aleatoric_scaled + epistemic_nig_scaled

    # 2. Extract the Unscaling Factor
    scale_factor = target_scaler.inverse_transform([[1.0]])[0,0] - target_scaler.inverse_transform([[0.0]])[0,0]
    
    # 3. Apply the Law of Total Variance (Averaging over the K dimension, axis=0)
    # Total Var = Mean(Evidential Var) + Var(Mean Predictions)
    mean_evidential_var = np.mean(evidential_var_scaled, axis=0) # Shape: (N,)
    credal_var_gamma = np.var(gamma, axis=0)                     # Shape: (N,)
    
    total_var_scaled = mean_evidential_var + credal_var_gamma
    
    # Convert total scaled variance to unscaled standard deviation
    total_std_unscaled = np.sqrt(total_var_scaled) * scale_factor

    # 4. Unscale the Prediction
    mean_gamma_scaled = np.mean(gamma, axis=0)
    mean_forecast = target_scaler.inverse_transform(mean_gamma_scaled.reshape(-1, 1)).flatten()

    # 5. Construct the mathematically correct IHDR
    # For a Normal/Student-T approximation:
    z_score = stats.norm.ppf(1.0 - (1.0 - confidence_level) / 2.0) 
    
    sigma_low = mean_forecast - (z_score * total_std_unscaled)
    sigma_high = mean_forecast + (z_score * total_std_unscaled)

    # Convert scaled tracking metrics to unscaled for the output dictionary
    aleatoric_unscaled = aleatoric_scaled * (scale_factor**2)
    epistemic_nig_unscaled = epistemic_nig_scaled * (scale_factor**2)
    credal_epistemic_var_unscaled = credal_var_gamma * (scale_factor**2)
    
    return {
        "mean_forecast": mean_forecast,                     # Shape: (N,)
        "ihdr_bounds": (sigma_low, sigma_high),             # Tuple of Shapes: (N,), (N,)
        "total_std": total_std_unscaled,                    # Shape: (N,)
        "aleatoric": np.mean(aleatoric_unscaled, axis=0),   # Averaged over K passes
        "epistemic_nig": np.mean(epistemic_nig_unscaled, axis=0),
        "credal_epistemic_var": credal_epistemic_var_unscaled
    }

if __name__ == "__main__":
    K = 100
    model_path = "global_hybrid_ibdl_v1.keras"

    model = tf.keras.models.load_model(
        model_path, 
        custom_objects={
            "PositionalEncoding": PositionalEncoding, 
            "TransformerEncoderBlock": TransformerEncoderBlock,
            "EvidentialRegressionHead": EvidentialRegressionHead,
            "MCDropout": MCDropout,
            "EvidentialLoss": EvidentialLoss 
        },
        safe_mode=False,
        compile=False
    )   

    model.jit_compile = False

    # Assuming you have separated your test data into the temporal, contextual, and target arrays
    # X_temp_test shape: (N, 21, 1)
    # X_cont_test shape: (N, 2)
    # y_test shape: (N,)

    X_temporal, X_contextual, y_target, scaler = build_global_tensor(max_stocks=500, seq_len=21)
    
    results = evaluate(model, X_temporal, X_contextual, y_target, scaler, K=K)

    # To view the results for a specific day (e.g., the last day in the test set):
    idx = -1
    print(f"\n--- IHDR Forecast Results for Final Day ({K} passes) ---")
    print(f"Mean Forecast (Target):  {results['mean_forecast'][idx]:.5f}")
    print(f"IHDR Interval:           [{results['ihdr_bounds'][0][idx]:.5f}, {results['ihdr_bounds'][1][idx]:.5f}]")
    print(f"Interval Width (Spread): {results['ihdr_bounds'][1][idx] - results['ihdr_bounds'][0][idx]:.5f}")
    print(f"Mean Aleatoric Noise:    {results['aleatoric'][idx]:.5f}")
    print(f"Empirical Epistemic Var: {results['credal_epistemic_var'][idx]:.7f}")