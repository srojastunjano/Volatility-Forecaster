import numpy as np
import tensorflow as tf
import data_extraction
from transformer import PositionalEncoding
from evidential_head import EvidentialRegressionHead
from MCDropout import MCDropout
from transformer import TransformerEncoderBlock


def evaluate(model, X_test, y_test, scaler, K=100):
    """
    Evaluates the model across the entire test set to check coverage.
    """
    coverages = []
    errors = []
    
    for i in range(len(X_test)):
        res = generate_credal_set(model, X_test[i:i+1], scaler, K=K)
        
        actual = y_test[i] # future target
        low, high = res['ihdr_bounds']
        
        actual_unscaled = scaler.inverse_transform(actual.reshape(-1, 1))[0,0]
        coverages.append(low <= actual_unscaled <= high)
        errors.append(abs(actual_unscaled - res['mean_forecast']))
        
    print(f"Average Error (MSE): {np.mean(errors):.5f}")
    print(f"Empirical Coverage: {np.mean(coverages) * 100:.2f}%")

    return res

def generate_credal_set(model, X_input, target_scaler, K=100):
    """
    Runs K stochastic forward passes to generate a Credal Set of NIG distributions,
    disentangling Aleatoric and Epistemic uncertainty to form the IHDR.
    """

    if len(X_input.shape) == 2:
        X_input = np.expand_dims(X_input, axis=0)

    # OPTIMIZATION: tile the input to process all K passes in one GPU operation
    X_tiled = tf.tile(X_input, [K, 1, 1]) 
    preds = model(X_tiled, training=True)
    gamma, v, alpha, beta = tf.split(preds, num_or_size_splits=4, axis=-1)
    
    gamma = gamma.numpy()
    v = v.numpy()
    alpha = alpha.numpy()
    beta = beta.numpy()

    gamma_real = target_scaler.inverse_transform(gamma)
    
    # leatoric 
    aleatoric = beta / (alpha - 1.0)
    
    # epistemic 
    epistemic_nig = beta / (v * (alpha - 1.0))
    
    # Empirical Epistemic
    credal_epistemic_var = np.var(gamma_real)
    
    # constructing the IHDR 
    sigma_low = np.percentile(gamma_real, 5)
    sigma_high = np.percentile(gamma_real, 95)
    mean_forecast = np.mean(gamma_real)

    # print(f"\n--- IHDR Forecast Results (K={K} passes) ---")
    # print(f"Mean Forecast (Target):  {mean_forecast:.5f}")
    # print(f"IHDR Interval:           [{sigma_low:.5f}, {sigma_high:.5f}]")
    # print(f"Interval Width (Spread): {sigma_high - sigma_low:.5f}")
    # print(f"Mean Aleatoric Noise:    {np.mean(aleatoric):.5f}")
    # print(f"Empirical Epistemic Var: {credal_epistemic_var:.7f}")
    
    return {
        "mean_forecast": mean_forecast,
        "ihdr_bounds": (sigma_low, sigma_high),
        "credal_gamma": gamma_real,
        "aleatoric": aleatoric,
        "epistemic_nig": epistemic_nig,
        "credal_epistemic_var": credal_epistemic_var
    }

if __name__ == "__main__":

    K = 100
    model_path = "ibdl_volatility_1y_v4.keras"

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

    X_train, X_test, y_train, y_test, scaler = data_extraction.prepare_data("GME","max",10) # overfitting: (V1:22, V2:10, V3: 63), v4: 10 ,underfitting(v5:22) 
    sample_window = X_test[0:1] 
    
    # results = generate_credal_set(model, sample_window, scaler, K)


    results = evaluate(model, X_test, y_test, scaler, K)

    print(f"\n--- IHDR Forecast Results ({K} passes) ---")
    print(f"Mean Forecast (Target):  {results['mean_forecast']:.5f}")
    print(f"IHDR Interval:           [{results['ihdr_bounds'][0]:.5f}, {results['ihdr_bounds'][1]:.5f}]")
    print(f"Interval Width (Spread): {results['ihdr_bounds'][1] - results['ihdr_bounds'][0]:.5f}")
    print(f"Mean Aleatoric Noise:    {np.mean(results['aleatoric']):.5f}")
    print(f"Empirical Epistemic Var: {results['credal_epistemic_var']:.7f}")

