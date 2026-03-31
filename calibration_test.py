import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import architecture.data_extraction as data_extraction
from architecture.transformer import PositionalEncoding, TransformerEncoderBlock
from architecture.evidential_head import EvidentialRegressionHead
from architecture.MCDropout import MCDropout


# bootstrapping is a resampling method taken at random from randokm points of the original dataset to create smaller samples
# used to calc approximation of sample statistics by considering the original dataset as the population 
# simulates experiment 1000 times with my own data and 
def get_bootstrap_ci(data, n_iterations=1000, alpha=0.05):
    """
    Calculates the confidence interval for the mean of binary data using bootstrapping.
    data: list of 1s (hit) and 0s (miss)
    """
    stats = []
    n_size = len(data)
    for _ in range(n_iterations):
        # Resample with replacement
        sample = np.random.choice(data, size=n_size, replace=True)
        stats.append(np.mean(sample))
    
    # Calculate percentiles for the 95% Confidence Interval
    lower = np.percentile(stats, (alpha/2) * 100)
    upper = np.percentile(stats, (1 - alpha/2) * 100)
    return lower, upper

def run_calibration_test(ticker="TSLA", K=100, model_path="ibdl_volatility_1y_v8.keras"):

    print(f"Loading model and fetching data for {ticker}...")
    
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

    X_train, X_test, y_train, y_test, scaler = data_extraction.prepare_data(ticker, "max", 10)
    num_days = len(X_test)
    

    print(f"Running the Credal Engine: {K} stochastic passes for {num_days} days...")
    
    credal_matrix = np.zeros((num_days, K))
    y_true_real = np.zeros(num_days)
    
    for i in range(num_days):
        # taking single window, shaping it and tiling it K times for the GPU efficiency
        X_input = np.expand_dims(X_test[i], axis=0)
        X_tiled = tf.tile(X_input, [K, 1, 1]) 
        
        # keeps MC Dropout on
        preds = model(X_tiled, training=True)
        gamma, _, _, _ = tf.split(preds, num_or_size_splits=4, axis=-1)
        
        # inverse transform the predictions back to real volatility percentages
        gamma_real = scaler.inverse_transform(gamma.numpy())

        credal_matrix[i, :] = gamma_real.flatten()
        
        # unscaling the ground truth 
        actual_scaled = y_test[i:i+1]
        actual_unscaled = scaler.inverse_transform(actual_scaled.reshape(-1, 1))[0,0]
        y_true_real[i] = actual_unscaled


    # TEST
    confidence_levels = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.99]
    observed_coverages = []
    lower_cis = []
    upper_cis = []
    
    print("\n--- Calibration Results with 95% Bootstrap CIs ---")
    
    for conf in confidence_levels:
        # Calculate the IHDR bounds for this specific target
        margin = (1.0 - conf) / 2.0
        lower_pct = margin * 100
        upper_pct = (1.0 - margin) * 100
        
        lower_bounds = np.percentile(credal_matrix, lower_pct, axis=1)
        upper_bounds = np.percentile(credal_matrix, upper_pct, axis=1)
        
        # Check if truth is inside (Boolean array of hits/misses)
        is_covered = (y_true_real >= lower_bounds) & (y_true_real <= upper_bounds)
        
        # Calculate Point Estimate
        empirical_coverage = np.mean(is_covered)
        
        # RUN BOOTSTRAP: Build the histogram and find the bars
        lower_ci, upper_ci = get_bootstrap_ci(is_covered, n_iterations=1000)
        
        # Store for plotting
        observed_coverages.append(empirical_coverage)
        lower_cis.append(lower_ci)
        upper_cis.append(upper_ci)
        
        print(f"Target: {conf*100:4.1f}% | Observed: {empirical_coverage*100:5.2f}% | 95% CI: [{lower_ci*100:5.2f}%, {upper_ci*100:5.2f}%]")

    # METRICS
    expected_arr = np.array(confidence_levels)
    observed_arr = np.array(observed_coverages)
    
    # Calculate Uncertainty Calibration Error (UCE)
    uce = np.mean(np.abs(expected_arr - observed_arr))
    
    print("\n--- Final Metric ---")
    print(f"Uncertainty Calibration Error (UCE): {uce:.4f} ({uce*100:.2f}%)")


    # ERROR BAR PREPARATION
    # Matplotlib's yerr expects the *relative distance* from the point, not the absolute value
    lower_ci_arr = np.array(lower_cis)
    upper_ci_arr = np.array(upper_cis)
    
    yerr_lower = observed_arr - lower_ci_arr
    yerr_upper = upper_ci_arr - observed_arr
    asymmetric_error = [yerr_lower, yerr_upper]

    # GRAPH
    plt.figure(figsize=(9, 9))
    
    plt.plot([0, 1], [0, 1], linestyle='--', color='black', linewidth=2, label='Perfect Calibration')
    
    # The Data Points with Bootstrap Error Bars
    plt.errorbar(expected_arr, observed_arr, yerr=asymmetric_error, fmt='o-', markersize=8, 
                 color='blue', linewidth=2, capsize=5, capthick=2, elinewidth=2,
                 label=f'IBDL Output (UCE: {uce:.4f})')
    
    # The Danger Zones
    plt.fill_between([0, 1], [0, 1], 1, color='red', alpha=0.1, label='Underconfident (Intervals too wide)')
    plt.fill_between([0, 1], 0, [0, 1], color='green', alpha=0.1, label='Overconfident (Intervals too narrow)')
    
    plt.title(f"IHDR Calibration Curve for {ticker}", fontsize=16, fontweight='bold')
    plt.xlabel("Expected Confidence Level", fontsize=14)
    plt.ylabel("Observed Empirical Coverage", fontsize=14)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.legend(loc="upper left", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)
    
    plt.tight_layout()
    plt.show()

run_calibration_test(ticker="HSBC", K=100)