import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from architecture.evidential_head import EvidentialRegressionHead
from architecture.MCDropout import MCDropout
from architecture.transformer import TransformerEncoderBlock, PositionalEncoding
import tensorflow as tf
from architecture.data_extraction import prepare_data
from aleatoric_experiemnt import run_noise_sensitivity_test
from epistemic_experiment import run_ood_domain_test, evaluate_domain



def run_disentanglement_analysis(baseline_results, noise_results, ood_results):
    """
    The Disentanglement Comparator
    Aggregates Baseline, Noisy, and OOD results to prove the "Chaos vs. Ignorance" hypothesis.
    
    Parameters:
    - baseline_results: Dict containing Mean_Aleatoric and Mean_Epistemic for clean TSLA data.
    - noise_results: Dict containing metrics for the Noisy TSLA data (e.g., at noise_scale=0.10).
    - ood_results: Dict containing metrics for the Alien/Crypto data.
    """
    print("\n--- Aggregating Disentanglement Results ---")
    
    data = {
        'Scenario': ['Baseline (Clean)', 'Stressor 1 (Noise)', 'Stressor 2 (OOD)'],
        'Aleatoric (AU)': [
            baseline_results['Mean_Aleatoric'],
            noise_results['Mean_Aleatoric'],
            ood_results['Mean_Aleatoric']
        ],
        'Epistemic (EU)': [
            baseline_results['Mean_Epistemic'],
            noise_results['Mean_Epistemic'],
            ood_results['Mean_Epistemic']
        ]
    }
    
    df_results = pd.DataFrame(data)
    
    # Calculate the Disentanglement Ratio (EU / AU)
    # A massive spike in this ratio indicates the model recognizes its own ignorance
    df_results['Disentanglement Ratio (EU/AU)'] = df_results['Epistemic (EU)'] / df_results['Aleatoric (AU)']
    
    print("\n--- Final Dissertation Table ---")
    print(df_results.to_string(index=False))
    
    plot_disentanglement_chart(df_results)
    
    return df_results

def plot_disentanglement_chart(df):
    """
    Generates a grouped bar chart comparing AU and EU across the three scenarios.
    """
    x = np.arange(len(df['Scenario']))
    width = 0.35 

    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars_au = ax.bar(x - width/2, df['Aleatoric (AU)'], width, label='Aleatoric (Market Noise)', color='coral')
    bars_eu = ax.bar(x + width/2, df['Epistemic (EU)'], width, label='Epistemic (Model Ignorance)', color='steelblue')

    ax.set_ylabel('Uncertainty Magnitude')
    ax.set_title('Chaos vs. Ignorance: Uncertainty Disentanglement in IBDL')
    ax.set_xticks(x)
    ax.set_xticklabels(df['Scenario'])
    ax.legend()

    # ad data labels on top of the bars for clarity
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.5f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    autolabel(bars_au)
    autolabel(bars_eu)

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def analyze_and_plot_disentanglement(baseline_res, noise_res, ood_res, save_path="disentanglement_results.png"):
    """
    Aggregates results and generates a publication-ready graph proving 
    the "Chaos vs. Ignorance" hypothesis.
    """
    print("\n--- Aggregating Disentanglement Results ---")
    
    labels = ['Baseline (Clean TSLA)', 'Stressor 1 (Noisy TSLA)', 'Stressor 2 (Alien BTC)']
    
    au_values = [
        baseline_res['Mean_Aleatoric'],
        noise_res['Mean_Aleatoric'],
        ood_res['Mean_Aleatoric']
    ]
    
    eu_values = [
        baseline_res['Mean_Epistemic'],
        noise_res['Mean_Epistemic'],
        ood_res['Mean_Epistemic']
    ]
    
    df = pd.DataFrame({
        'Scenario': labels,
        'Aleatoric (AU - Noise)': au_values,
        'Epistemic (EU - Ignorance)': eu_values
    })

    df['Disentanglement Ratio (EU/AU)'] = df['Epistemic (EU - Ignorance)'] / df['Aleatoric (AU - Noise)']
    
    print("\n--- Final Dissertation Table ---")
    print(df.to_string(index=False))
    

    # Generate Publication-Quality Graph
    x = np.arange(len(labels))  # Label locations
    width = 0.35                # Width of the bars
    
    # Set high DPI for print-quality resolution
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300) 
    
    bars_au = ax.bar(x - width/2, au_values, width, 
                     label='Aleatoric (Market Noise)', color='#ff9999', edgecolor='black')
    bars_eu = ax.bar(x + width/2, eu_values, width, 
                     label='Epistemic (Model Ignorance)', color='#66b3ff', edgecolor='black')
    
    ax.set_ylabel('Uncertainty Magnitude', fontsize=12, fontweight='bold')
    ax.set_title('Chaos vs. Ignorance: Uncertainty Disentanglement in IBDL', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(fontsize=11, loc='upper left')

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            # Formatting to 5 decimal places to capture small variance shifts
            ax.annotate(f'{height:.5f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 4),  # 4 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)
    
    autolabel(bars_au)
    autolabel(bars_eu)
    
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    
    fig.tight_layout()
    
    plt.savefig(save_path, bbox_inches='tight')
    print(f"\n Graph successfully saved to: {save_path}")
    
    plt.show()
    
    return df

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

X_train, X_test_tsla, y_train, y_test, tsla_scaler = prepare_data("TSLA","max",10)
X_train, X_test_btc, y_train, y_test, btc_scaler = prepare_data("BTC","max",10)

baseline_metrics = evaluate_domain(model, tsla_scaler, X_test_tsla, "Baseline", K) 

# Get Stressor 1 (Noise)
noisy_results_list = run_noise_sensitivity_test(model, X_test_tsla, tsla_scaler, K=100)
noise_metrics = next(item for item in noisy_results_list if item["Noise_Scale"] == 0.10)

# Get Stressor 2 (OOD)
ood_metrics = evaluate_domain(model, btc_scaler, X_test_btc, "Alien Data", K) 

final_df = run_disentanglement_analysis(baseline_metrics, noise_metrics, ood_metrics)
