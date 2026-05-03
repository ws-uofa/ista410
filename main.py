import json
import numpy as np
import scipy.stats
import pandas as pd
import matplotlib.pyplot as plt
from cmdstanpy import CmdStanModel
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

def run_beir_inference(db_name, b_data, model_infer):
    print(f"[{db_name}] start inferencing...")
    fit_beir = model_infer.sample(
        data=b_data, 
        chains=2, 
        parallel_chains=2,
        threads_per_chain=6,
        iter_warmup=300, 
        iter_sampling=300, 
        show_progress=False
    )
    
    theta_b_samples = fit_beir.stan_variable(var='theta')
    b_ci_widths = np.percentile(theta_b_samples, 97.5, axis=0) - np.percentile(theta_b_samples, 2.5, axis=0)
    b_theta_mean = np.mean(theta_b_samples, axis=0)
    b_entropy = scipy.stats.entropy(b_theta_mean, axis=1, base=2)
    
    return {
        "Dataset": db_name,
        "Mean_95_CI_Width": np.mean(b_ci_widths),
        "Mean_Entropy": np.mean(b_entropy)
    }

if __name__ == '__main__':
    # ---------------- Configuration & Setup ----------------
    os.makedirs("report_figures", exist_ok=True)

    # Compile Stan Models
    print("[1/5] Compiling Stan models with STAN_THREADS enabled...")
    model_train = CmdStanModel(
        stan_file='lda_train.stan', 
        cpp_options={'STAN_THREADS': 'true'}
    )
    model_infer = CmdStanModel(
        stan_file='lda_infer.stan',
        cpp_options={'STAN_THREADS': 'true'}
    )

    # ---------------- MS MARCO (In-Domain) Analysis ----------------
    print("\n[2/5] Starting MCMC sampling on MS MARCO (In-Domain)...")
    with open("data_msmarco.json", "r") as f:
        ms_data = json.load(f)

    fit_ms = model_train.sample(
        data=ms_data, 
        chains=4,               
        parallel_chains=4,      
        threads_per_chain=11,
        inits=0.1,
        max_treedepth=8,
        iter_warmup=300,
        iter_sampling=500
    )

    # Extract parameters
    phi_samples = fit_ms.stan_variable(var='phi') 
    phi_mean = np.mean(phi_samples, axis=0).tolist() # [K, V]
    theta_ms_samples = fit_ms.stan_variable(var='theta') # [samples, M, K]

    # Calculate uncertainty for MS MARCO
    ms_ci_widths = np.percentile(theta_ms_samples, 97.5, axis=0) - np.percentile(theta_ms_samples, 2.5, axis=0)
    ms_theta_mean = np.mean(theta_ms_samples, axis=0)
    ms_entropy = scipy.stats.entropy(ms_theta_mean, axis=1, base=2)

    ms_mean_ci = np.mean(ms_ci_widths)
    ms_mean_ent = np.mean(ms_entropy)
    print(f"[MS MARCO] Mean 95% Credible Interval Width: {ms_mean_ci:.4f}")
    print(f"[MS MARCO] Mean Shannon Entropy: {ms_mean_ent:.4f}")

    # ---------------- Model Diagnostics ----------------
    print("\n[3/5] Performing model diagnostics and generating plots...")
    ms_summary = fit_ms.summary()
    max_rhat = ms_summary['R_hat'].max()
    mean_rhat = ms_summary['R_hat'].mean()
    print(f"[Diagnostics] Max R-hat: {max_rhat:.4f}")
    print(f"[Diagnostics] Mean R-hat: {mean_rhat:.4f}")

    plt.figure(figsize=(10, 4))
    draws = fit_ms.draws_pd()
    theta_cols = [col for col in draws.columns if col.startswith('theta[1,')]
    for i, col in enumerate(theta_cols[:3]): 
        plt.plot(draws[col].values, alpha=0.7, label=f'Topic {i+1}')
    plt.title("Trace Plot for Document 1 Topic Proportions (MS MARCO)")
    plt.xlabel("Iteration")
    plt.ylabel("Theta Probability")
    plt.legend()
    plt.tight_layout()
    plt.savefig("report_figures/trace_plot.png")
    plt.close()

    V = ms_data["V"]
    M = ms_data["M"]
    actual_word_counts = np.zeros(V)
    for d, w, c in zip(ms_data['d'], ms_data['w'], ms_data['count']):
        actual_word_counts[w - 1] += c

    doc_lengths = np.zeros(M)
    for d, c in zip(ms_data['d'], ms_data['count']):
        doc_lengths[d - 1] += c

    expected_word_counts = np.zeros(V)
    doc_word_prob = np.dot(ms_theta_mean, np.array(phi_mean))
    for m in range(M):
        expected_word_counts += doc_lengths[m] * doc_word_prob[m]

    plt.figure(figsize=(6, 6))
    plt.scatter(actual_word_counts, expected_word_counts, alpha=0.5)
    plt.plot([0, max(actual_word_counts)], [0, max(actual_word_counts)], 'r--', label='Perfect Fit')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Observed Word Frequency (Log scale)")
    plt.ylabel("Expected Word Frequency (Log scale)")
    plt.title("Posterior Predictive Check (PPC)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("report_figures/ppc_plot.png")
    plt.close()

    # ---------------- BEIR (Out-of-Domain) Inference ----------------
    print("\n[4/5] Starting concurrent inference on BEIR datasets (Out-of-Domain)...")
    with open("data_beir.json", "r") as f:
        beir_data_dict = json.load(f)

    for db_name, b_data in beir_data_dict.items():
        b_data["K"] = ms_data["K"]
        b_data["V"] = ms_data["V"]
        b_data["alpha"] = ms_data["alpha"]
        b_data["phi"] = phi_mean  

    results_list = []
    max_workers = 4 
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_beir_inference, db, data, model_infer): db for db, data in beir_data_dict.items()}
        for future in as_completed(futures):
            db_name = futures[future]
            try:
                res = future.result()
                results_list.append(res)
                print(f"[{db_name}] Finished:")
                print(f"  └─ Mean 95% CI Width: {res['Mean_95_CI_Width']:.4f}")
                print(f"  └─ Mean Shannon Entropy: {res['Mean_Entropy']:.4f}")
            except Exception as exc:
                print(f"[{db_name}] generated an exception: {exc}")

    results = {
        "Dataset": ["MS MARCO"] + [r["Dataset"] for r in results_list],
        "Mean_95_CI_Width": [ms_mean_ci] + [r["Mean_95_CI_Width"] for r in results_list],
        "Mean_Entropy": [ms_mean_ent] + [r["Mean_Entropy"] for r in results_list]
    }

    # ---------------- Final Visualizations & Summary ----------------
    print("\n[5/5] Generating cross-domain comparison charts...")
    df_results = pd.DataFrame(results)

    plt.figure(figsize=(10, 5))
    colors = ['skyblue' if ds == 'MS MARCO' else 'salmon' for ds in df_results['Dataset']]
    bars = plt.bar(df_results['Dataset'], df_results['Mean_95_CI_Width'], color=colors)
    plt.axhline(y=ms_mean_ci, color='gray', linestyle='--') 
    plt.title("Model Uncertainty: 95% Credible Interval Width Across Datasets")
    plt.ylabel("Mean 95% CI Width")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("report_figures/ci_width_comparison.png")
    plt.close()

    plt.figure(figsize=(10, 5))
    bars = plt.bar(df_results['Dataset'], df_results['Mean_Entropy'], color=colors)
    plt.axhline(y=ms_mean_ent, color='gray', linestyle='--')
    plt.title("Model Uncertainty: Shannon Entropy Across Datasets")
    plt.ylabel("Mean Shannon Entropy")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("report_figures/entropy_comparison.png")
    plt.close()