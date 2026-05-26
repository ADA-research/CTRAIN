"""
Analyze correlation between incomplete and complete verification results
using Spearman rank correlation on Pareto fronts discovered after 100 trials.
"""

import os
import json
import optuna
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import pickle
from util import get_pareto_front

def calculate_hypervolume(pareto_points, reference_point=None):
    """Calculate hypervolume for 2D Pareto points."""
    if reference_point is None:
        reference_point = (0, 0)
    
    if len(pareto_points) == 0:
        return 0.0
    
    if len(pareto_points) == 1:
        return pareto_points[0][0] * pareto_points[0][1]
    
    points = sorted(pareto_points, key=lambda p: p[0])
    hypervolume = 0.0
    
    for i, (x, y) in enumerate(points):
        if i == 0:
            width = x - reference_point[0]
        else:
            width = x - points[i-1][0]
        
        height = max(p[1] for p in points[i:]) - reference_point[1]
        hypervolume += width * height
    
    return hypervolume


def get_pareto_fronts_from_hpo(dataset, network, method, eps, trial_count=100):
    """
    Load and process Pareto fronts from HPO studies for 100 trials per seed.
    Returns the Pareto front (certified_acc, clean_acc) tuples.
    """
    mega_study = optuna.create_study(
        directions=["maximize", "maximize"], 
        study_name="moctrain"
    )
    
    for seed in range(3):
        try:
            study = optuna.load_study(
                study_name="moctrain",
                storage=f"sqlite:///../results/hpo/optuna_results/{dataset}_{network}_{method}_{eps}_{seed}_optuna_study.db",
            )
            mega_study.add_trials(study.trials[:trial_count])
        except Exception as e:
            print(f"Warning: Could not load study for {dataset}_{network}_{method}_{eps}_{seed}: {e}")
            continue
    
    # Extract completed trials
    trial_data = []
    for trial in mega_study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            trial_data.append({
                "Objective0": trial.values[0],  # Clean accuracy
                "Objective1": trial.values[1],  # Certified accuracy
            })
    
    df = pd.DataFrame(trial_data)
    
    if len(df) == 0:
        return []
    
    # Filter based on thresholds (from front_status.py)
    if eps == 2 / 255:
        df = df[df["Objective0"] > 0.6]
        df = df[df["Objective1"] > 0.4]
    elif eps == 8 / 255:
        df = df[df["Objective0"] > 0.4]
        df = df[df["Objective1"] > 0.25]
    
    if len(df) == 0:
        return []
    
    # Remove dominated solutions
    for id, row in df.iterrows():
        for id2, row2 in df.iterrows():
            if (row2["Objective0"] > row["Objective0"] and 
                row2["Objective1"] >= row["Objective1"]):
                df.drop(id, inplace=True)
                break
    
    # Calculate Pareto front
    objs = df[["Objective0", "Objective1"]].values
    mask = []
    for i, o in enumerate(objs):
        dominated = False
        for j, oj in enumerate(objs):
            if j == i:
                continue
            if (oj[0] >= o[0] and oj[1] >= o[1]) and (
                oj[0] > o[0] or oj[1] > o[1]
            ):
                dominated = True
                break
        mask.append(not dominated)
    
    mask = np.array(mask, dtype=bool)
    pareto = df[mask]
    
    return list(zip(pareto["Objective1"].values, pareto["Objective0"].values))


def get_verification_accuracy(method, dataset, network, eps, incomplete=True):
    """
    Get all verification results (not just Pareto front) for the configuration.
    Returns a list of (certified_accuracy, clean_accuracy) tuples.
    """
    if incomplete:
        # Incomplete verification will be computed from HPO fronts
        return None
    else:
        # Complete verification results - get ALL results, not just Pareto front
        with open('../results/verification/summary_results.json', 'r') as f:
            results = json.load(f)
        
        # Filter to this specific configuration (without Pareto filtering)
        matching = [r for r in results 
                   if r['dataset'] == dataset 
                   and r['architecture'] == network
                   and abs(float(r['eps']) - float(eps)) < 1e-6
                   and r['cert_train_method'] == method]
        
        return [(r['certified_accuracy'], r['clean_classification_accuracy']) for r in matching]


def compare_incomplete_vs_complete():
    """
    Compare incomplete (HPO objective) vs complete verification rankings.
    """
    correlation_results = []
    
    for method in ["mtl_ibp", "sabr", "shi", "crown_ibp_nofusion"]:
        for dataset in ["cifar10", "tinyimagenet"]:
            if dataset == "cifar10" and method in ["crown_ibp_nofusion"]:
                method_complete = method_incomplete = "crown_ibp_nofusion"
            elif dataset == "tinyimagenet" and method in ["crown_ibp_nofusion"]:
                method_complete = "crown_ibp_nofusion"
                method_incomplete = "crown_ibp"
            else:
                method_complete = method_incomplete = method
            for network in ["cnn7"]:
                for eps in [2/255, 8/255] if dataset == "cifar10" else [1/255]:
                    try:
                        # Get incomplete verification (from HPO after 100 trials)
                        incomplete_front = get_pareto_fronts_from_hpo(dataset, network, method_incomplete, eps, trial_count=100)
                        
                        if len(incomplete_front) < 2:
                            print(f"Skipping {dataset}_{network}_{method}_{eps}: not enough incomplete verification points")
                            continue
                        
                        # Get complete verification results
                        complete_front = get_verification_accuracy(method_complete, dataset, network, eps, incomplete=False)
                        
                        if not complete_front or len(complete_front) < 2:
                            print(f"Skipping {dataset}_{network}_{method}_{eps}: no complete verification results")
                            continue
                        
                        # Match configurations based on natural accuracy (clean_acc)
                        # incomplete_front and complete_front are lists of (cert_acc, clean_acc) tuples

                        # Extract clean accuracies for matching
                        incomplete_clean = [p[1] * 100 for p in incomplete_front]
                        complete_clean = [p[1] for p in complete_front]
                        
                        incomplete_cert = [p[0] * 100 for p in incomplete_front]
                        complete_cert = [p[0] for p in complete_front]

                        
                        if len(incomplete_cert) > 0 and len(complete_cert) > 0:
                            # Match incomplete configurations to complete configurations based on clean accuracy
                            # For each incomplete config, find the closest complete config by clean accuracy
                            matched_incomplete_cert = []
                            matched_complete_cert = []
                            
                            for i_clean, i_cert in zip(incomplete_clean, incomplete_cert):
                                # Find closest complete config by clean accuracy
                                closest_idx = min(range(len(complete_clean)), 
                                                 key=lambda j: abs(complete_clean[j] - i_clean))
                                
                                # Only match if within reasonable distance (e.g., 5% clean acc)
                                if abs(complete_clean[closest_idx] - i_clean) < 0.01:
                                    matched_incomplete_cert.append(i_cert)
                                    matched_complete_cert.append(complete_cert[closest_idx])
                            
                            if len(matched_incomplete_cert) < 2:
                                print(f"Skipping {dataset}_{network}_{method}_{eps}: not enough matched configurations")
                                continue
                            
                            # Sort by matched incomplete certified accuracy to maintain order
                            sorted_pairs = sorted(zip(matched_incomplete_cert, matched_complete_cert), 
                                                key=lambda x: x[0], reverse=True)
                            sorted_incomplete = [p[0] for p in sorted_pairs]
                            sorted_complete = [p[1] for p in sorted_pairs]
                            
                            # Spearman correlation
                            corr, pval = spearmanr(sorted_incomplete, sorted_complete)
                            
                            correlation_results.append({
                                'Dataset': dataset,
                                'Network': network,
                                'Method': method,
                                'Epsilon': eps,
                                'Incomplete_Points': len(incomplete_cert),
                                'Complete_Points': len(complete_cert),
                                'Correlation_Points': len(matched_incomplete_cert),
                                'Spearman_Rho': corr,
                                'P_Value': pval,
                            })
                            
                            print(f"{dataset}_{network}_{method}_{eps}: ρ={corr:.4f}, p={pval:.4f}, n={len(matched_incomplete_cert)}")
                    
                    except Exception as e:
                        print(f"Error processing {dataset}_{network}_{method}_{eps}: {e}")
    
    return correlation_results


def generate_correlation_table(correlation_results):
    """Generate TeX table for correlation results."""
    if not correlation_results:
        print("No correlation results to tabulate")
        return
    
    # Create DataFrame
    df = pd.DataFrame(correlation_results)
    
    # Define method order: SHI (IBP) - CROWN_IBP - SABR - MTL_IBP
    method_order = {'shi': 0, 'crown_ibp_nofusion': 1, 'crown_ibp': 2, 'sabr': 3, 'mtl_ibp': 4}
    
    # Define epsilon order for CIFAR-10
    eps_order_cifar = {2/255: 0, 8/255: 1}
    
    # Sort by dataset, epsilon, then method
    def sort_key(row):
        dataset_priority = 0 if row['Dataset'] == 'cifar10' else 1
        if row['Dataset'] == 'cifar10':
            eps_priority = eps_order_cifar.get(row['Epsilon'], 999)
        else:
            eps_priority = 0  # Only one epsilon for tinyimagenet
        method_priority = method_order.get(row['Method'], 999)
        return (dataset_priority, eps_priority, method_priority)
    
    df_sorted = df.copy()
    df_sorted['sort_key'] = df_sorted.apply(sort_key, axis=1)
    df_sorted = df_sorted.sort_values('sort_key').drop('sort_key', axis=1)
    
    # Generate TeX table
    tex_content = "\\begin{table}[htbp]\n"
    tex_content += "\\centering\n"
    tex_content += "\\caption{Spearman Rank Correlation: Incomplete vs Complete Verification}\n"
    tex_content += "\\label{tab:verification_correlation}\n"
    tex_content += "\\begin{tabular}{llll|rrr}\n"
    tex_content += "\\toprule\n"
    tex_content += "Dataset & Network & Method & $\\epsilon$ & $\\rho$ & $p$-value & $n$ \\\\\n"
    tex_content += "\\midrule\n"
    
    for _, row in df_sorted.iterrows():
        dataset_str = "CIFAR-10" if row['Dataset'] == "cifar10" else "Tiny-ImageNet"
        
        # Format epsilon
        if row['Epsilon'] == 2/255:
            eps_str = "2/255"
        elif row['Epsilon'] == 8/255:
            eps_str = "8/255"
        elif row['Epsilon'] == 1/255:
            eps_str = "1/255"
        else:
            eps_str = f"{row['Epsilon']:.4f}"
        
        n = int(row['Correlation_Points'])
        rho = row['Spearman_Rho']
        pval = row['P_Value']
        if np.isnan(rho):
            rho = 0.0
        if np.isnan(pval):
            pval = 1.0
        
        tex_content += f"{dataset_str} & {row['Network'].upper()} & {row['Method'].upper()} & {eps_str} & "
        tex_content += f"{rho:.4f} & {pval:.4e} & {n} \\\\\n"
    
    tex_content += "\\bottomrule\n"
    tex_content += "\\end{tabular}\n"
    tex_content += "\\end{table}\n"
    
    # Save table
    os.makedirs("tables", exist_ok=True)
    output_path = "tables/verification_correlation.tex"
    with open(output_path, 'w') as f:
        f.write(tex_content)
    
    print(f"\nGenerated TeX table: {output_path}")
    
    # Also save as CSV for reference
    csv_path = "tables/verification_correlation.csv"
    df.to_csv(csv_path, index=False)
    print(f"Generated CSV table: {csv_path}")


if __name__ == "__main__":
    print("Analyzing correlation between incomplete and complete verification...")
    correlation_results = compare_incomplete_vs_complete()
    if correlation_results:
        rhos = np.array([r.get('Spearman_Rho', np.nan) for r in correlation_results], dtype=float)
        valid_count = int(np.sum(~np.isnan(rhos)))
        if valid_count == 0:
            print("\nAverage Spearman rho over all benchmarks: nan (no valid rho values)")
        else:
            avg_rho = float(np.nanmean(rhos))
            print(f"\nAverage Spearman rho over all benchmarks (n={valid_count}): {avg_rho:.4f}")
    else:
        print("\nNo correlation results to compute average rho")
    generate_correlation_table(correlation_results)
    print("\nDone!")
