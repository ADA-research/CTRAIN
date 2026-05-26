import os
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import seaborn as sns
from ConfigSpace.hyperparameters import NumericalHyperparameter
from collections import defaultdict
import numpy as np
import pickle
import optuna
import pandas as pd
from scipy.spatial import ConvexHull

renames = {
    "Objective0": "Clean Accuracy",
    "Objective1": "Cert. Accuracy",
    "train_eps_factor": "Train Eps. Factor",
    "warm_up_epochs": "Warmup Epochs",
    "l1_reg_weight": "l1 Reg. Weight",
    "lr_decay_factor": "LR Decay Factor",
    "mtl_ibp:mtl_ibp_alpha": "MTL IBP α",
    "learning_rate": "Learning Rate",
    "mtl_ibp:pgd_alpha": "PGD α",
    "mtl_ibp:pgd_steps": "PGD Steps",
    "sabr:subselection_ratio": "Subsel. Ratio",
    "sabr:pgd_steps": "PGD Steps",
    "sabr:pgd_eps_factor": "PGD Eps. Factor",
    "sabr:pgd_alpha": "PGD α",
    "shi:end_kappa": "End κ",
    "shi:start_kappa": "Start κ",
    "ramp_up_epochs": "Ramp-up Epochs",
    "lr_decay_epoch_1": "LR Decay Epoch 1",
    "lr_decay_epoch_2": "LR Decay Epoch 2",
    "crown_ibp:end_kappa": "End κ",
    "crown_ibp:start_kappa": "Start κ",
    "crown_ibp:end_beta": "End β",
    "crown_ibp:start_beta": "Start β",
    "shi_reg_weight": "Shi Reg. Weight",
    "mtl_ibp:mtl_ibp_eps_factor": "PGD Eps. Factor",
    "optimizer_func": "Optimiser",
}

def calculate_hypervolume(pareto_points, reference_point=None):
    """
    Calculate hypervolume indicator for a set of 2D Pareto points.
    Uses the WFG algorithm for 2D case (which is simple polygon area).
    
    Args:
        pareto_points: List of (x, y) tuples or array of shape (n, 2)
        reference_point: Reference point for hypervolume (default: (0, 0))
    
    Returns:
        Hypervolume value
    """
    if reference_point is None:
        reference_point = (0, 0)
    
    if len(pareto_points) == 0:
        return 0.0
    
    if len(pareto_points) == 1:
        # Single point: rectangle from origin to point
        return pareto_points[0][0] * pareto_points[0][1]
    
    # Sort points by x coordinate
    points = sorted(pareto_points, key=lambda p: p[0])
    
    # Calculate hypervolume as sum of rectangles
    hypervolume = 0.0
    prev_y = reference_point[1]
    
    for i, (x, y) in enumerate(points):
        # Width of rectangle
        if i == 0:
            width = x - reference_point[0]
        else:
            width = x - points[i-1][0]
        
        # Height is the maximum y from this point onwards
        height = max(p[1] for p in points[i:]) - reference_point[1]
        
        hypervolume += width * height
    
    return hypervolume

if __name__ == "__main__":
    os.makedirs("importance_analysis/", exist_ok=True)
    
    # Dictionary to collect all hypervolume data
    all_hypervolume_data = {}
    
    for method in ["mtl_ibp", "sabr", "shi", "crown_ibp_nofusion", "crown_ibp"]:
        for dataset in ["cifar10", "tinyimagenet"]:
            if dataset == "tinyimagenet" and method in ["crown_ibp_nofusion"]:
                continue    
            if dataset == "cifar10" and method in ["crown_ibp"]:
                continue
            for network in [
                "cnn7"
            ]:
                for eps in [2 / 255, 8 / 255] if dataset == "cifar10" else [1 / 255]:
                    # produce a single combined figure (not separate 'nat'/'cert')
                    for obj in ["combined"]:
                        # Plot Pareto front after specific trial counts and collect for combined figure
                        trial_counts = [10, 20, 50, 75, 100]
                        pareto_map = {}
                        out_dir = "fronts_development"
                        os.makedirs(out_dir, exist_ok=True)
                        
                        for n in trial_counts:
                            # Create a new mega_study for this trial count
                            mega_study = optuna.create_study(
                                directions=["maximize", "maximize"], study_name="moctrain"
                            )
                            for seed in range(3):
                                print(
                                    f"sqlite:///../results/hpo/optuna_results/{dataset}_{network}_{method}_{eps}_{seed}_optuna_study.db"
                                )
                                study = optuna.load_study(
                                    study_name="moctrain",
                                    storage=f"sqlite:///../results/hpo/optuna_results/{dataset}_{network}_{method}_{eps}_{seed}_optuna_study.db",
                                )
                                # Add only the first n trials from this seed
                                mega_study.add_trials(study.trials[:n])
                            
                            # Extract data from mega_study
                            trial_data = []
                            for trial in mega_study.trials:
                                if trial.state == optuna.trial.TrialState.COMPLETE:
                                    trial_data.append({
                                        "Objective0": trial.values[0],
                                        "Objective1": trial.values[1],
                                    })
                            df = pd.DataFrame(trial_data)

                            if eps == 2 / 255:
                                df = df[df["Objective0"] > 0.6]
                                df = df[df["Objective1"] > 0.4]
                            elif eps == 8 / 255:
                                df = df[df["Objective0"] > 0.4]
                                df = df[df["Objective1"] > 0.25]
                            for id, row in df.iterrows():
                                for id2, row2 in df.iterrows():
                                    if (
                                        row2["Objective0"] > row["Objective0"]
                                        and row2["Objective1"] >= row["Objective1"]
                                    ):
                                        df.drop(id, inplace=True)
                                        break
                            
                            if df.shape[0] == 0:
                                continue

                            objs = df[["Objective0", "Objective1"]].values
                            mask = []
                            for i, o in enumerate(objs):
                                dominated = False
                                for j, oj in enumerate(objs):
                                    if j == i:
                                        continue
                                    # maximize both objectives: oj dominates o if oj >= o in both and strictly greater in at least one
                                    if (oj[0] >= o[0] and oj[1] >= o[1]) and (
                                        oj[0] > o[0] or oj[1] > o[1]
                                    ):
                                        dominated = True
                                        break
                                mask.append(not dominated)
                            mask = np.array(mask, dtype=bool)

                            pareto = df[mask]
                            if pareto.shape[0] == 0:
                                continue

                            pareto_sorted = pareto.sort_values(by="Objective0", ascending=False)
                            pareto_map[n] = pareto_sorted

                            # fig = go.Figure()
                            # fig.add_trace(
                            #     go.Scatter(
                            #         x=df["Objective0"],
                            #         y=df["Objective1"],
                            #         mode="markers",
                            #         marker=dict(size=6, color="lightgray"),
                            #         name=f"All ({n} trials per seed)",
                            #     )
                            # )
                            # fig.add_trace(
                            #     go.Scatter(
                            #         x=pareto_sorted["Objective0"],
                            #         y=pareto_sorted["Objective1"],
                            #         mode="lines+markers",
                            #         marker=dict(size=8, color="red"),
                            #         line=dict(color="red"),
                            #         name="Pareto front",
                            #     )
                            # )

                            # fig.update_layout(
                            #     title=f"Pareto front after {n} trials per seed: {dataset} {network} {method} eps={eps} ({obj})",
                            #     xaxis_title=renames["Objective0"],
                            #     yaxis_title=renames["Objective1"],
                            #     template="seaborn",
                            #     width=600,
                            #     height=450,
                            # )

                            # out_path = f"{out_dir}/{dataset}_{network}_{method}_{eps}_{obj}_trials{n}.pdf"
                            # fig.write_image(out_path)

                        # Combined figure: overlay Pareto fronts from different trial counts
                        if len(pareto_map) > 0:
                            # Prepare data for plotting
                            sns.set_style("darkgrid")
                            fig, ax = plt.subplots(figsize=(10, 10))
                            
                            # Color palette for trial counts
                            palette = sns.color_palette("Set2", len(pareto_map))
                            colors_map = {n: palette[idx] for idx, n in enumerate(sorted(pareto_map.keys()))}
                            
                            # Marker styles for different trial counts
                            markers = ['o', 's', '^', 'D', '*']  # circle, square, triangle, diamond, star
                            markers_map = {n: markers[idx % len(markers)] for idx, n in enumerate(sorted(pareto_map.keys()))}
                            
                            # Get all data points for axis limits
                            all_x = []
                            all_y = []
                            for n in sorted(pareto_map.keys()):
                                ps = pareto_map[n]
                                all_x.extend(ps["Objective1"].values)
                                all_y.extend(ps["Objective0"].values)
                            
                            # Plot Pareto fronts for each trial count
                            for n in sorted(pareto_map.keys()):
                                ps = pareto_map[n]
                                x = ps["Objective1"].values
                                y = ps["Objective0"].values
                                
                                # Scatter plot for points with different markers
                                ax.scatter(x, y, s=200, c=[colors_map[n]], marker=markers_map[n],
                                          edgecolor='black', linewidth=2.5, alpha=0.95, zorder=2)
                                
                                # Line connecting Pareto points (sorted by x)
                                if len(ps) > 1:
                                    points = sorted(zip(x, y))
                                    px, py = zip(*points)
                                    ax.plot(px, py, color=colors_map[n], linewidth=10, alpha=0.5, zorder=1)
                            
                            # Styling
                            ax.set_xlabel('Certified Accuracy', fontsize=35, fontweight='bold', labelpad=16)
                            ax.set_ylabel('Natural Accuracy', fontsize=35, fontweight='bold', labelpad=16)
                            ax.tick_params(axis='both', which='major', labelsize=25, colors='black', length=8, width=2)
                            sns.despine(ax=ax, top=True, right=True, left=False, bottom=False)
                            
                            # Set axis limits with padding
                            if all_x and all_y:
                                min_x = min(all_x)
                                max_x = max(all_x)
                                min_y = min(all_y)
                                max_y = max(all_y)
                                x_range = max_x - min_x if max_x > min_x else 0.1
                                y_range = max_y - min_y if max_y > min_y else 0.1
                                ax.set_xlim(max(0, min_x - 0.05 * x_range), min(1, max_x + 0.05 * x_range))
                                ax.set_ylim(max(0, min_y - 0.05 * y_range), min(1, max_y + 0.05 * y_range))
                            
                            # Save with and without legend
                            for show_legend in [False, True]:
                                if show_legend:
                                    legend_elements = [
                                        Line2D([0], [0], marker=markers_map[n], color='w', label=f"{n} trials", 
                                               markerfacecolor=colors_map[n], markersize=20, 
                                               markeredgecolor='black', linewidth=0) 
                                        for n in sorted(pareto_map.keys())
                                    ]
                                    ax.legend(handles=legend_elements, fontsize=30, frameon=True)
                                
                                plt.tight_layout()
                                suffix = '' if show_legend else '_nolegend'
                                out_path = f"{out_dir}/{dataset}_{network}_{method}_{eps}_{obj}_all_fronts{suffix}.pdf"
                                plt.savefig(out_path, dpi=300, bbox_inches='tight', transparent=False)
                            
                            plt.close()
                            
                            # Calculate hypervolumes and generate table
                            hypervolumes = {}
                            for n in sorted(pareto_map.keys()):
                                ps = pareto_map[n]
                                points = list(zip(ps["Objective1"].values, ps["Objective0"].values))
                                hypervolumes[n] = calculate_hypervolume(points)
                            
                            # Store hypervolume data for combined table
                            row_key = (dataset, network, method, eps)
                            all_hypervolume_data[row_key] = hypervolumes
    
    # Generate combined TeX table with all results
    if all_hypervolume_data:
        trial_counts = [10, 20, 50, 75, 100]
        
        tex_content = "\\begin{table}[htbp]\n"
        tex_content += "\\centering\n"
        tex_content += "\\caption{Hypervolume Indicator for Different Trial Counts}\n"
        tex_content += "\\label{tab:combined_hypervolume}\n"
        tex_content += "\\begin{tabular}{lll" + "|" + "c" * len(trial_counts) + "}\n"
        tex_content += "\\toprule\n"
        tex_content += "Dataset & Method & $\\epsilon$ & " + " & ".join([f"{n}" for n in trial_counts]) + " \\\\\n"
        tex_content += "\\midrule\n"
        
        # Sort the keys for consistent output: Dataset -> Epsilon -> Method
        # Method order: SHI (0) -> CROWN_IBP (1) -> SABR (2) -> MTL_IBP (3)
        method_order = {'shi': 0, 'crown_ibp_nofusion': 1, 'crown_ibp': 2, 'sabr': 3, 'mtl_ibp': 4}
        eps_order = {2/255: 0, 8/255: 1, 1/255: 2}
        
        def sort_key(x):
            dataset, network, method, eps = x
            dataset_priority = 0 if dataset == 'cifar10' else 1
            eps_priority = eps_order.get(eps, 999)
            method_priority = method_order.get(method, 999)
            return (dataset_priority, eps_priority, method_priority)
        
        sorted_keys = sorted(all_hypervolume_data.keys(), key=sort_key)
        
        for dataset, network, method, eps in sorted_keys:
            hypervolumes = all_hypervolume_data[(dataset, network, method, eps)]
            
            # Format dataset name
            dataset_str = "CIFAR-10" if dataset == "cifar10" else "Tiny-ImageNet"
            
            # Format method name
            method_str = method.upper().replace("_", "-")
            
            # Format epsilon
            if eps == 2/255:
                eps_str = "2/255"
            elif eps == 8/255:
                eps_str = "8/255"
            elif eps == 1/255:
                eps_str = "1/255"
            else:
                eps_str = f"{eps:.4f}"
            
            # Build row
            tex_content += f"{dataset_str} & {method_str} & {eps_str}"
        
            for idx, n in enumerate(trial_counts):
                if n in hypervolumes:
                    hv = hypervolumes[n]
                    hv_str = f"{hv:.4f}"
                    
                    # Add improvement percentage for non-first trials
                    if idx > 0:
                        prev_n = trial_counts[idx - 1]
                        if prev_n in hypervolumes:
                            prev_hv = hypervolumes[prev_n]
                            if prev_hv > 0:
                                improvement = ((hv - prev_hv) / prev_hv * 100)
                                hv_str += f" ({improvement:+.2f}\\%)"
                    
                    tex_content += f" & {hv_str}"
                else:
                    tex_content += " & ---"
            
            tex_content += " \\\\\n"
        
        tex_content += "\\bottomrule\n"
        tex_content += "\\end{tabular}\n"
        tex_content += "\\end{table}\n"
        
        # Save combined TeX table
        out_dir = "fronts_development"
        combined_tex_path = f"{out_dir}/hypervolume_summary.tex"
        with open(combined_tex_path, 'w') as f:
            f.write(tex_content)
        
        print(f"\\nGenerated combined hypervolume table: {combined_tex_path}")

                        