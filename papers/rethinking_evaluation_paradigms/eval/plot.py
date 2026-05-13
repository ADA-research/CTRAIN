import os
from matplotlib.lines import Line2D
import numpy as np
import json

from util import get_pareto_front

PLOTS_PATH = "./plots"
TEST_SAMPLES = 10000
# for results in appendix that investigate differences across networks
# TEST_SAMPLES = 1000
ARTIFICIAL_TIMEOUT = 300
# for results in appendix that investigate differences across networks
# ARTIFICIAL_TIMEOUT = 300

# Configuration for comparison plots
GENERATE_TESTSAMPLES_COMPARISON = False  # Generate plots comparing 10k vs 1k samples
GENERATE_TIMEOUT_COMPARISON = False     # Generate plots comparing 1000s vs 300s timeouts

def plot_results(pareto_results_sorted, methods=["mtl_ibp", "sabr", "shi", "crown_ibp_nofusion"], include_literature=True):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    for dataset in ['cifar10', 'tinyimagenet', 'mnist']:
        for eps in [1/255, 2/255, 8/255, 0.3]:
           for architecture in ['cnn7', 'wide_cnn7', 'narrow_cnn7', 'cnn5', 'cnn9', 'resnet18']:
            # Hardcoded data: (Certified Accuracy, Natural Accuracy, Name)
                literature_results = {
                    "cifar10": {
                        "0.00784313725490196": [
                            (66.84, 52.85, 'SHI-IBP'),
                            (71.52, 53.97, 'CROWN-IBP'),
                            (79.24, 62.84, 'SABR'),
                            (80.11, 63.24, 'MTL-IBP'),
                            (67.49, 55.99, 'SHI-IBP \n(CTBENCH)'),
                            (67.60, 57.11, 'CROWN-IBP \n(CTBENCH)'),
                            (77.86, 63.61, 'SABR \n(CTBENCH)'),
                            (78.82, 64.41, 'MTL-IBP \n(CTBENCH)'),
                        ],
                        "0.03137254901960784": [
                            (48.94, 34.97, 'SHI-IBP'),
                            (46.29, 33.38, 'CROWN-IBP'),
                            (52.38, 35.13, 'SABR'),
                            (53.35, 35.44, 'MTL-IBP'),
                            (48.51, 35.28, 'SHI-IBP \n(CTBENCH)'),
                            (48.25, 32.59, 'CROWN-IBP \n(CTBENCH)'),
                            (52.71, 35.34, 'SABR \n(CTBENCH)'),
                            (54.28, 35.41, 'MTL-IBP \n(CTBENCH)'),
                        ]
                    },
                    "tinyimagenet": {
                        "0.00392156862745098": [
                            (25.92, 17.87, 'SHI-IBP'),
                            (25.62, 17.93, 'CROWN-IBP'),
                            (28.85, 20.46, 'SABR'),
                            (37.56, 26.09, 'MTL-IBP'),
                            (26.77, 19.82, 'SHI-IBP \n(CTBENCH)'),
                            (28.44, 22.14, 'CROWN-IBP \n(CTBENCH)'),
                            (30.58, 20.96, 'SABR \n(CTBENCH)'),
                            (35.97, 27.73, 'MTL-IBP \n(CTBENCH)'),
                        ],
                    },
                    'mnist': {
                        "0.3": [
                            (97.67, 93.10, 'SHI-IBP'),
                            (98.18, 92.98, 'CROWN-IBP'),
                            (98.75, 92.98, 'SABR'),
                            (98.80, 93.62, 'MTL-IBP'),
                            (98.54, 93.80, 'SHI-IBP \n(CTBENCH)'),
                            (98.48, 93.90, 'CROWN-IBP \n(CTBENCH)'),
                            (98.66, 93.68, 'SABR \n(CTBENCH)'),
                            (98.74, 93.90, 'MTL-IBP \n(CTBENCH)'),
                        ],
                    }
                }
                if not literature_results.get(dataset) or not literature_results[dataset].get(str(eps)):
                    continue
                
                lit_to_ctrain_map = {
                    "SHI-IBP": "shi",
                    "CROWN-IBP": "crown_ibp_nofusion",
                    "SABR": "sabr",
                    "MTL-IBP": "mtl_ibp",
                }
                ctrain_to_lit_map = {v: k for k, v in lit_to_ctrain_map.items()}
                
                # Filter literature results to only include specified methods
                lit_plot = literature_results[dataset][str(eps)] = [
                    result for result in literature_results[dataset][str(eps)]
                    if any(ctrain_to_lit_map[method] in result[2] for method in methods)
                ]
                
                # data = [
                #     (res['clean_classification_accuracy'], res['certified_accuracy'], f"{res['cert_train_method']} (ours)") for res in pareto_results_sorted if res['dataset'] == dataset and abs(float(res['eps']) - eps) < 1e-6 and res['total_samples'] == 10_000 and res['cert_train_method'] in methods
                # ]
                # all_data = literature_results[dataset][str(eps)] + [(res['clean_classification_accuracy'], res['certified_accuracy'], f"{res['cert_train_method']} (ours)") for res in pareto_results_sorted if res['dataset'] == dataset and abs(float(res['eps']) - eps) < 1e-6 and res['total_samples'] == 10_000]
                # TODO: REMOVE THIS HACK!
                data = [
                    (res['clean_classification_accuracy'], res['certified_accuracy'], f"{res['cert_train_method']} (ours)") for res in pareto_results_sorted if res['dataset'] == dataset and abs(float(res['eps']) - eps) < 1e-6 and res['cert_train_method'] in methods and res['total_samples'] > 1 and res['architecture'] == architecture
                ]
                if len(data) == 0:
                    continue
                all_data = literature_results[dataset][str(eps)] + [(res['clean_classification_accuracy'], res['certified_accuracy'], f"{res['cert_train_method']} (ours)") for res in pareto_results_sorted if res['dataset'] == dataset and abs(float(res['eps']) - eps) < 1e-6 and res['total_samples'] > 1]
                
                if include_literature:
                    data = data + lit_plot
                
                # annotation_offsets = [
                #     (10, 10),    # IBP
                #     (10, 10),   # CROWN-IBP
                #     (10, -30),  # SABR
                #     (-90, 20),   # MTL-IBP
                #     (15, -50),    # IBP (CTBENCH)
                #     (10, 10),   # CROWN-IBP (CTBENCH)
                #     (15, -50),  # SABR (CTBENCH)
                #     (15, -50),   # MTL-IBP (CTBENCH)
                #     (10, 6),    # IBP (CTRAIN)
                #     (10, 10),   # CROWN-IBP (CTRAIN)
                #     (15, -50),  # SABR (CTRAIN)
                #     (10, 10),   # MTL-IBP (CTRAIN)
                # ]

                x = [d[1] for d in data]
                y = [d[0] for d in data]
                labels = [d[2] for d in data]

                # Use white background and clean style
                sns.set_style("darkgrid")
                fig, ax = plt.subplots(figsize=(10, 10))

                groups = []            
                for label in labels:
                    if '(ours)' in label:
                        if "mtl_ibp" in label:
                            groups.append(0)
                        elif "sabr" in label:
                            groups.append(1)
                        elif "crown_ibp" in label:
                            groups.append(2)
                        elif "shi" in label:
                            groups.append(3)
                    else:
                        if "MTL-IBP" in label:
                            groups.append(4)
                        elif "SABR" in label:
                            groups.append(5)
                        elif "CROWN-IBP" in label:
                            groups.append(6)
                        elif "SHI-IBP" in label:
                            groups.append(7)
                            
                group_labels = ['MTL (ours)', 'SABR (ours)', 'C-IBP (ours)', 'IBP (ours)', 'MTL (lit.)', 'SABR (lit.)', 'C-IBP (lit.)', 'IBP (lit.)']
                # group_labels = [group_labels[i] for i in range(len(group_labels)) if i in groups]
                
                print(groups)

                # Assign a color to each group
                palette = sns.color_palette("Set2", 8)
                point_colors = [palette[g] for g in groups]

                # Use seaborn's scatterplot for consistent style
                ax.scatter(x, y, s=500, c=point_colors, edgecolor='black', linewidth=2.5, alpha=0.95, zorder=2)
                # Add lines connecting points within each unique group
                for group_id in set(groups):
                    if '(lit' in group_labels[group_id]:
                        continue
                    mask = [i for i, g in enumerate(groups) if g == group_id]
                    if len(mask) > 1:
                        group_x = [x[i] for i in mask]
                        group_y = [y[i] for i in mask]
                        # Sort points by x-coordinate to connect them properly
                        points = sorted(zip(group_x, group_y))
                        group_x, group_y = zip(*points)
                        ax.plot(group_x, group_y, color=palette[group_id], linewidth=10, alpha=0.5, zorder=1)

                # Annotate each point
                # for i, label in enumerate(labels):
                #     if "CTRAIN" in label:
                #         continue
                #     ax.annotate(label, (x[i], y[i]), 
                #                 # textcoords="offset points", 
                #                 # xytext=annotation_offsets[i],
                #                 ha='left',
                #                 fontsize=30, fontweight='bold', color=point_colors[i], zorder=4)


                ax.set_xlabel('Certified Accuracy', fontsize=35, fontweight='bold', labelpad=16)
                ax.set_ylabel('Natural Accuracy', fontsize=35, fontweight='bold', labelpad=16)
                # ax.set_title(f'CIFAR10 - $\epsilon = {eps}$', fontsize=40, fontweight='bold', pad=30, color='black')

                # Remove spines for a clean look
                sns.despine(ax=ax, top=True, right=True, left=False, bottom=False)

                # Set ticks
                ax.tick_params(axis='both', which='major', labelsize=25, colors='black', length=8, width=2)
                # ax.set_facecolor('white')
                # Find global max values across all data points for consistent scales
                max_cert_acc = max(d[1] for d in all_data)
                min_cert_acc = min(d[1] for d in all_data)
                max_nat_acc = max(d[0] for d in all_data)
                min_nat_acc = min(d[0] for d in all_data)

                # Set axis limits with some padding
                ax.set_xlim(max(0, min_cert_acc - 2), min(100, max_cert_acc + 2))
                ax.set_ylim(max(0, min_nat_acc - 2), min(max_nat_acc + 2, 100))
                # fig.patch.set_facecolor('white')
                
                for legend in [False, True]:
                    if legend:
                        legend_elements = [
                            Line2D([0], [0], marker='o', color='w', label=group_labels[i], markerfacecolor=palette[i], markersize=20, markeredgecolor='black', linewidth=0) for i in set(groups)
                        ]
                        ax.legend(handles=legend_elements, fontsize=30, frameon=True)
                                    
                        plt.tight_layout()
                        plt.savefig(f'{PLOTS_PATH}/comparison_{architecture}_{dataset}_{eps}_{methods}.pdf', dpi=300, bbox_inches='tight', transparent=False)
                    else:
                        plt.tight_layout()
                        plt.savefig(f'{PLOTS_PATH}/comparison_{architecture}_{dataset}_{eps}_{methods}_nolegend.pdf', dpi=300, bbox_inches='tight', transparent=False)
                
                plt.close()
 
def plot_pareto_front_across_networks(pareto_results_sorted, method):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    # Define network types and their display names
    network_types = {
        'cnn5': 'CNN-5',
        'cnn7': 'CNN-7',
        'wide_cnn7': 'CNN-7 Wide',
        'narrow_cnn7': 'CNN-7 Narrow',
        'cnn9': 'CNN-9',
        'resnet18': 'ResNet-18'
    }

    for dataset in ['cifar10']:
        for eps in [2/255]:
            # Create one plot for all networks
            sns.set_style("darkgrid")
            fig, ax = plt.subplots(figsize=(10, 10))

            # Create color palette for networks
            palette = sns.color_palette("Set2", len(network_types))
            all_points = []
            legend_elements = []

            # Process each network
            for idx, (architecture, network_name) in enumerate(network_types.items()):
                # Get results for this specific network
                results = [res for res in pareto_results_sorted 
                          if res['dataset'] == dataset 
                          and abs(float(res['eps']) - eps) < 1e-6 
                          and res['cert_train_method'] == method 
                          and res['architecture'] == architecture 
                          and res['total_samples'] > 1]
                
                if not results:
                    continue

                # Extract data
                x = [res['certified_accuracy'] for res in results]
                y = [res['clean_classification_accuracy'] for res in results]
                points = list(zip(x, y))
                all_points.extend(points)

                # Plot points
                ax.scatter(x, y, s=500, c=[palette[idx]], edgecolor='black', 
                          linewidth=2.5, alpha=0.9, zorder=2)

                # Find Pareto front for this network
                pareto_points = []
                for i, point in enumerate(points):
                    is_dominated = False
                    for other_point in points:
                        if (other_point[0] >= point[0] and other_point[1] >= point[1] and 
                            (other_point[0] > point[0] or other_point[1] > point[1])):
                            is_dominated = True
                            break
                    if not is_dominated:
                        pareto_points.append(point)

                # Sort Pareto points and plot
                if pareto_points:
                    pareto_points.sort()
                    pareto_x, pareto_y = zip(*pareto_points)
                    ax.plot(pareto_x, pareto_y, '-', color=palette[idx], 
                           linewidth=10, zorder=3, alpha=.5)

                # Add to legend
                legend_elements.append(
                    Line2D([0], [0], color=palette[idx], label=network_name,
                          marker='o', markersize=15, linewidth=3,
                          markerfacecolor=palette[idx], markeredgecolor='black')
                )

            # Styling
            ax.set_xlabel('Certified Accuracy', fontsize=35, fontweight='bold', labelpad=16)
            ax.set_ylabel('Natural Accuracy', fontsize=35, fontweight='bold', labelpad=16)
            ax.tick_params(axis='both', which='major', labelsize=25, colors='black', length=8, width=2)
            sns.despine(ax=ax, top=True, right=True, left=False, bottom=False)

            # Set axis limits with padding
            if all_points:
                min_x = min(p[0] for p in all_points)
                max_x = max(p[0] for p in all_points)
                min_y = min(p[1] for p in all_points)
                max_y = max(p[1] for p in all_points)
                ax.set_xlim(max(0, min_x - 2), min(100, max_x + 2))
                ax.set_ylim(max(0, min_y - 2), min(max_y + 2, 100))

            # Save with and without legend
            for show_legend in [False, True]:
                if show_legend:
                    ax.legend(handles=legend_elements, fontsize=25,)
                
                plt.tight_layout()
                suffix = '' if show_legend else '_nolegend'
                plt.savefig(f'{PLOTS_PATH}/network_comparison_{method}_{dataset}_{eps}{suffix}.pdf',
                          dpi=300, bbox_inches='tight', transparent=False)
            
            plt.close()

def plot_testsamples_comparison(all_results):
    """
    Generate comparison plots between 10,000 samples and 1,000 randomly sampled instances.
    Generates plots for dataset/architecture/eps/method combinations that have results for both sample sizes.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    # Load full results (10k samples)
    with open('../results/verification/summary_results.json', 'r') as f:
        results_10k_raw = json.load(f)
    
    # Load sampled results (1k samples)
    with open('../results/verification/summary_results_testsamples1000.json', 'r') as f:
        results_1k_raw = json.load(f)
    
    # Filter to actual 10k and 1k samples respectively
    results_10k = [r for r in results_10k_raw if r['total_samples'] == 10000]
    results_1k = [r for r in results_1k_raw if r['total_samples'] == 1000]
    
    # Create Pareto fronts for both
    pareto_10k = get_pareto_front(results_10k)
    pareto_1k = get_pareto_front(results_1k)
    
    plots_path = "./plots_testsamples_comparison"
    os.makedirs(plots_path, exist_ok=True)
    
    # Group results by dataset, architecture, eps, method
    grouping_keys = ['dataset', 'architecture', 'eps', 'cert_train_method']
    
    # Find groups that exist in both 10k and 1k datasets
    groups_10k = set(tuple(r[k] for k in grouping_keys) for r in results_10k)
    groups_1k = set(tuple(r[k] for k in grouping_keys) for r in results_1k)
    valid_groups = groups_10k & groups_1k
    
    if not valid_groups:
        print("No groups with results for both 10k and 1k samples found")
        return
    
    print(f"Found {len(valid_groups)} groups with results for both 10k and 1k samples")
    
    # Create comparison plots for each valid group
    for (dataset, architecture, eps, method) in valid_groups:
        # Get data for 10k
        data_10k = [
            (res['clean_classification_accuracy'], res['certified_accuracy'], "10k samples") 
            for res in pareto_10k 
            if res['dataset'] == dataset 
            and res['architecture'] == architecture 
            and abs(float(res['eps']) - float(eps)) < 1e-6 
            and res['cert_train_method'] == method
        ]
        
        # Get data for 1k
        data_1k = [
            (res['clean_classification_accuracy'], res['certified_accuracy'], "1k samples") 
            for res in pareto_1k 
            if res['dataset'] == dataset 
            and res['architecture'] == architecture 
            and abs(float(res['eps']) - float(eps)) < 1e-6 
            and res['cert_train_method'] == method
        ]
        
        if not data_10k or not data_1k:
            continue
        
        # Combine data
        data = data_10k + data_1k
        
        x = [d[1] for d in data]
        y = [d[0] for d in data]
        labels = [d[2] for d in data]
        
        # Create plot
        sns.set_style("darkgrid")
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Color by samples
        colors = ['#1f77b4' if '10k' in label else '#ff7f0e' for label in labels]
        ax.scatter(x, y, s=500, c=colors, edgecolor='black', linewidth=2.5, alpha=0.95, zorder=2)
        
        # Add lines connecting Pareto fronts
        for sample_type, color, plot_data in [("10k samples", '#1f77b4', data_10k), ("1k samples", '#ff7f0e', data_1k)]:
            if len(plot_data) > 1:
                points = sorted([(p[1], p[0]) for p in plot_data])
                px, py = zip(*points)
                ax.plot(px, py, color=color, linewidth=10, alpha=0.5, zorder=1)
        
        # Styling
        ax.set_xlabel('Certified Accuracy', fontsize=35, fontweight='bold', labelpad=16)
        ax.set_ylabel('Natural Accuracy', fontsize=35, fontweight='bold', labelpad=16)
        ax.tick_params(axis='both', which='major', labelsize=25, colors='black', length=8, width=2)
        sns.despine(ax=ax, top=True, right=True, left=False, bottom=False)
        
        # Set axis limits
        min_x = min(x)
        max_x = max(x)
        min_y = min(y)
        max_y = max(y)
        ax.set_xlim(max(0, min_x - 2), min(100, max_x + 2))
        ax.set_ylim(max(0, min_y - 2), min(max_y + 2, 100))
        
        # Save with and without legend
        for show_legend in [False, True]:
            if show_legend:
                legend_elements = [
                    Line2D([0], [0], marker='o', color='w', label="10k samples", 
                           markerfacecolor='#1f77b4', markersize=20, markeredgecolor='black', linewidth=0),
                    Line2D([0], [0], marker='o', color='w', label="1k samples", 
                           markerfacecolor='#ff7f0e', markersize=20, markeredgecolor='black', linewidth=0),
                ]
                ax.legend(handles=legend_elements, fontsize=30, frameon=True)
            
            plt.tight_layout()
            suffix = '' if show_legend else '_nolegend'
            plt.savefig(f'{plots_path}/comparison_{architecture}_{dataset}_{eps}_{method}_testsamples{suffix}.pdf', 
                       dpi=300, bbox_inches='tight', transparent=False)
        
        plt.close()
    
    print(f"Generated {len(valid_groups)} test samples comparison plots")

def plot_timeout_comparison(all_results):
    """
    Generate comparison plots between 1000s timeout and 300s timeout.
    Generates plots for dataset/architecture/eps/method combinations that have results for both timeouts.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    # Load results with 1000s timeout (default results)
    with open('../results/verification/summary_results.json', 'r') as f:
        results_1000s = json.load(f)
    
    # Load results with 300s timeout
    with open('../results/verification/summary_results_timeout300.json', 'r') as f:
        results_300s = json.load(f)
    
    # Create Pareto fronts for both
    pareto_1000s = get_pareto_front(results_1000s)
    pareto_300s = get_pareto_front(results_300s)
    
    plots_path = "./plots_timeout_comparison"
    os.makedirs(plots_path, exist_ok=True)
    
    # Group results by dataset, architecture, eps, method
    grouping_keys = ['dataset', 'architecture', 'eps', 'cert_train_method']
    
    # Find groups that exist in both datasets
    groups_1000s = set(tuple(r[k] for k in grouping_keys) for r in results_1000s)
    groups_300s = set(tuple(r[k] for k in grouping_keys) for r in results_300s)
    valid_groups = groups_1000s & groups_300s
    
    if not valid_groups:
        print("No groups with results for both 1000s and 300s timeout found")
        return
    
    print(f"Found {len(valid_groups)} groups with results for both 1000s and 300s timeout")
    
    # Create comparison plots for each valid group
    for (dataset, architecture, eps, method) in valid_groups:
        # Get data for 1000s
        data_1000s = [
            (res['clean_classification_accuracy'], res['certified_accuracy'], "1000s timeout") 
            for res in pareto_1000s 
            if res['dataset'] == dataset 
            and res['architecture'] == architecture 
            and abs(float(res['eps']) - float(eps)) < 1e-6 
            and res['cert_train_method'] == method
        ]
        
        # Get data for 300s
        data_300s = [
            (res['clean_classification_accuracy'], res['certified_accuracy'], "300s timeout") 
            for res in pareto_300s 
            if res['dataset'] == dataset 
            and res['architecture'] == architecture 
            and abs(float(res['eps']) - float(eps)) < 1e-6 
            and res['cert_train_method'] == method
        ]
        
        if not data_1000s or not data_300s:
            continue
        
        # Combine data
        data = data_1000s + data_300s
        
        x = [d[1] for d in data]
        y = [d[0] for d in data]
        labels = [d[2] for d in data]
        
        # Create plot
        sns.set_style("darkgrid")
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Color by timeout
        colors = ['#2ca02c' if '1000s' in label else '#d62728' for label in labels]
        ax.scatter(x, y, s=500, c=colors, edgecolor='black', linewidth=2.5, alpha=0.95, zorder=2)
        
        # Add lines connecting Pareto fronts
        for timeout_type, color, plot_data in [("1000s timeout", '#2ca02c', data_1000s), ("300s timeout", '#d62728', data_300s)]:
            if len(plot_data) > 1:
                points = sorted([(p[1], p[0]) for p in plot_data])
                px, py = zip(*points)
                ax.plot(px, py, color=color, linewidth=10, alpha=0.5, zorder=1)
        
        # Styling
        ax.set_xlabel('Certified Accuracy', fontsize=35, fontweight='bold', labelpad=16)
        ax.set_ylabel('Natural Accuracy', fontsize=35, fontweight='bold', labelpad=16)
        ax.tick_params(axis='both', which='major', labelsize=25, colors='black', length=8, width=2)
        sns.despine(ax=ax, top=True, right=True, left=False, bottom=False)
        
        # Set axis limits
        min_x = min(x)
        max_x = max(x)
        min_y = min(y)
        max_y = max(y)
        ax.set_xlim(max(0, min_x - 2), min(100, max_x + 2))
        ax.set_ylim(max(0, min_y - 2), min(max_y + 2, 100))
        
        # Save with and without legend
        for show_legend in [False, True]:
            if show_legend:
                legend_elements = [
                    Line2D([0], [0], marker='o', color='w', label="1000s timeout", 
                           markerfacecolor='#2ca02c', markersize=20, markeredgecolor='black', linewidth=0),
                    Line2D([0], [0], marker='o', color='w', label="300s timeout", 
                           markerfacecolor='#d62728', markersize=20, markeredgecolor='black', linewidth=0),
                ]
                ax.legend(handles=legend_elements, fontsize=30, frameon=True)
            
            plt.tight_layout()
            suffix = '' if show_legend else '_nolegend'
            plt.savefig(f'{plots_path}/comparison_{architecture}_{dataset}_{eps}_{method}_timeout{suffix}.pdf', 
                       dpi=300, bbox_inches='tight', transparent=False)
        
        plt.close()
    
    print(f"Generated {len(valid_groups)} timeout comparison plots")

def plot_testsamples_multiple_comparison(all_results):
    """
    Generate comparison plots between multiple test sample sizes (10000, 5000, 2500, 1000).
    Generates plots for dataset/architecture/eps/method combinations that have results for the specified sample sizes.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    sample_sizes = [10000, 5000, 2500, 1000]
    result_files = {
        10000: '../results/verification/summary_results.json',
        5000: '../results/verification/summary_results_testsamples5000.json',
        2500: '../results/verification/summary_results_testsamples2500.json',
        1000: '../results/verification/summary_results_testsamples1000.json',
    }
    
    # Load results for each sample size
    results_by_sample = {}
    pareto_by_sample = {}
    
    for sample_size, file_path in result_files.items():
        try:
            with open(file_path, 'r') as f:
                results_raw = json.load(f)
            results_filtered = [r for r in results_raw if r['total_samples'] == sample_size]
            results_by_sample[sample_size] = results_filtered
            pareto_by_sample[sample_size] = get_pareto_front(results_filtered)
        except FileNotFoundError:
            print(f"Warning: File {file_path} not found. Skipping sample size {sample_size}.")
            continue
    
    if len(results_by_sample) < 2:
        print("Not enough sample size results available for comparison")
        return
    
    plots_path = "./plots_testsamples_multiple_comparison"
    os.makedirs(plots_path, exist_ok=True)
    
    # Group results by dataset, architecture, eps, method
    grouping_keys = ['dataset', 'architecture', 'eps', 'cert_train_method']
    
    # Find groups that exist in all available datasets
    all_groups = [set(tuple(r[k] for k in grouping_keys) for r in results_by_sample[s]) 
                  for s in results_by_sample.keys()]
    valid_groups = set.intersection(*all_groups) if all_groups else set()
    
    if not valid_groups:
        print("No groups with results for all sample sizes found")
        return
    
    print(f"Found {len(valid_groups)} groups with results for multiple sample sizes")
    
    # Color palette for different sample sizes
    colors_map = {1000: '#d62728', 2500: '#ff7f0e', 5000: '#2ca02c', 10000: '#1f77b4'}
    
    # Create comparison plots for each valid group
    for (dataset, architecture, eps, method) in valid_groups:
        data_all = []
        
        for sample_size in sorted(results_by_sample.keys()):
            data = [
                (res['clean_classification_accuracy'], res['certified_accuracy'], f"{sample_size} samples") 
                for res in pareto_by_sample[sample_size]
                if res['dataset'] == dataset 
                and res['architecture'] == architecture 
                and abs(float(res['eps']) - float(eps)) < 1e-6 
                and res['cert_train_method'] == method
            ]
            data_all.extend(data)
        
        if not data_all:
            continue
        
        x = [d[1] for d in data_all]
        y = [d[0] for d in data_all]
        labels = [d[2] for d in data_all]
        
        # Create plot
        sns.set_style("darkgrid")
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Color by sample size - extract numeric value from label (e.g., '250 samples' -> 250)
        colors = [colors_map[int(label.split()[0])] for label in labels]
        ax.scatter(x, y, s=500, c=colors, edgecolor='black', linewidth=2.5, alpha=0.95, zorder=2)
        
        # Add lines connecting Pareto fronts for each sample size
        for sample_size in sorted(results_by_sample.keys()):
            plot_data = [
                (res['clean_classification_accuracy'], res['certified_accuracy']) 
                for res in pareto_by_sample[sample_size]
                if res['dataset'] == dataset 
                and res['architecture'] == architecture 
                and abs(float(res['eps']) - float(eps)) < 1e-6 
                and res['cert_train_method'] == method
            ]
            if len(plot_data) > 1:
                points = sorted([(p[1], p[0]) for p in plot_data])
                px, py = zip(*points)
                ax.plot(px, py, color=colors_map[sample_size], linewidth=10, alpha=0.5, zorder=1)
        
        # Styling
        ax.set_xlabel('Certified Accuracy', fontsize=35, fontweight='bold', labelpad=16)
        ax.set_ylabel('Natural Accuracy', fontsize=35, fontweight='bold', labelpad=16)
        ax.tick_params(axis='both', which='major', labelsize=25, colors='black', length=8, width=2)
        sns.despine(ax=ax, top=True, right=True, left=False, bottom=False)
        
        # Set axis limits
        min_x = min(x)
        max_x = max(x)
        min_y = min(y)
        max_y = max(y)
        ax.set_xlim(max(0, min_x - 2), min(100, max_x + 2))
        ax.set_ylim(max(0, min_y - 2), min(max_y + 2, 100))
        
        # Save with and without legend
        for show_legend in [False, True]:
            if show_legend:
                legend_elements = [
                    Line2D([0], [0], marker='o', color='w', label="1000 samples", 
                           markerfacecolor=colors_map[1000], markersize=20, markeredgecolor='black', linewidth=0),
                    Line2D([0], [0], marker='o', color='w', label="2500 samples", 
                           markerfacecolor=colors_map[2500], markersize=20, markeredgecolor='black', linewidth=0),
                    Line2D([0], [0], marker='o', color='w', label="5000 samples", 
                           markerfacecolor=colors_map[5000], markersize=20, markeredgecolor='black', linewidth=0),
                    Line2D([0], [0], marker='o', color='w', label="10000 samples", 
                           markerfacecolor=colors_map[10000], markersize=20, markeredgecolor='black', linewidth=0),
                ]
                ax.legend(handles=legend_elements, fontsize=30, frameon=True)
            
            plt.tight_layout()
            suffix = '' if show_legend else '_nolegend'
            plt.savefig(f'{plots_path}/comparison_{architecture}_{dataset}_{eps}_{method}_testsamples{suffix}.pdf', 
                       dpi=300, bbox_inches='tight', transparent=False)
        
        plt.close()
    
    print(f"Generated {len(valid_groups)} multi-sample comparison plots")

def plot_timeout_multiple_comparison(all_results):
    """
    Generate comparison plots between multiple timeout values (1000s, 500s, 250s, 100s).
    Generates plots for dataset/architecture/eps/method combinations that have results for the specified timeouts.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    timeout_values = [1000, 500, 250, 100]
    result_files = {
        1000: '../results/verification/summary_results.json',  # Default 1000s
        500: '../results/verification/summary_results_timeout500.json',
        250: '../results/verification/summary_results_timeout250.json',
        100: '../results/verification/summary_results_timeout100.json',
    }
    
    # Load results for each timeout
    results_by_timeout = {}
    pareto_by_timeout = {}
    
    for timeout, file_path in result_files.items():
        try:
            with open(file_path, 'r') as f:
                results_raw = json.load(f)
            results_by_timeout[timeout] = results_raw
            pareto_by_timeout[timeout] = get_pareto_front(results_raw)
        except FileNotFoundError:
            print(f"Warning: File {file_path} not found. Skipping timeout {timeout}s.")
            continue
    
    if len(results_by_timeout) < 2:
        print("Not enough timeout results available for comparison")
        return
    
    plots_path = "./plots_timeout_multiple_comparison"
    os.makedirs(plots_path, exist_ok=True)
    
    # Group results by dataset, architecture, eps, method
    grouping_keys = ['dataset', 'architecture', 'eps', 'cert_train_method']
    
    # Find groups that exist in all available timeout datasets
    all_groups = [set(tuple(r[k] for k in grouping_keys) for r in results_by_timeout[t]) 
                  for t in results_by_timeout.keys()]
    valid_groups = set.intersection(*all_groups) if all_groups else set()
    
    if not valid_groups:
        print("No groups with results for all timeouts found")
        return
    
    print(f"Found {len(valid_groups)} groups with results for multiple timeouts")
    
    # Color palette for different timeouts
    colors_map = {100: '#d62728', 250: '#ff7f0e', 500: '#2ca02c', 1000: '#1f77b4'}
    
    # Create comparison plots for each valid group
    for (dataset, architecture, eps, method) in valid_groups:
        data_all = []
        
        for timeout in sorted(results_by_timeout.keys()):
            data = [
                (res['clean_classification_accuracy'], res['certified_accuracy'], f"{timeout}s timeout") 
                for res in pareto_by_timeout[timeout]
                if res['dataset'] == dataset 
                and res['architecture'] == architecture 
                and abs(float(res['eps']) - float(eps)) < 1e-6 
                and res['cert_train_method'] == method
            ]
            data_all.extend(data)
        
        if not data_all:
            continue
        
        x = [d[1] for d in data_all]
        y = [d[0] for d in data_all]
        labels = [d[2] for d in data_all]
        
        # Create plot
        sns.set_style("darkgrid")
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Color by timeout - extract numeric value from label (e.g., '250s timeout' -> 250)
        colors = [colors_map[int(label.split()[0].rstrip('s'))] for label in labels]
        ax.scatter(x, y, s=500, c=colors, edgecolor='black', linewidth=2.5, alpha=0.95, zorder=2)
        
        # Add lines connecting Pareto fronts for each timeout
        for timeout in sorted(results_by_timeout.keys()):
            plot_data = [
                (res['clean_classification_accuracy'], res['certified_accuracy']) 
                for res in pareto_by_timeout[timeout]
                if res['dataset'] == dataset 
                and res['architecture'] == architecture 
                and abs(float(res['eps']) - float(eps)) < 1e-6 
                and res['cert_train_method'] == method
            ]
            if len(plot_data) > 1:
                points = sorted([(p[1], p[0]) for p in plot_data])
                px, py = zip(*points)
                ax.plot(px, py, color=colors_map[timeout], linewidth=10, alpha=0.5, zorder=1)
        
        # Styling
        ax.set_xlabel('Certified Accuracy', fontsize=35, fontweight='bold', labelpad=16)
        ax.set_ylabel('Natural Accuracy', fontsize=35, fontweight='bold', labelpad=16)
        ax.tick_params(axis='both', which='major', labelsize=25, colors='black', length=8, width=2)
        sns.despine(ax=ax, top=True, right=True, left=False, bottom=False)
        
        # Set axis limits
        min_x = min(x)
        max_x = max(x)
        min_y = min(y)
        max_y = max(y)
        ax.set_xlim(max(0, min_x - 2), min(100, max_x + 2))
        ax.set_ylim(max(0, min_y - 2), min(max_y + 2, 100))
        
        # Save with and without legend
        for show_legend in [False, True]:
            if show_legend:
                legend_elements = [
                    Line2D([0], [0], marker='o', color='w', label="100s timeout", 
                           markerfacecolor=colors_map[100], markersize=20, markeredgecolor='black', linewidth=0),
                    Line2D([0], [0], marker='o', color='w', label="250s timeout", 
                           markerfacecolor=colors_map[250], markersize=20, markeredgecolor='black', linewidth=0),
                    Line2D([0], [0], marker='o', color='w', label="500s timeout", 
                           markerfacecolor=colors_map[500], markersize=20, markeredgecolor='black', linewidth=0),
                    Line2D([0], [0], marker='o', color='w', label="1000s timeout", 
                           markerfacecolor=colors_map[1000], markersize=20, markeredgecolor='black', linewidth=0),
                ]
                ax.legend(handles=legend_elements, fontsize=30, frameon=True)
            
            plt.tight_layout()
            suffix = '' if show_legend else '_nolegend'
            plt.savefig(f'{plots_path}/comparison_{architecture}_{dataset}_{eps}_{method}_timeout{suffix}.pdf', 
                       dpi=300, bbox_inches='tight', transparent=False)
        
        plt.close()
    
    print(f"Generated {len(valid_groups)} multi-timeout comparison plots")

if __name__ == "__main__":
    results_file_name = "summary_results"
    if ARTIFICIAL_TIMEOUT != 1000:
        results_file_name += f"_timeout{ARTIFICIAL_TIMEOUT}"
        PLOTS_PATH += f"_timeout{ARTIFICIAL_TIMEOUT}"
    if TEST_SAMPLES != 10000:
        results_file_name += f"_testsamples{TEST_SAMPLES}"
        PLOTS_PATH += f"_testsamples{TEST_SAMPLES}"
    results_file_name += ".json"
    RESULTS_SUMMARY_PATH = f'../results/verification/{results_file_name}'
    
    with open(RESULTS_SUMMARY_PATH, 'r') as f:
        results_sorted = json.load(f)
    
    pareto_results_sorted = get_pareto_front(results_sorted)
    
    os.makedirs(PLOTS_PATH, exist_ok=True)
    
    plot_results(pareto_results_sorted, include_literature=False)

    for method in ["mtl_ibp", "sabr", "shi", "crown_ibp_nofusion"]:
        print(f"Results for method {method}:")
        plot_results(pareto_results_sorted, methods=[method])
        print(f"Plotting Pareto front for {method} across networks...")
        plot_pareto_front_across_networks(pareto_results_sorted, method)
    
    # Generate comparison plots
    if GENERATE_TESTSAMPLES_COMPARISON:
        # print("Generating test samples comparison plots (10k vs 1k samples)...")
        # plot_testsamples_comparison(results_sorted)
        
        print("Generating multi-sample comparison plots (10000, 5000, 2500 samples)...")
        plot_testsamples_multiple_comparison(results_sorted)
    
    if GENERATE_TIMEOUT_COMPARISON:
        # print("Generating timeout comparison plots (1000s vs 300s timeout)...")
        # plot_timeout_comparison(results_sorted)
        
        print("Generating multi-timeout comparison plots (1000s, 500s, 250s timeout)...")
        plot_timeout_multiple_comparison(results_sorted)