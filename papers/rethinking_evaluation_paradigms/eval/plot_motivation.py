"""
Create motivating plot for introduction:
- Combined Pareto front from all methods on CIFAR-10 eps=2/255
- All literature results from Müller et al. and CTBench
- Annotated points showing improvement and method used
"""

import os
import json
import numpy as np
import pandas as pd
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

# Set up plotting style
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['font.size'] = 14
plt.rcParams['xtick.labelsize'] = 25
plt.rcParams['ytick.labelsize'] = 25
plt.rcParams['axes.labelsize'] = 35
plt.rcParams['legend.fontsize'] = 30

def load_combined_front():
    """Load Pareto front for MTL-IBP and SABR methods on CIFAR-10 with eps=2/255"""
    import json
    
    # Load complete verification results
    with open('../results/verification/summary_results.json', 'r') as f:
        results = json.load(f)
    
    # Filter for CIFAR-10 with eps=2/255
    dataset = "cifar10"
    eps = 2/255
    
    matching = [r for r in results 
               if r['dataset'] == dataset 
               and abs(float(r['eps']) - float(eps)) < 1e-6
               and r['architecture'] == 'cnn7'
               and r['cert_train_method'] in ['mtl_ibp', 'sabr']]
    
    if not matching:
        print("No matching verification results found!")
        return {}, {}
    
    print(f"Found {len(matching)} verification results for CIFAR-10 eps=2/255")
    
    # Separate points by method and compute Pareto front for each
    mtl_ibp_all = np.array([[r['clean_classification_accuracy'], r['certified_accuracy']] 
                             for r in matching if r['cert_train_method'] == 'mtl_ibp'])
    sabr_all = np.array([[r['clean_classification_accuracy'], r['certified_accuracy']] 
                         for r in matching if r['cert_train_method'] == 'sabr'])
    
    def compute_pareto_front(points):
        """Compute Pareto front from points array"""
        if len(points) == 0:
            return []
        pareto = []
        sorted_indices = np.argsort(points[:, 0])[::-1]
        max_cert_acc = -np.inf
        for idx in sorted_indices:
            cert_acc = points[idx, 1]
            if cert_acc > max_cert_acc:
                max_cert_acc = cert_acc
                pareto.append(tuple(points[idx]))
        return sorted(pareto, key=lambda p: p[0])
    
    mtl_ibp_front = compute_pareto_front(mtl_ibp_all)
    sabr_front = compute_pareto_front(sabr_all)
    
    print(f"MTL-IBP Pareto front has {len(mtl_ibp_front)} points")
    for i, point in enumerate(mtl_ibp_front):
        print(f"  {i+1}. ({point[0]:.1f}%, {point[1]:.1f}%)")
    
    print(f"SABR Pareto front has {len(sabr_front)} points")
    for i, point in enumerate(sabr_front):
        print(f"  {i+1}. ({point[0]:.1f}%, {point[1]:.1f}%)")
    
    return mtl_ibp_front, sabr_front


def create_motivation_plot():
    """Create the motivating plot"""
    
    # Literature results from plot.py: (cert_acc, nat_acc, name)
    # Format: (certified_accuracy, natural_accuracy, name)
    all_literature_results = [
        # Original results - only SABR and MTL-IBP
        (62.84, 79.24, 'SABR', 'Müller et al.'),
        (63.24, 80.11, 'MTL-IBP', 'De Palma et al.'),
        # CTBench results - only SABR and MTL-IBP
        (63.61, 77.86, 'SABR', 'CTBench'),
        (64.41, 78.82, 'MTL-IBP', 'CTBench'),
    ]
    
    # Load our discovered fronts (all points for each method)
    mtl_ibp_points, sabr_points = load_combined_front()
    
    if not mtl_ibp_points and not sabr_points:
        print("Could not load Pareto fronts!")
        return
    
    # Create figure with consistent design from plot.py
    sns.set_style("darkgrid")
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Get palette (Set2 with 8 colors)
    palette = sns.color_palette("Set2", 8)
    
    # Color assignments for methods - use more distinct colors
    mtl_color = palette[0]       # Red
    sabr_color = palette[1]      # Cyan/Teal
    
    # Plot MTL-IBP front
    if mtl_ibp_points:
        mtl_front_array = np.array(mtl_ibp_points)
        mtl_x = mtl_front_array[:, 1]  # certified accuracy
        mtl_y = mtl_front_array[:, 0]  # natural accuracy
        
        # Scatter plot for MTL-IBP front
        ax.scatter(mtl_x, mtl_y, s=500, color=mtl_color, edgecolor='black', 
                  linewidth=2.5, alpha=0.95, zorder=2)
        
        # Connect MTL-IBP points
        if len(mtl_ibp_points) > 1:
            points_sorted = sorted(zip(mtl_x, mtl_y))
            mtl_x_sorted, mtl_y_sorted = zip(*points_sorted)
            ax.plot(mtl_x_sorted, mtl_y_sorted, color=mtl_color, linewidth=3.5, alpha=0.6, zorder=1)
        
        # Shade MTL-IBP hypervolume in red
        mtl_certs_sorted = np.array([p[0] for p in points_sorted])
        mtl_nats_sorted = np.array([p[1] for p in points_sorted])
        mtl_suffix_max = np.maximum.accumulate(mtl_nats_sorted[::-1])[::-1]
        
        mtl_vertices_x = [0.0, 0.0]
        mtl_vertices_y = [0.0, float(mtl_suffix_max[0])]
        for i in range(len(mtl_certs_sorted)):
            mtl_vertices_x.append(float(mtl_certs_sorted[i]))
            mtl_vertices_y.append(float(mtl_suffix_max[i]))
        mtl_vertices_x.extend([float(mtl_certs_sorted[-1]), 0.0])
        mtl_vertices_y.extend([0.0, 0.0])
        
        ax.fill(mtl_vertices_x, mtl_vertices_y, color=mtl_color, alpha=0.25, zorder=0)
    
    # Plot SABR front
    if sabr_points:
        sabr_front_array = np.array(sabr_points)
        sabr_x = sabr_front_array[:, 1]  # certified accuracy
        sabr_y = sabr_front_array[:, 0]  # natural accuracy
        
        # Scatter plot for SABR front
        ax.scatter(sabr_x, sabr_y, s=500, color=sabr_color, edgecolor='black', 
                  linewidth=2.5, alpha=0.95, zorder=2)
        
        # Connect SABR points
        if len(sabr_points) > 1:
            points_sorted = sorted(zip(sabr_x, sabr_y))
            sabr_x_sorted, sabr_y_sorted = zip(*points_sorted)
            ax.plot(sabr_x_sorted, sabr_y_sorted, color=sabr_color, linewidth=3.5, alpha=0.6, zorder=1)
        
        # Shade SABR hypervolume in blue
        sabr_certs_sorted = np.array([p[0] for p in points_sorted])
        sabr_nats_sorted = np.array([p[1] for p in points_sorted])
        sabr_suffix_max = np.maximum.accumulate(sabr_nats_sorted[::-1])[::-1]
        
        sabr_vertices_x = [0.0, 0.0]
        sabr_vertices_y = [0.0, float(sabr_suffix_max[0])]
        for i in range(len(sabr_certs_sorted)):
            sabr_vertices_x.append(float(sabr_certs_sorted[i]))
            sabr_vertices_y.append(float(sabr_suffix_max[i]))
        sabr_vertices_x.extend([float(sabr_certs_sorted[-1]), 0.0])
        sabr_vertices_y.extend([0.0, 0.0])
        
        ax.fill(sabr_vertices_x, sabr_vertices_y, color=sabr_color, alpha=0.25, zorder=0)
    
    # Plot literature points with annotation
    for cert_acc, nat_acc, method, source in all_literature_results:
        lit_color = palette[5]  # orange for all literature
        ax.scatter([cert_acc], [nat_acc], s=500, color=lit_color, edgecolor='black', 
                  linewidth=2.5, alpha=0.95, zorder=2, marker='s')
        
        
        ax.annotate(f'{method}\n({source})',
                    xy=(cert_acc, nat_acc), xytext=(-140, 30),
                    textcoords='offset points', fontsize=20, weight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor=lit_color, alpha=0.7, edgecolor='black', linewidth=1),
                    )

    # Formatting with consistent design
    ax.set_xlabel('Certified Accuracy', fontsize=35, fontweight='bold', labelpad=16)
    ax.set_ylabel('Natural Accuracy', fontsize=35, fontweight='bold', labelpad=16)
    
    # Remove spines
    sns.despine(ax=ax, top=True, right=True, left=False, bottom=False)
    
    # Set ticks
    ax.tick_params(axis='both', which='major', labelsize=25, colors='black', length=8, width=2)
    
    # Set axis limits with specific ranges
    ax.set_xlim(59.8, 64.75)  # Certified accuracy
    ax.set_ylim(77.5, 83.4)  # Natural accuracy
    
    # Create legend with method colors (ours) and literature marker
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='MTL-IBP (ours)', 
               markerfacecolor=mtl_color, markersize=15, markeredgecolor='black', linewidth=2),
        Line2D([0], [0], marker='o', color='w', label='SABR (ours)', 
               markerfacecolor=sabr_color, markersize=15, markeredgecolor='black', linewidth=2),
        Line2D([0], [0], marker='s', color='w', label='Literature', 
               markerfacecolor=palette[5], markersize=15, markeredgecolor='black', linewidth=2),
    ]
    ax.legend(handles=legend_elements, fontsize=22, frameon=True, loc='lower left')
    
    # Tight layout and save
    plt.tight_layout()
    
    # Save figure
    os.makedirs('plots', exist_ok=True)
    output_path = 'plots/motivation_sabr_cifar10.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', transparent=False)
    print(f"\nSaved motivation plot to: {output_path}")
    
    # Also save as PNG
    output_path_png = 'plots/motivation_sabr_cifar10.png'
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight', transparent=False)
    print(f"Also saved as PNG: {output_path_png}")
    
    plt.close()


if __name__ == "__main__":
    print("Creating motivation plot for introduction...")
    create_motivation_plot()
