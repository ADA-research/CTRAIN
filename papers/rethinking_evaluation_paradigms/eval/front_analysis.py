import numpy as np
import json
from util import get_pareto_front

# TEST_SAMPLES = 10_000
TEST_SAMPLES = 1000
# ARTIFICIAL_TIMEOUT = np.inf
# for results in appendix that investigate differences across networks
ARTIFICIAL_TIMEOUT = 300

def count_pareto_front_methods(results_sorted):
    groups = {}
    for result in results_sorted:
        key = (result['dataset'], result['architecture'], result['eps'])
        if key not in groups:
            groups[key] = []
        groups[key].append(result)
        
    # Find pareto front for each group
    for key, group_results in groups.items():
        dataset, architecture, eps = key
        print(f"\nPareto Front Analysis for {dataset}, {architecture}, eps={eps}:")
        
        if not len(set(r['total_samples'] for r in group_results)) == 1:
            print("Warning: Different total_samples in this group, skipping analysis.")
            print("Please set TEST_SAMPLES to a smaller value.")
            continue
        
        # Create pareto front for this group
        pareto_front = []
        for result in group_results:
            is_dominated = False
            for other in group_results:
                if (other['certified_accuracy'] >= result['certified_accuracy'] and 
                    other['clean_classification_accuracy'] >= result['clean_classification_accuracy'] and
                    (other['certified_accuracy'] > result['certified_accuracy'] or 
                    other['clean_classification_accuracy'] > result['clean_classification_accuracy'])):
                    is_dominated = True
                    break
            if not is_dominated:
                pareto_front.append(result)
        
        # Count methods in pareto front
        method_counts = {}
        for result in pareto_front:
            method = result['cert_train_method']
            method_counts[method] = method_counts.get(method, 0) + 1
        
        print("Method counts in Pareto front:")
        for method, count in method_counts.items():
            print(f"{method}: {count} configurations")
            
        print("\nDetailed Pareto front configurations:")
        for result in pareto_front:
            print(f"\nMethod: {result['cert_train_method']}")
            print(f"Clean Classification Accuracy: {result['clean_classification_accuracy']:.2f}%")
            print(f"Certified Accuracy: {result['certified_accuracy']:.2f}%")
            print(f"Adversarial Accuracy: {result['adversarial_accuracy']:.2f}%")

def analyze_network_contributions(results_sorted):
    """Analyze which network architectures contribute to the combined Pareto front for each method."""
    
    # Group results by dataset, method and epsilon
    groups = {}
    for result in results_sorted:
        key = (result['dataset'], result['cert_train_method'], result['eps'])
        if key not in groups:
            groups[key] = []
        groups[key].append(result)
        
    # Find pareto front for each group
    for key, group_results in groups.items():
        dataset, method, eps = key
        print(f"\nPareto Front Analysis for {dataset}, {method}, eps={eps}:")
        
        if not len(set(r['total_samples'] for r in group_results)) == 1:
            print("Warning: Different total_samples in this group, skipping analysis.")
            print("Please set TEST_SAMPLES to a smaller value.")
            continue
        if len(set(r['architecture'] for r in group_results)) == 1:
            print("Only one architecture in this group, skipping analysis.")
            continue
        
        # Create pareto front for this method (combining all architectures)
        pareto_front = []
        for result in group_results:
            is_dominated = False
            for other in group_results:
                if (other['certified_accuracy'] >= result['certified_accuracy'] and 
                    other['clean_classification_accuracy'] >= result['clean_classification_accuracy'] and
                    (other['certified_accuracy'] > result['certified_accuracy'] or 
                    other['clean_classification_accuracy'] > result['clean_classification_accuracy'])):
                    is_dominated = True
                    break
            if not is_dominated:
                pareto_front.append(result)
        
        # Count architectures in pareto front
        arch_counts = {}
        for result in pareto_front:
            arch = result['architecture']
            arch_counts[arch] = arch_counts.get(arch, 0) + 1
        
        print("\nNetwork architecture contributions to Pareto front:")
        for arch, count in sorted(arch_counts.items()):
            print(f"{arch}: {count} configurations ({count/len(pareto_front)*100:.1f}% of Pareto front)")
            
        print("\nDetailed Pareto front configurations:")
        for result in sorted(pareto_front, key=lambda x: (-x['clean_classification_accuracy'], -x['certified_accuracy'])):
            print(f"\nArchitecture: {result['architecture']}")
            print(f"Clean Classification Accuracy: {result['clean_classification_accuracy']:.2f}%")
            print(f"Certified Accuracy: {result['certified_accuracy']:.2f}%")
            print(f"Adversarial Accuracy: {result['adversarial_accuracy']:.2f}%")


if __name__ == "__main__":
    results_file_name = "summary_results"
    if ARTIFICIAL_TIMEOUT != np.inf:
        results_file_name += f"_timeout{ARTIFICIAL_TIMEOUT}"
    if TEST_SAMPLES not in [10_000, np.inf]:
        results_file_name += f"_testsamples{TEST_SAMPLES}"
    results_file_name += ".json"
    RESULTS_SUMMARY_PATH = f'../results/verification/{results_file_name}'
    
    with open(RESULTS_SUMMARY_PATH, 'r') as f:
        results_sorted = json.load(f)
    
    pareto_results_sorted = get_pareto_front(results_sorted)
    count_pareto_front_methods(results_sorted)
    print("\n" + "="*80 + "\n")
    analyze_network_contributions(results_sorted)
