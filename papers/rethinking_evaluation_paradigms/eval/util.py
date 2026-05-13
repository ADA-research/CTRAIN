def get_pareto_front(results_sorted):
    # filter pareto optimal configurations per dataset, architecture, eps and method
    pareto_optimal = []

    # Group by these dimensions
    grouping_keys = ['dataset', 'architecture', 'eps', 'cert_train_method']
    result_groups = {}
    for r in results_sorted:
        key = tuple(r[k] for k in grouping_keys)
        if key not in result_groups:
            result_groups[key] = []
        result_groups[key].append(r)

    # Find pareto optimal points within each group
    for group_results in result_groups.values():
        # For each result, check if any other result dominates it
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
                pareto_optimal.append(result)

    # Replace results_sorted with pareto optimal results
    pareto_results_sorted = sorted(pareto_optimal, key=lambda x: (x['dataset'], x['architecture'], x['eps'], x['cert_train_method'], x['certified_accuracy']))
    print(f"Pareto Optimal Results ({len(pareto_results_sorted)} points)")
    
    return pareto_results_sorted