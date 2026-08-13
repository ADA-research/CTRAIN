GROUPING_KEYS = ("dataset", "architecture", "eps", "cert_train_method")
OBJECTIVE_KEYS = ("certified_accuracy", "clean_classification_accuracy")


def dominates(left, right):
    """Return whether ``left`` strictly Pareto-dominates ``right``."""
    return all(a >= b for a, b in zip(left, right)) and any(
        a > b for a, b in zip(left, right)
    )


def pareto_front(results, objective_values=None):
    """Return nondominated items when every supplied objective is maximised."""
    if objective_values is None:
        objective_values = lambda result: tuple(
            result[key] for key in OBJECTIVE_KEYS
        )
    return [
        result
        for result in results
        if not any(
            other is not result
            and dominates(objective_values(other), objective_values(result))
            for other in results
        )
    ]


def hypervolume_2d(items, objective_values, reference_point=(0.0, 0.0)):
    """Return dominated hypervolume for a two-objective maximisation front."""
    ref_x, ref_y = reference_point
    points = sorted(
        {
            (max(float(x), ref_x), max(float(y), ref_y))
            for x, y in map(objective_values, items)
            if x > ref_x and y > ref_y
        },
        reverse=True,
    )
    volume = 0.0
    best_y = ref_y
    for x, y in points:
        if y > best_y:
            volume += (x - ref_x) * (y - best_y)
            best_y = y
    return volume


def get_pareto_front(results):
    """Filter Pareto-optimal configurations within each experimental group."""
    groups = {}
    for result in results:
        key = tuple(result[name] for name in GROUPING_KEYS)
        groups.setdefault(key, []).append(result)

    pareto_results = []
    for group_results in groups.values():
        pareto_results.extend(pareto_front(group_results))

    pareto_results.sort(
        key=lambda item: (
            item["dataset"],
            item["architecture"],
            float(item["eps"]),
            item["cert_train_method"],
            item["certified_accuracy"],
            item.get("hash", ""),
        )
    )
    print(f"Pareto Optimal Results ({len(pareto_results)} points)")
    return pareto_results
