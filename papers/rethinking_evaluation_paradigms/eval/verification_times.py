import json
import os
import numpy as np
from util import get_pareto_front


PAPER_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def resolve_paper_path(path):
    if os.path.isabs(path):
        return path
    if path.startswith("../"):
        return os.path.abspath(os.path.join(os.path.dirname(__file__), path))
    return os.path.abspath(os.path.join(PAPER_ROOT, path))


def get_verification_times(results_file_path, artificial_timeout=1000000):
    """
    Extract average and total verification times from a summary results file.
    """
    if not os.path.exists(results_file_path):
        print(f"File not found: {results_file_path}")
        return None

    with open(results_file_path, 'r') as f:
        results = json.load(f)

    results_pareto = results

    aggregated_results = []

    for res in results_pareto:
        # Resolve the path to the individual results.json
        # The 'file' path in summary_results.json is relative to the results/verification dir or similar
        # Based on tables.py: RESULTS_SUMMARY_PATH = f'../results/verification/{results_file_name}'
        # So 'file' path like "../results/verification/cifar10/..." needs to be adjusted.
        
        raw_file_path = resolve_paper_path(res['file'])

        if not os.path.exists(raw_file_path):
            print(f"Raw results file not found: {raw_file_path}")
            continue

        with open(raw_file_path, 'r') as f:
            raw_data = json.load(f)

        times = [v['running_time'] for k, v in raw_data.items() if isinstance(v, dict) and 'running_time' in v]
        
        # Filter out crashed results which have extremely high running times (e.g., 10^10)
        times = [t for t in times if t < 1e9]

        for idx, time in enumerate(times):
            if time > artificial_timeout:
                times[idx] = artificial_timeout
        
        if not times:
            continue

        total_time_s = sum(times)
        avg_time_s = np.mean(times)
        total_time_h = total_time_s / 3600.0
        
        if avg_time_s > 1000 or total_time_h > 10000:
            print(f"DEBUG: Unusual times for {raw_file_path}: avg={avg_time_s}, total_h={total_time_h}, num_samples={len(times)}")

        entry = {
            'dataset': res['dataset'],
            'architecture': res['architecture'],
            'method': res['cert_train_method'],
            'eps': res['eps'],
            'avg_time_s': avg_time_s,
            'total_time_h': total_time_h,
            'timeout': res.get('timeout', '1000') # Default timeout is 1000
        }
        aggregated_results.append(entry)

    return aggregated_results

def generate_time_table(aggregated_results, timeout_val):
    """
    Generate a LaTeX table for verification times.
    """
    header = (
        "\\begin{tabular}{llllcc}\n"
        "\\toprule\n"
        f"\\multicolumn{{6}}{{c}}{{{timeout_val}s Timeout Verification Times}} \\\\\n"
        "\\midrule\n"
        "Dataset & Architecture & Epsilon & Method & Avg. Time (s) & Total Time (h) \\\\\n"
        "\\midrule"
    )
    
    rows = []
    # Sort by dataset, architecture, epsilon, then method
    sorted_res = sorted(aggregated_results, key=lambda x: (x['dataset'], x['architecture'], float(x['eps']), x['method']))
    
    for res in sorted_res:
        method_tex = res['method'].replace('_', '\\_')
        arch_tex = res['architecture'].replace('_', '\\_')
        eps_val = f"{float(res['eps']):.4f}"
        row = (
            f"{res['dataset']} & {arch_tex} & {eps_val} & {method_tex} & "
            f"{res['avg_time_s']:.2f} & "
            f"{res['total_time_h']:.2f} \\\\"
        )
        rows.append(row)
        
    footer = "\\bottomrule\n\\end{tabular}"
    return "\n".join([header] + rows + [footer])

def main():
    base_results_path = os.path.join(PAPER_ROOT, "results", "verification")
    output_dir = os.path.join(PAPER_ROOT, "eval", "tables_verification_times")
    os.makedirs(output_dir, exist_ok=True)

    timeouts = [1000, 250, 300, 100]
    
    for timeout in timeouts:
        if timeout == 1000:
            file_name = "summary_results.json"
        else:
            file_name = f"summary_results_timeout{timeout}.json"
            
        full_path = os.path.join(base_results_path, file_name)
        print(f"Processing {full_path}...")
        
        results = get_verification_times(full_path, artificial_timeout=timeout)
        if not results:
            print(f"No results for timeout {timeout}")
            continue
            
        # Group by dataset, architecture, epsilon, and method
        summary = {}
        for r in results:
            key = (r['dataset'], r['architecture'], r['eps'], r['method'])
            if key not in summary:
                summary[key] = []
            summary[key].append(r)
            
        final_summary = []
        for key, group in summary.items():
            final_summary.append({
                'dataset': key[0],
                'architecture': key[1],
                'eps': key[2],
                'method': key[3],
                'avg_time_s': np.mean([g['avg_time_s'] for g in group]),
                'total_time_h': np.sum([g['total_time_h'] for g in group])
            })
            
        table_tex = generate_time_table(final_summary, timeout)
        
        output_file = os.path.join(output_dir, f"verification_times_timeout{timeout}.tex")
        with open(output_file, 'w') as f:
            f.write(table_tex)
        print(f"Saved table to {output_file}")

if __name__ == "__main__":
    main()
