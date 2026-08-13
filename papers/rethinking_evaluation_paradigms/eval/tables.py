import argparse
import json
from pathlib import Path

from util import get_pareto_front

PAPER_ROOT = Path(__file__).resolve().parents[1]


LITERATURE_RESULTS = {
                    "cifar10": {
                        "0.00784313725490196": [
                            (66.84, 52.85, 'shi'),
                            (71.52, 53.97, 'crown_ibp_nofusion'),
                            (79.24, 62.84, 'sabr'),
                            (80.11, 63.24, 'mtl_ibp'),
                        ],
                        "0.03137254901960784": [
                            (48.94, 34.97, 'shi'),
                            (46.29, 33.38, 'crown_ibp_nofusion'),
                            (52.38, 35.13, 'sabr'),
                            (53.35, 35.44, 'mtl_ibp'),
                        ]
                    },
                    "tinyimagenet": {
                        "0.00392156862745098": [
                            (25.92, 17.87, 'shi'),
                            (25.62, 17.93, 'crown_ibp'),
                            (28.85, 20.46, 'sabr'),
                            (37.56, 26.09, 'mtl_ibp'),
                        ],
                    },
                    'mnist': {
                        "0.3": [
                            (97.67, 93.10, 'shi'),
                            (98.18, 92.98, 'crown_ibp_nofusion'),
                            (98.75, 92.98, 'sabr'),
                            (98.80, 93.62, 'mtl_ibp'),
                        ],
                    }
                }

def generate_latex_table(results_sorted):
    """Generate a LaTeX table from the complete verification results."""
    header = (
        # "\\begin{tabular}{lrrrrrrccc}"
        "\\begin{tabular}{llllllll}"
        "\\toprule\n"
        # "File & Total & Unsat & Sat & Unknown & Miscl. & Errors & Adv. Acc. (\%) & Cert. Acc. (\%) & Clean Acc. (\%) \\\\"
        "Dataset & Architecture & Method & Epsilon & Test Samples & Clean Acc. (\%) & Cert. Acc. (\%) & Adv Acc. (\%) \\\\"
        "\\midrule"
    )
    rows = []
    for res in results_sorted:
        method = res['cert_train_method']
        method_tex = method.replace('_', '\\_')  # Escape underscores for LaTeX
        eps = res["eps"]
        row = (
            f"{res['dataset']} & {res['architecture']} & "
            f"{method_tex} & "
            f"{round(float(eps), 4)} & "
            f"{res['total_samples']} & "
            # f"{res['unsat']} & "
            # f"{res['sat']} & "
            # f"{res['unknown']} & "
            # f"{res['misclassified']} & "
            # f"{res['error']} & "
            f"{res['clean_classification_accuracy']:.2f} & "
            f"{res['certified_accuracy']:.2f} & "
            f"{res['adversarial_accuracy']:.2f} "
            f"\\\\" 
        )
        rows.append(row)
    footer = "\\bottomrule\n\\end{tabular}"
    table = '\n'.join([header] + rows + [footer])
    return table

def bold_if_better(ours, lit, higher_is_better=True):
    try:
        ours_f = float(ours)
        lit_f = float(lit)
        if (higher_is_better and ours_f - lit_f > .5) or (not higher_is_better and lit_f - ours_f > .5):
            return f"\\textbf{{{ours}}}"
        elif (higher_is_better and ours_f > lit_f) or (not higher_is_better and lit_f > ours_f):
            return f"\\underline{{{ours}}}"
        else:
            return ours
    except Exception:
        return ours

def generate_latex_table_with_literature_columns(results_sorted, literature_results):
    """Generate a LaTeX table with both our and literature results as columns for each method/epsilon.
    Results are bold if they are better than the literature result.
    """
    header = (
        "\\begin{tabular}{llcccccc}\n"
        "\\toprule\n"
        " & \\multicolumn{2}{c}{Clean Acc.} & \\multicolumn{2}{c}{Cert. Acc.} & \\multicolumn{2}{c}{Adv Acc.} \\\\\n"
        "Method & Ours & Lit. & Ours & Lit. & Ours & Lit. \\\\\n"
        "\\midrule"
    )
    rows = []
    for res in results_sorted:
        if res['total_samples'] <= 1:
            continue
        method = res['cert_train_method']
        method_tex = method.replace('_', '\\_')  # Escape underscores for LaTeX
        display_eps = f"{float(res['eps']):.4f}"
        eps = res['eps']
        ours_clean = f"{res['clean_classification_accuracy']:.2f}"
        ours_cert = f"{res['certified_accuracy']:.2f}"
        ours_adv = f"{res['adversarial_accuracy']:.2f}"
        literature_match = [
            lit for lit in literature_results[res['dataset']][eps] if lit[2] == method
        ]
        if not literature_match:
            continue
        lit_clean, lit_cert, _ = literature_match[0]

        ours_clean_disp = bold_if_better(ours_clean, lit_clean, higher_is_better=True) if lit_clean is not None else ours_clean
        ours_cert_disp = bold_if_better(ours_cert, lit_cert, higher_is_better=True) if lit_cert is not None else ours_cert
        ours_adv_disp = ours_adv  # we do not compare adversarial accuracy

        lit_clean_disp = f"{lit_clean:.2f}" if lit_clean is not None else "N/A"
        lit_cert_disp = f"{lit_cert:.2f}" if lit_cert is not None else "N/A"

        row = (
            f"{method_tex} & "
            f"{ours_clean_disp} & "
            f"{lit_clean_disp} & "
            f"{ours_cert_disp} & "
            f"{lit_cert_disp} & "
            f"{ours_adv_disp} & "
            f"\\\\" 
        )
        rows.append(row)
    footer = "\\bottomrule\n\\end{tabular}"
    table = '\n'.join([header] + rows + [footer])
    return table

    
def main():
    parser = argparse.ArgumentParser(
        description="Generate LaTeX result tables from a verification summary."
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=PAPER_ROOT / "results" / "verification" / "main" / "summary_results.json",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PAPER_ROOT / "tables" / "main"
    )
    args = parser.parse_args()

    with args.summary.open() as f:
        results_sorted = json.load(f)
    
    pareto_results_sorted = get_pareto_front(results_sorted)
        
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results_table = generate_latex_table(pareto_results_sorted)
    with (args.output_dir / "all_results.tex").open('w') as f:
        f.write(all_results_table)

    for dataset in LITERATURE_RESULTS.keys():
        for eps in LITERATURE_RESULTS[dataset].keys():
            architectures = sorted(
                {
                    res['architecture']
                    for res in pareto_results_sorted
                    if res['dataset'] == dataset and res['eps'] == eps
                }
            )
            for architecture in architectures:
                filtered_results = [
                    res for res in pareto_results_sorted
                    if res['total_samples'] > 1 and res['eps'] == eps and res['dataset'] == dataset and res['architecture'] == architecture
                ]
                table = generate_latex_table_with_literature_columns(filtered_results, LITERATURE_RESULTS)
                
                if not len(set(res['total_samples'] for res in filtered_results)) == 1:
                    print(f"Warning: Multiple different test sample sizes found for {dataset}, {architecture}, eps {eps}. Skipping table generation.")
                    print("Please set TEST_SAMPLES to a fixed value and re-generate the tables.")
                    continue
                
                output_path = args.output_dir / f"results_{dataset}_{architecture}_eps{float(eps):.4f}.tex"
                with output_path.open('w') as f:
                    f.write(table)


if __name__ == "__main__":
    main()
