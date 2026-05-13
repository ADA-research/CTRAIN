# TEST_SAMPLES = np.inf
# for results in appendix that investigate differences across networks
import json
import os
import numpy as np

from util import get_pareto_front

TEST_SAMPLES = np.inf
# TEST_SAMPLES = 1000
ARTIFICIAL_TIMEOUT = np.inf
# for results in appendix that investigate differences across networks
# ARTIFICIAL_TIMEOUT = 300

TEX_PATH = './tables'


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
                            (25.62, 17.93, 'crown_ibp_nofusion'),
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
        # if res['total_samples'] != 10_000:
        if res['total_samples'] == 1: # TODO: REMOVE THIS HACK!
            continue
        method = res['cert_train_method']
        method_tex = method.replace('_', '\\_')  # Escape underscores for LaTeX
        display_eps = f"{float(res['eps']):.4f}"
        eps = res['eps']
        ours_clean = f"{res['clean_classification_accuracy']:.2f}"
        ours_cert = f"{res['certified_accuracy']:.2f}"
        ours_adv = f"{res['adversarial_accuracy']:.2f}"
        lit_clean = [lit for lit in LITERATURE_RESULTS[res['dataset']][eps] if lit[2] == method][0][0]
        lit_cert = [lit for lit in LITERATURE_RESULTS[res['dataset']][eps] if lit[2] == method][0][1]

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

    
if __name__ == "__main__":
    results_file_name = "summary_results"
    if ARTIFICIAL_TIMEOUT != np.inf:
        results_file_name += f"_timeout{ARTIFICIAL_TIMEOUT}"
        TEX_PATH += f"_timeout{ARTIFICIAL_TIMEOUT}"
    if TEST_SAMPLES != np.inf:
        results_file_name += f"_testsamples{TEST_SAMPLES}"
        TEX_PATH += f"_testsamples{TEST_SAMPLES}"
    results_file_name += ".json"
    RESULTS_SUMMARY_PATH = f'../results/verification/{results_file_name}'
    
    with open(RESULTS_SUMMARY_PATH, 'r') as f:
        results_sorted = json.load(f)
    
    pareto_results_sorted = get_pareto_front(results_sorted)
        
    os.makedirs(TEX_PATH, exist_ok=True)
    
    all_results_table = generate_latex_table(pareto_results_sorted)
    with open(f"{TEX_PATH}/all_results.tex", 'w') as f:
        f.write(all_results_table)

    for dataset in LITERATURE_RESULTS.keys():
        for eps in LITERATURE_RESULTS[dataset].keys():
            for architecture in set(res['architecture'] for res in pareto_results_sorted if res['dataset'] == dataset and res['eps'] == eps):
                filtered_results = [
                    res for res in pareto_results_sorted
                    if res['total_samples'] > 1 and res['eps'] == eps and res['dataset'] == dataset and res['architecture'] == architecture
                ]
                table = generate_latex_table_with_literature_columns(filtered_results, LITERATURE_RESULTS)
                
                if not len(set(res['total_samples'] for res in filtered_results)) == 1:
                    print(f"Warning: Multiple different test sample sizes found for {dataset}, {architecture}, eps {eps}. Skipping table generation.")
                    print("Please set TEST_SAMPLES to a fixed value and re-generate the tables.")
                    continue
                
                with open(f"{TEX_PATH}/results_{dataset}_{architecture}_eps{float(eps):.4f}.tex", 'w') as f:
                    f.write(table)
            