import glob
import json
import os
import numpy as np

TEST_SAMPLES = 1000
# TEST_SAMPLES = 2500
ARTIFICIAL_TIMEOUT = 1000
# for results in appendix that investigate differences across networks
# ARTIFICIAL_TIMEOUT = 300

# Set random seed for reproducible sampling of test indices
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def parse_results():

    results = []
    
    # Generate random indices for sampling test set (once, used for all results)
    # We'll sample them once we know the total number of available test samples
    test_indices = None
    
    for file in glob.glob("../results/verification/**/results.json", recursive=True):
        with open(file, "r") as f:
            data = json.load(f)
        
        # Initialize test_indices on first file if needed
        if test_indices is None:
            total_test_samples = len(data)
            test_indices = np.random.choice(total_test_samples, size=min(TEST_SAMPLES, total_test_samples), replace=False)
            test_indices = sorted(test_indices)
        
        hash_folder = os.path.basename(os.path.dirname(file))
            
        unsat = 0
        sat = 0
        unknown = 0
        misclassified = 0
        error = 0
        max_running_time = 0
        
        for idx, result in data.items():
            int_idx = int(idx)
            # Only process samples in our randomly selected indices
            if int_idx not in test_indices:
                continue
            
            # Track maximum running time for timeout validation
            if result['running_time'] is not None and result['running_time'] != 10000000000: # this value indicates crashed runs
                max_running_time = max(max_running_time, result['running_time'])
            
            if result['result'] is None:
                unknown += 1
                continue
            if result["running_time"] > ARTIFICIAL_TIMEOUT:
                unknown += 1
                continue
            if result['result'] == 'unsat':
                unsat += 1
            elif result['result'] == 'sat' and result['method'] == 'clean_classification':
                sat += 1
                misclassified += 1
            elif result['result'] == 'sat' and result['method'] != 'clean_classification':
                sat += 1
            elif result['result'] == 'timeout':
                unknown += 1
            elif result['result'] == 'unknown':
                error += 1
                unknown += 1
        
        total_samples = unsat + sat + unknown
        print(file)
        _, _, _, dataset, architecture, eps, cert_train_method, config_hash, _ = file.split('/')
        
        # Validate timeout: if ARTIFICIAL_TIMEOUT is set, check that max_running_time is roughly equal to it
        # This ensures the experiment was run with the correct timeout, not a lower one
        if ARTIFICIAL_TIMEOUT != np.inf:
            # Allow 10% tolerance for the timeout check
            timeout_tolerance = ARTIFICIAL_TIMEOUT * 0.1
            if max_running_time < ARTIFICIAL_TIMEOUT - timeout_tolerance:
                print(f"Skipping {file}: max running time ({max_running_time:.2f}s) is lower than expected timeout ({ARTIFICIAL_TIMEOUT}s). Experiment likely ran with a shorter timeout.")
                continue
        
        if not os.path.exists(f'../results/clean_classification/{dataset}_{architecture}_{cert_train_method}{eps}_{config_hash}_nat_acc.json'):
            print(f"Clean classification results for {dataset}_{architecture}_{cert_train_method}{eps}_{config_hash} not found, skipping.")
            continue
        
        with open(f'../results/clean_classification/{dataset}_{architecture}_{cert_train_method}{eps}_{config_hash}_nat_acc.json', 'r') as f:
            clean_classification_results = json.load(f)
            clean_classification_accuracy = clean_classification_results.get('std_acc', -100) * 100
        
        if cert_train_method == "crown_ibp":
            cert_train_method = "crown_ibp_nofusion"
        if total_samples < TEST_SAMPLES:
            print(f"Warning: Only {total_samples} samples verified for {file}, expected at least {TEST_SAMPLES}.")
            print("We do not include this result in the combined results.")
            continue
        results.append({
            "file": file,
            "dataset": dataset,
            "architecture": architecture,
            "eps": eps,
            "cert_train_method": cert_train_method,
            "hash": hash_folder,
            "total_samples": total_samples,
            "unsat": unsat,
            "sat": sat,
            "unknown": unknown,
            "misclassified": misclassified,
            "error": error,
            "adversarial_accuracy": (total_samples - sat) / total_samples * 100 if total_samples > 0 else 0,
            "certified_accuracy": unsat / total_samples * 100 if total_samples else 0,
            "clean_classification_accuracy": clean_classification_accuracy,
        })
        
        # fix numerical precision errors
        for result in results:
            result['adversarial_accuracy'] = round(result['adversarial_accuracy'], 2)
            result['certified_accuracy'] = round(result['certified_accuracy'], 2)
            result['clean_classification_accuracy'] = round(result['clean_classification_accuracy'], 2)
    
    # Sort results by total_samples descending
    results_sorted = sorted(results, key=lambda x: x["total_samples"], reverse=True)

    for res in results_sorted:
        print(f"File: {res['file']}")
        print(f"COMPLETE RESULT:")
        print(f"\t Total Samples: {res['total_samples']}")
        print(f"\t Unsat: {res['unsat']}, Sat: {res['sat']}, Unknown: {res['unknown']}, Misclassified: {res['misclassified']}, Verification Errors: {res['error']}")
        print(f"\t Adversarial Accuracy: {res['adversarial_accuracy']:.2f}%")
        print(f"\t Certified Accuracy: {res['certified_accuracy']:.2f}%")
        print(f"\t Clean Classification Accuracy: {res['clean_classification_accuracy']:.2f}%")

        print("-" * 40)
        print()
    
    return results_sorted
        
if __name__ == "__main__":
    results = parse_results()
    file_name = "summary_results"
    if ARTIFICIAL_TIMEOUT != 1000:
        file_name += f"_timeout{ARTIFICIAL_TIMEOUT}"
    if TEST_SAMPLES != 10_000:
        file_name += f"_testsamples{TEST_SAMPLES}"
    file_name += ".json"
    print(f"Saving summary results to ../results/verification/{file_name}")
    with open(f"../results/verification/{file_name}", "w") as f:
        json.dump(results, f, indent=4)