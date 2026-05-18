"""
Submit chunked complete verification jobs for MO-HPO checkpoints.

Edit the constants below, then run this file from the CTRAIN repository root:

    python papers/rethinking_evaluation_paradigms/submitit_experiments/submit_chunked_complete_verification.py

The script searches for Optuna studies under HPO_RESULTS_ROOT. Every study must
have a sibling nets/ directory containing checkpoints named {config_hash}.pt.
Each submitted job verifies one checkpoint on one dataset slice and writes a
chunk-local result file, so jobs for the same checkpoint do not race on one JSON
file.
"""

import json
import os
import re
import sys
from pathlib import Path

import optuna
import submitit
import torch

THIS_FILE = Path(__file__).resolve()
PAPER_ROOT = THIS_FILE.parents[1]
REPO_ROOT = PAPER_ROOT.parents[1]
MO_HPO_DIR = PAPER_ROOT / "mo_hpo"
sys.path.insert(0, str(MO_HPO_DIR))

from run_hpo import build_loaders, build_model, build_wrapper  # noqa: E402


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_ROOT = os.environ.get("CTRAIN_DATA_ROOT", str(REPO_ROOT / "data"))
HPO_RESULTS_ROOT = PAPER_ROOT / "results" / "mo_hpo"
VERIFICATION_RESULTS_ROOT = PAPER_ROOT / "results" / "verification_chunked"
SUBMITIT_LOG_ROOT = PAPER_ROOT / "submitit_logs" / "complete_verification_chunked"

# Optional filters. Leave empty to use everything found under HPO_RESULTS_ROOT.
DATASETS = ["cifar10", "mnist", "tinyimagenet"]
NETWORKS = ["cnn3", "cnn5", "cnn7", "cnn9", "cnn11", "wide_cnn7", "narrow_cnn7", "resnet18"]
METHODS = ["mtl_ibp", "sabr", "shi", "crown_ibp", "crown_ibp_nofusion"]
EPS_VALUES = []  # Example: [2 / 255, 8 / 255, 0.3]
SEEDS = []  # Example: [0, 1, 2]

# Which checkpoints from each Optuna study should be verified:
# - "pareto_feasible": Pareto trials satisfying stored constraints, falling back
#   to the unconstrained Pareto front if none are feasible.
# - "pareto": Optuna's full Pareto front.
# - "all_complete": every completed trial with a matching checkpoint.
TRIAL_SELECTION = "pareto_feasible"

# ---------------------------------------------------------------------------
# Verification arguments
# ---------------------------------------------------------------------------
TEST_SAMPLES = 10_000
INSTANCES_PER_CHUNK = 500
DATA_LOADER_BATCH_SIZE = 512
DEFAULT_NUM_EPOCHS_FOR_WRAPPER = 1
DEVICE = "cuda"
WARM_START = True

TIMEOUT = 1000
ABCROWN_BATCH_SIZE = 512
NO_CORES = 14
ABCROWN_CONFIG_DICT = None

# Optional per-dataset/network overrides.
TIMEOUT_OVERRIDES = {
    # ("mnist", "cnn7"): 300,
}
ABCROWN_BATCH_SIZE_OVERRIDES = {
    # ("resnet18",): 256,
    # ("tinyimagenet", "cnn9"): 256,
}

# ---------------------------------------------------------------------------
# submitit / SLURM resources
# ---------------------------------------------------------------------------
DRY_RUN = True
SLURM_PARTITION = "CLUSTER"
SLURM_JOB_NAME = "CTRAIN_CHUNKED_VERIFY"
SLURM_ARRAY_PARALLELISM = 16
TIMEOUT_MIN = 60 * 24 * 7
GPUS_PER_NODE = 1
CPUS_PER_TASK = 14
MEM_GB = 15.7 * 14
SLURM_ADDITIONAL_PARAMETERS = {"qos": "gpu"}
SLURM_SETUP = ["module load CUDA/12.1.1", "module load Python/3.11"]


def _enabled(value, allowed_values):
    return not allowed_values or value in allowed_values


def _float_enabled(value, allowed_values):
    return not allowed_values or any(abs(value - allowed) < 1e-12 for allowed in allowed_values)


def parse_study_dir_name(study_dir):
    """
    Parse directories produced by mo_hpo/run_hpo.py:
    {dataset}_{network}_{method}_{eps}_{seed}
    """
    name = study_dir.name
    for dataset in sorted(DATASETS or ["cifar10", "mnist", "gtsrb", "tinyimagenet"], key=len, reverse=True):
        prefix = f"{dataset}_"
        if not name.startswith(prefix):
            continue
        rest = name[len(prefix):]
        for network in sorted(NETWORKS, key=len, reverse=True):
            network_prefix = f"{network}_"
            if not rest.startswith(network_prefix):
                continue
            rest_after_network = rest[len(network_prefix):]
            for method in sorted(METHODS, key=len, reverse=True):
                method_prefix = f"{method}_"
                if not rest_after_network.startswith(method_prefix):
                    continue
                eps_seed = rest_after_network[len(method_prefix):]
                match = re.fullmatch(r"(.+)_([0-9]+)", eps_seed)
                if match is None:
                    continue
                return {
                    "dataset": dataset,
                    "network": network,
                    "method": method,
                    "eps": float(match.group(1)),
                    "seed": int(match.group(2)),
                }
    raise ValueError(f"Could not parse study directory name: {study_dir}")


def constraints_feasible(trial):
    return all(value <= 0 for value in trial.user_attrs.get("constraints", (0.0, 0.0)))


def dominates(values, other_values):
    return all(value >= other for value, other in zip(values, other_values)) and any(
        value > other for value, other in zip(values, other_values)
    )


def pareto_trials(trials):
    candidates = [trial for trial in trials if trial.values is not None and len(trial.values) == 2]
    return [
        trial for trial in candidates
        if not any(dominates(other.values, trial.values) for other in candidates if other.number != trial.number)
    ]


def selected_trials(study):
    if TRIAL_SELECTION == "all_complete":
        return [trial for trial in study.trials if trial.values is not None]
    front = pareto_trials(study.trials)
    if TRIAL_SELECTION == "pareto":
        return front
    if TRIAL_SELECTION == "pareto_feasible":
        feasible = [trial for trial in front if constraints_feasible(trial)]
        return feasible if feasible else front
    raise ValueError(f"Unknown TRIAL_SELECTION: {TRIAL_SELECTION}")


def load_study(study_db):
    storage = f"sqlite:///{study_db}"
    summaries = optuna.get_all_study_summaries(storage=storage)
    if not summaries:
        raise RuntimeError(f"No Optuna studies found in {study_db}")
    return optuna.load_study(study_name=summaries[0].study_name, storage=storage)


def discover_jobs():
    jobs = []
    for study_db in sorted(Path(HPO_RESULTS_ROOT).rglob("optuna_study.db")):
        study_dir = study_db.parent
        nets_dir = study_dir / "nets"
        if not nets_dir.is_dir():
            print(f"Skipping {study_dir}: missing nets/ directory")
            continue

        metadata = parse_study_dir_name(study_dir)
        if not _enabled(metadata["dataset"], DATASETS):
            continue
        if not _enabled(metadata["network"], NETWORKS):
            continue
        if not _enabled(metadata["method"], METHODS):
            continue
        if not _float_enabled(metadata["eps"], EPS_VALUES):
            continue
        if not _enabled(metadata["seed"], SEEDS):
            continue

        study = load_study(study_db)
        for trial in selected_trials(study):
            config_hash = trial.user_attrs.get("config_hash")
            if config_hash is None:
                print(f"Skipping trial {trial.number} in {study_dir}: missing config_hash")
                continue
            checkpoint_path = nets_dir / f"{config_hash}.pt"
            if not checkpoint_path.exists():
                print(f"Skipping {config_hash}: missing checkpoint {checkpoint_path}")
                continue
            for start_idx in range(0, TEST_SAMPLES, INSTANCES_PER_CHUNK):
                end_idx = min(TEST_SAMPLES, start_idx + INSTANCES_PER_CHUNK)
                jobs.append(
                    {
                        **metadata,
                        "study_dir": str(study_dir),
                        "config_hash": config_hash,
                        "checkpoint_path": str(checkpoint_path),
                        "start_idx": start_idx,
                        "end_idx": end_idx,
                    }
                )
    return jobs


def result_chunk_complete(results_path, results_filename, start_idx, end_idx):
    results_file = Path(results_path) / results_filename
    if not results_file.exists():
        return False
    with open(results_file, "r") as handle:
        results = json.load(handle)
    for idx in range(start_idx, end_idx):
        item = results.get(str(idx)) or results.get(idx)
        if item is None or item.get("result") is None:
            return False
    return True


def verification_parameters(dataset, network):
    timeout = TIMEOUT_OVERRIDES.get((dataset, network), TIMEOUT_OVERRIDES.get((dataset,), TIMEOUT))
    batch_size = ABCROWN_BATCH_SIZE_OVERRIDES.get(
        (dataset, network),
        ABCROWN_BATCH_SIZE_OVERRIDES.get((network,), ABCROWN_BATCH_SIZE),
    )
    return timeout, batch_size


def run_chunk(job):
    dataset = job["dataset"]
    network = job["network"]
    method = job["method"]
    eps = job["eps"]
    start_idx = job["start_idx"]
    end_idx = job["end_idx"]
    config_hash = job["config_hash"]

    results_path = Path(VERIFICATION_RESULTS_ROOT) / dataset / network / str(eps) / method / config_hash
    results_filename = f"results_{start_idx:05d}_{end_idx:05d}.json"
    if WARM_START and result_chunk_complete(results_path, results_filename, start_idx, end_idx):
        print(f"Chunk already complete: {results_path / results_filename}")
        return

    loaders, input_shape, n_classes = build_loaders(
        dataset,
        network,
        DATA_LOADER_BATCH_SIZE,
        DATA_ROOT,
        val_split=False,
    )
    _, test_loader = loaders

    device = torch.device(DEVICE if torch.cuda.is_available() or DEVICE == "cpu" else "cpu")
    model = build_model(network, input_shape, n_classes)
    wrapper = build_wrapper(
        method,
        model=model,
        input_shape=input_shape,
        eps=eps,
        epochs=DEFAULT_NUM_EPOCHS_FOR_WRAPPER,
        device=device,
    )
    wrapper.load_state_dict(torch.load(job["checkpoint_path"], map_location=device))
    wrapper.eval()

    timeout, abcrown_batch_size = verification_parameters(dataset, network)
    print(
        f"Verifying {dataset}/{network}/{method}/eps={eps}/{config_hash} "
        f"chunk [{start_idx}, {end_idx})"
    )
    return wrapper.evaluate_complete(
        test_loader,
        test_samples=TEST_SAMPLES,
        timeout=timeout,
        no_cores=NO_CORES,
        abcrown_batch_size=abcrown_batch_size,
        abcrown_config_dict=ABCROWN_CONFIG_DICT,
        results_path=str(results_path),
        warm_start=WARM_START,
        start_idx=start_idx,
        end_idx=end_idx,
        results_filename=results_filename,
    )


def main():
    jobs = discover_jobs()
    print(f"Discovered {len(jobs)} chunk jobs.")
    if DRY_RUN:
        for job in jobs[:20]:
            print(job)
        if len(jobs) > 20:
            print(f"... {len(jobs) - 20} more jobs")
        print("DRY_RUN=True, not submitting. Set DRY_RUN=False to submit.")
        return

    executor = submitit.AutoExecutor(folder=str(SUBMITIT_LOG_ROOT))
    executor.update_parameters(
        timeout_min=TIMEOUT_MIN,
        slurm_partition=SLURM_PARTITION,
        gpus_per_node=GPUS_PER_NODE,
        slurm_array_parallelism=SLURM_ARRAY_PARALLELISM,
        cpus_per_task=CPUS_PER_TASK,
        mem_gb=MEM_GB,
        slurm_additional_parameters=SLURM_ADDITIONAL_PARAMETERS,
        slurm_job_name=SLURM_JOB_NAME,
        slurm_setup=SLURM_SETUP,
    )

    with executor.batch():
        submitted = [executor.submit(run_chunk, job) for job in jobs]
    print(f"Submitted {len(submitted)} jobs.")


if __name__ == "__main__":
    main()
