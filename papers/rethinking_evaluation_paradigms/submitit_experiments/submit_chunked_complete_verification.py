"""
Submit chunked complete verification jobs for final-front MO-HPO checkpoints.

Edit the constants below, then run this file from the CTRAIN repository root:

    python papers/rethinking_evaluation_paradigms/submitit_experiments/submit_chunked_complete_verification.py

The script searches for Optuna studies under HPO_RESULTS_ROOT, but accepts only
checkpoints listed by the final, subselected Pareto-front artifact.
Every study must have a sibling nets/ directory containing checkpoints named
{config_hash}.pt. Each submitted job verifies one selected checkpoint on one
dataset slice and writes a chunk-local result file, so jobs for the same
checkpoint do not race on one JSON file. Auditing atomically consolidates every
completed instance into an authoritative results.json. Later submissions may
use a different chunk size and schedule only the still-missing indices.
"""

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
PAPER_ROOT = THIS_FILE.parents[1]
REPO_ROOT = PAPER_ROOT.parents[1]
MO_HPO_DIR = PAPER_ROOT / "mo_hpo"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(PAPER_ROOT))

import optuna
import torch
from mo_hpo.run_hpo import build_loaders, build_model, build_wrapper  # noqa: E402
from mo_hpo.front_utils import trial_config_hash  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_ROOT = os.environ.get("CTRAIN_DATA_ROOT", str(REPO_ROOT / "data"))
RESULTS_ROOT = Path(os.environ.get("CTRAIN_PAPER_RESULTS_ROOT", str(PAPER_ROOT / "results")))
HPO_RESULTS_ROOT = Path(
    os.environ.get(
        "CTRAIN_PAPER_HPO_ROOT",
        str(RESULTS_ROOT / "hpo" / "main" / "optuna_studies"),
    )
)
VERIFICATION_RESULTS_ROOT = RESULTS_ROOT / "verification" / "main"
SUBMITIT_LOG_ROOT = PAPER_ROOT / "logs" / "submitit" / "complete_verification_chunked"

# Names used to parse study directories. Keep these broader than the active
# filters below, otherwise excluded directories can fail parsing before they
# are skipped.
KNOWN_DATASETS = ["cifar10", "mnist", "gtsrb", "tinyimagenet"]
KNOWN_NETWORKS = ["cnn3", "cnn5", "cnn7", "cnn9", "cnn11", "wide_cnn7", "narrow_cnn7", "resnet18"]
KNOWN_METHODS = ["mtl_ibp", "sabr", "shi", "crown_ibp", "crown_ibp_nofusion"]

# Optional filters. Leave empty to use everything found under HPO_RESULTS_ROOT.
# DATASETS = ["cifar10", "mnist", "tinyimagenet"]
DATASETS = []
NETWORKS = ["cnn3", "cnn5", "cnn7", "cnn9", "cnn11", "wide_cnn7", "narrow_cnn7", "resnet18"]
METHODS = ["mtl_ibp", "sabr", "shi", "crown_ibp", "crown_ibp_nofusion"]
EPS_VALUES = []  # Example: [2 / 255, 8 / 255, 0.3]
SEEDS = []  # Example: [0, 1, 2]

# Main-paper fronts are the default. Override this with --fronts-root for, e.g.,
# validation-split fronts under results/hpo/validation/pareto_fronts.
PARETO_FRONTS_ROOT = RESULTS_ROOT / "hpo" / "main" / "pareto_fronts"

# ---------------------------------------------------------------------------
# Verification arguments
# ---------------------------------------------------------------------------
TEST_SAMPLES = 10_000
INSTANCES_PER_CHUNK = 5000  # Override with --instances-per-chunk.
DATA_LOADER_BATCH_SIZE = 512
DEFAULT_NUM_EPOCHS_FOR_WRAPPER = 1
DEVICE = "cuda"
WARM_START = True

TIMEOUT = 1000
ABCROWN_BATCH_SIZE = 512
NO_CORES = 14
ABCROWN_CONFIG_DICT = {
    "bab": {
        # The custom libc-backed allocator does not release its final buffers
        # between in-process verification instances.
        "hugetensor_allocator": False,
    },
}

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
# Limit how many unfinished chunk jobs this invocation submits. Set to None to
# submit every unfinished chunk discovered by the script.
MAX_JOBS_TO_SUBMIT = None  # Example: 100
SLURM_PARTITION = "KathleenG"
SLURM_JOB_NAME = "CTRAIN_CHUNKED_VERIFY"
SLURM_ARRAY_PARALLELISM = 1
TIMEOUT_MIN = 60 * 24 * 50
GPUS_PER_NODE = 1
CPUS_PER_TASK = 14
MEM_GB = 15.7 * CPUS_PER_TASK
SLURM_ADDITIONAL_PARAMETERS = {"qos": "gpu"}
# SLURM_ADDITIONAL_PARAMETERS = {}
SLURM_SETUP = [
    "module load GCCcore/.13.2.0",
    "module load Python/3.11.5",
    f"export PYTHONPATH={PAPER_ROOT}:{REPO_ROOT}:${{PYTHONPATH}}",
]
# SLURM_SETUP = []
# SLURM_ACCOUNT = "rwth1939"  # Set to your SLURM account name, or None to not specify an account
SLURM_ACCOUNT = None

CHUNK_RESULTS_PATTERN = re.compile(r"results_(\d+)_(\d+)\.json")

def chunk_results_path(dataset, network, eps, method, config_hash):
    return Path(VERIFICATION_RESULTS_ROOT) / dataset / network / str(eps) / method / config_hash


def chunk_results_filename(start_idx, end_idx):
    return f"results_{start_idx:05d}_{end_idx:05d}.json"


def chunk_bounds(path):
    match = CHUNK_RESULTS_PATTERN.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Invalid chunk filename: {path}")
    start_idx, end_idx = map(int, match.groups())
    if start_idx < 0 or end_idx <= start_idx:
        raise ValueError(
            f"Invalid chunk range [{start_idx}, {end_idx}) in {path}"
        )
    return start_idx, end_idx


def _enabled(value, allowed_values):
    return not allowed_values or value in allowed_values


def _float_enabled(value, allowed_values):
    return not allowed_values or any(abs(value - allowed) < 1e-12 for allowed in allowed_values)


def parse_study_dir_name(study_dir):
    """
    Parse directories produced by mo_hpo/run_hpo.py and submitit variants:
    {dataset}_{network}_{method}_{eps}_{seed}
    {dataset}_{network}_{method}_{eps}_{seed}_complete_{True|False}
    """
    name = study_dir.name
    datasets = sorted(set(KNOWN_DATASETS) | set(DATASETS), key=len, reverse=True)
    networks = sorted(set(KNOWN_NETWORKS) | set(NETWORKS), key=len, reverse=True)
    methods = sorted(set(KNOWN_METHODS) | set(METHODS), key=len, reverse=True)

    for dataset in datasets:
        prefix = f"{dataset}_"
        if not name.startswith(prefix):
            continue
        rest = name[len(prefix):]
        for network in networks:
            network_prefix = f"{network}_"
            if not rest.startswith(network_prefix):
                continue
            rest_after_network = rest[len(network_prefix):]
            for method in methods:
                method_prefix = f"{method}_"
                if not rest_after_network.startswith(method_prefix):
                    continue
                suffix = rest_after_network[len(method_prefix):]
                match = re.fullmatch(r"(.+)_([0-9]+)(?:_complete_(True|False))?", suffix)
                if match is None:
                    continue
                return {
                    "dataset": dataset,
                    "network": network,
                    "method": method,
                    "eps": float(match.group(1)),
                    "seed": int(match.group(2)),
                    "complete_verify": None if match.group(3) is None else match.group(3) == "True",
                }
    raise ValueError(f"Could not parse study directory name: {study_dir}")


def subselected_front_path(dataset, network, method, eps):
    stem = f"pareto_front_{method}_{network}_{dataset}_{eps}"
    csv_path = PARETO_FRONTS_ROOT / f"{stem}.csv"
    text_path = PARETO_FRONTS_ROOT / f"{stem}_subselected0.05.txt"
    if csv_path.exists():
        return csv_path
    if text_path.exists():
        return text_path
    raise FileNotFoundError(
        f"Missing final front for {dataset}/{network}/{method}/eps={eps}. "
        f"Expected {csv_path} or {text_path}."
    )


def parse_subselected_hashes(front_path):
    if not front_path.exists():
        raise FileNotFoundError(
            f"Missing Pareto-front artifact: {front_path}. "
            "Run mo_hpo/calculate_fronts.py before submitting complete verification."
        )
    if front_path.suffix == ".txt":
        if not front_path.name.endswith("_subselected0.05.txt"):
            raise ValueError(
                f"Refusing non-subselected legacy front: {front_path}"
            )
        hashes = re.findall(
            r"^Config hash:\s*([0-9a-f]{32})\s*$",
            front_path.read_text(),
            flags=re.MULTILINE,
        )
        if not hashes:
            raise ValueError(f"No config hashes found in {front_path}")
        return set(hashes)
    if front_path.suffix != ".csv":
        raise ValueError(f"Unsupported Pareto-front format: {front_path}")

    with front_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"config_hash", "subselected"}
    if rows and not required.issubset(rows[0]):
        raise ValueError(f"{front_path} lacks required columns {sorted(required)}")
    hashes = [
        row["config_hash"]
        for row in rows
        if row["subselected"].strip().lower() == "true"
    ]
    if not hashes:
        raise ValueError(f"No subselected config hashes found in {front_path}")
    return set(hashes)


def load_study(study_db):
    storage = f"sqlite:///{study_db}"
    summaries = optuna.get_all_study_summaries(storage=storage)
    if not summaries:
        raise RuntimeError(f"No Optuna studies found in {study_db}")
    return optuna.load_study(study_name=summaries[0].study_name, storage=storage)


def result_chunk_complete(results_path, results_filename, start_idx, end_idx):
    results_file = Path(results_path) / results_filename
    if not results_file.exists():
        return False
    try:
        with open(results_file, "r") as handle:
            results = json.load(handle)
    except (json.JSONDecodeError, OSError):
        return False
    for idx in range(start_idx, end_idx):
        item = results.get(str(idx)) or results.get(idx)
        if item is None or item.get("result") is None:
            return False
    return True


def load_completed_results(path, start_idx, end_idx):
    """Load completed entries from a result file, ignoring placeholders."""
    try:
        with path.open() as handle:
            raw_results = json.load(handle)
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"Cannot read {path}: {exc}") from exc
    if not isinstance(raw_results, dict):
        raise ValueError(f"Expected a JSON object in {path}")

    results = {}
    for raw_index, item in raw_results.items():
        try:
            index = int(raw_index)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Non-integer result index {raw_index!r} in {path}"
            ) from exc
        if not start_idx <= index < end_idx:
            raise ValueError(
                f"Index {index} in {path} lies outside its declared range "
                f"[{start_idx}, {end_idx})"
            )
        if not isinstance(item, dict):
            raise ValueError(f"Malformed result at index {index} in {path}")
        if item.get("result") is None:
            continue
        results[index] = item
    return results


def collect_config_results(config_dir):
    """Combine completed entries across the authoritative file and all chunks."""
    merged = {}
    sources = []
    authoritative_path = config_dir / "results.json"
    if authoritative_path.exists():
        sources.append((authoritative_path, 0, TEST_SAMPLES))

    chunk_paths = sorted(config_dir.glob("results_*.json"), key=chunk_bounds)
    for path in chunk_paths:
        start_idx, end_idx = chunk_bounds(path)
        if end_idx > TEST_SAMPLES:
            raise ValueError(
                f"Chunk {path} ends at {end_idx}, beyond the expected "
                f"{TEST_SAMPLES} samples"
            )
        sources.append((path, start_idx, end_idx))

    for path, start_idx, end_idx in sources:
        for index, item in load_completed_results(path, start_idx, end_idx).items():
            if index in merged and merged[index] != item:
                raise ValueError(
                    f"Conflicting results for index {index} in {path}"
                )
            merged[index] = item
    return merged


def write_merged_results(config_dir, merged):
    output_path = config_dir / "results.json"
    serialized = {str(index): merged[index] for index in sorted(merged)}
    if output_path.exists():
        try:
            with output_path.open() as handle:
                if json.load(handle) == serialized:
                    return "unchanged"
        except (json.JSONDecodeError, OSError):
            pass

    temporary_path = config_dir / ".results.json.tmp"
    try:
        with temporary_path.open("w") as handle:
            json.dump(serialized, handle)
            handle.write("\n")
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return "written"


def merge_completed_results():
    config_dirs = sorted(
        {path.parent for path in VERIFICATION_RESULTS_ROOT.rglob("results_*.json")}
        | {path.parent for path in VERIFICATION_RESULTS_ROOT.rglob("results.json")}
    )
    written = unchanged = incomplete = errors = 0
    for config_dir in config_dirs:
        relative = config_dir.relative_to(VERIFICATION_RESULTS_ROOT)
        if len(relative.parts) != 5:
            print(f"MERGE ERROR: unexpected result directory layout: {config_dir}")
            errors += 1
            continue
        try:
            merged = collect_config_results(config_dir)
            if not merged:
                incomplete += 1
                continue
            status = write_merged_results(config_dir, merged)
        except ValueError as exc:
            print(f"MERGE ERROR: {exc}")
            errors += 1
            continue
        written += status == "written"
        unchanged += status == "unchanged"
        incomplete += len(merged) < TEST_SAMPLES
    return written, unchanged, incomplete, errors


def missing_ranges(completed_indices, chunk_size):
    """Return contiguous missing ranges, split to at most ``chunk_size``."""
    missing = sorted(set(range(TEST_SAMPLES)) - set(completed_indices))
    if not missing:
        return []

    ranges = []
    run_start = previous = missing[0]
    for index in missing[1:] + [None]:
        if index is not None and index == previous + 1:
            previous = index
            continue
        run_end = previous + 1
        for start_idx in range(run_start, run_end, chunk_size):
            ranges.append((start_idx, min(run_end, start_idx + chunk_size)))
        if index is not None:
            run_start = previous = index
    return ranges


def discover_selected_checkpoints():
    """Return exactly the checkpoints marked subselected on canonical fronts."""
    checkpoints = []
    seen = set()
    selected_hashes_by_group = {}
    hpo_root = Path(HPO_RESULTS_ROOT)
    if not hpo_root.is_dir():
        raise FileNotFoundError(f"HPO results root does not exist: {hpo_root}")
    study_dbs = sorted(hpo_root.rglob("optuna_study.db"))
    if not study_dbs:
        raise FileNotFoundError(
            f"No optuna_study.db files found under HPO results root: {hpo_root}"
        )

    for study_db in study_dbs:
        study_dir = study_db.parent
        nets_dir = study_dir / "nets"
        if not nets_dir.is_dir():
            print(f"Skipping {study_dir}: missing nets/ directory")
            continue

        try:
            metadata = parse_study_dir_name(study_dir)
        except ValueError as exc:
            print(f"Skipping {study_dir}: {exc}")
            continue
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

        group_key = (metadata["dataset"], metadata["network"], metadata["method"], metadata["eps"])
        if group_key not in selected_hashes_by_group:
            try:
                front_path = subselected_front_path(*group_key)
            except FileNotFoundError as exc:
                print(f"Skipping study group without a final front: {exc}")
                selected_hashes_by_group[group_key] = None
            else:
                selected_hashes_by_group[group_key] = parse_subselected_hashes(
                    front_path
                )
        selected_hashes = selected_hashes_by_group[group_key]
        if selected_hashes is None:
            continue

        study = load_study(study_db)
        for trial in study.trials:
            if trial.values is None:
                continue
            try:
                config_hash = trial_config_hash(trial, metadata["method"], metadata["eps"])
            except KeyError as exc:
                print(f"Skipping trial {trial.number} in {study_dir}: cannot reconstruct config hash, missing {exc}")
                continue
            if config_hash not in selected_hashes:
                continue
            checkpoint_path = nets_dir / f"{config_hash}.pt"
            if not checkpoint_path.exists():
                print(f"Skipping {config_hash}: missing checkpoint {checkpoint_path}")
                continue
            checkpoint_key = (*group_key, config_hash)
            if checkpoint_key in seen:
                continue
            seen.add(checkpoint_key)
            checkpoints.append(
                {
                    **metadata,
                    "study_dir": str(study_dir),
                    "config_hash": config_hash,
                    "checkpoint_path": str(checkpoint_path),
                }
            )
    if not checkpoints:
        raise RuntimeError(
            f"No selected checkpoints were discovered from {len(study_dbs)} "
            f"studies under {hpo_root}. Check --fronts-root and the active filters."
        )
    return checkpoints


def discover_jobs():
    jobs = []
    completed_instances = 0
    for checkpoint in discover_selected_checkpoints():
        config_dir = chunk_results_path(
            checkpoint["dataset"],
            checkpoint["network"],
            checkpoint["eps"],
            checkpoint["method"],
            checkpoint["config_hash"],
        )
        completed = collect_config_results(config_dir) if config_dir.exists() else {}
        completed_instances += len(completed)
        for start_idx, end_idx in missing_ranges(completed, INSTANCES_PER_CHUNK):
            jobs.append({
                **checkpoint,
                "start_idx": start_idx,
                "end_idx": end_idx,
            })
    return jobs, completed_instances


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

    results_path = chunk_results_path(dataset, network, eps, method, config_hash)
    results_filename = chunk_results_filename(start_idx, end_idx)
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
    jobs, completed_instances = discover_jobs()
    pending_jobs = len(jobs)
    print(f"Discovered {pending_jobs} unfinished chunk jobs.")
    if completed_instances:
        print(f"Found {completed_instances} completed verification instances.")
    if MAX_JOBS_TO_SUBMIT is not None:
        if MAX_JOBS_TO_SUBMIT < 0:
            raise ValueError("MAX_JOBS_TO_SUBMIT must be None or a non-negative integer")
        jobs = jobs[:MAX_JOBS_TO_SUBMIT]
        print(f"Submitting at most {MAX_JOBS_TO_SUBMIT} jobs from this invocation.")
    if DRY_RUN:
        for job in jobs[:20]:
            print(job)
        if len(jobs) > 20:
            print(f"... {len(jobs) - 20} more jobs")
        print("Dry run only; pass --submit to submit these jobs.")
        return

    try:
        import submitit
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Submitting cluster jobs requires the optional 'submitit' package."
        ) from exc
    executor = submitit.AutoExecutor(folder=str(SUBMITIT_LOG_ROOT))
    submitit_parameters = {
        "timeout_min": TIMEOUT_MIN,
        "slurm_partition": SLURM_PARTITION,
        "gpus_per_node": GPUS_PER_NODE,
        "slurm_array_parallelism": SLURM_ARRAY_PARALLELISM,
        "cpus_per_task": CPUS_PER_TASK,
        "mem_gb": MEM_GB,
        "slurm_additional_parameters": SLURM_ADDITIONAL_PARAMETERS,
        "slurm_job_name": SLURM_JOB_NAME,
        "slurm_setup": SLURM_SETUP,
    }
    if SLURM_ACCOUNT is not None:
        submitit_parameters["slurm_account"] = SLURM_ACCOUNT
    executor.update_parameters(**submitit_parameters)

    with executor.batch():
        submitted = [executor.submit(run_chunk, job) for job in jobs]
    print(f"Submitted {len(submitted)} jobs out of {pending_jobs} unfinished chunks.")


def audit_results():
    written, unchanged, incomplete, errors = merge_completed_results()
    print(
        "Authoritative result files: "
        f"{written} written, {unchanged} unchanged, "
        f"{incomplete} incomplete configurations, {errors} errors."
    )
    jobs, _ = discover_jobs()
    pending_instances = sum(
        job["end_idx"] - job["start_idx"] for job in jobs
    )
    for job in jobs[:20]:
        print(
            "PENDING RANGE: "
            f"{job['dataset']}/{job['network']}/{job['method']}/"
            f"eps={job['eps']}/{job['config_hash']} "
            f"[{job['start_idx']}, {job['end_idx']})"
        )
    if len(jobs) > 20:
        print(f"... {len(jobs) - 20} more pending ranges")
    print(
        f"Total pending instances: {pending_instances} "
        f"across {len(jobs)} chunks."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hpo-root",
        type=Path,
        default=HPO_RESULTS_ROOT,
        help="Directory containing study folders and their nets/ checkpoints.",
    )
    parser.add_argument(
        "--instances-per-chunk",
        type=int,
        default=INSTANCES_PER_CHUNK,
        help=(
            "Maximum number of currently missing instances per submitted job "
            f"(default: {INSTANCES_PER_CHUNK})."
        ),
    )
    parser.add_argument(
        "--fronts-root",
        type=Path,
        default=PARETO_FRONTS_ROOT,
        help=(
            "Directory containing final Pareto-front CSVs or exact "
            "_subselected0.05.txt artifacts (default: main-paper fronts)."
        ),
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=VERIFICATION_RESULTS_ROOT,
        help="Directory for authoritative results and chunk-local files.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--audit-results", action="store_true")
    mode.add_argument(
        "--submit",
        action="store_true",
        help="Submit the configured unfinished jobs instead of performing a dry run.",
    )
    cli_args = parser.parse_args()
    if cli_args.instances_per_chunk <= 0:
        parser.error("--instances-per-chunk must be positive")
    INSTANCES_PER_CHUNK = cli_args.instances_per_chunk
    HPO_RESULTS_ROOT = cli_args.hpo_root.resolve()
    PARETO_FRONTS_ROOT = cli_args.fronts_root.resolve()
    VERIFICATION_RESULTS_ROOT = cli_args.results_root.resolve()
    if cli_args.audit_results:
        audit_results()
    else:
        if cli_args.submit:
            DRY_RUN = False
        main()
