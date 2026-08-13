"""Submit one complete-verification job per final-front checkpoint.

Edit the scheduler constants below, then run this file from the repository root.
The launcher uses the same final-front loader as the chunked launcher, but
verifies the full evaluation set in one job per checkpoint and writes one
results.json file. Main-paper fronts are used by default; pass --fronts-root
to select another front directory.
"""

import argparse
import json
from pathlib import Path

import torch

import submit_chunked_complete_verification as shared


PAPER_ROOT = Path(__file__).resolve().parents[1]
VERIFICATION_RESULTS_ROOT = PAPER_ROOT / "results" / "verification" / "main"
SUBMITIT_LOG_ROOT = PAPER_ROOT / "logs" / "submitit" / "complete_verification"

DRY_RUN = True
MAX_JOBS_TO_SUBMIT = None
SLURM_PARTITION = shared.SLURM_PARTITION
SLURM_JOB_NAME = "CTRAIN_COMPLETE_VERIFY"
SLURM_ARRAY_PARALLELISM = 13
TIMEOUT_MIN = shared.TIMEOUT_MIN
GPUS_PER_NODE = shared.GPUS_PER_NODE
CPUS_PER_TASK = shared.CPUS_PER_TASK
MEM_GB = shared.MEM_GB
SLURM_ADDITIONAL_PARAMETERS = shared.SLURM_ADDITIONAL_PARAMETERS
SLURM_SETUP = shared.SLURM_SETUP
SLURM_ACCOUNT = shared.SLURM_ACCOUNT


def results_path(job):
    return (
        VERIFICATION_RESULTS_ROOT
        / job["dataset"]
        / job["network"]
        / str(job["eps"])
        / job["method"]
        / job["config_hash"]
    )


def result_complete(job):
    path = results_path(job) / "results.json"
    if not path.exists():
        return False
    try:
        with path.open() as handle:
            results = json.load(handle)
    except (json.JSONDecodeError, OSError):
        return False
    if not isinstance(results, dict):
        return False
    return all(
        isinstance(results.get(str(index)), dict)
        and results[str(index)].get("result") is not None
        for index in range(shared.TEST_SAMPLES)
    )


def discover_jobs():
    checkpoints = shared.discover_selected_checkpoints()
    jobs = [job for job in checkpoints if not result_complete(job)]
    return jobs, len(checkpoints) - len(jobs)


def run_complete(job):
    if shared.WARM_START and result_complete(job):
        print(f"Result already complete: {results_path(job) / 'results.json'}")
        return

    loaders, input_shape, n_classes = shared.build_loaders(
        job["dataset"],
        job["network"],
        shared.DATA_LOADER_BATCH_SIZE,
        shared.DATA_ROOT,
        val_split=False,
    )
    _, test_loader = loaders

    device = torch.device(
        shared.DEVICE
        if torch.cuda.is_available() or shared.DEVICE == "cpu"
        else "cpu"
    )
    model = shared.build_model(job["network"], input_shape, n_classes)
    wrapper = shared.build_wrapper(
        job["method"],
        model=model,
        input_shape=input_shape,
        eps=job["eps"],
        epochs=shared.DEFAULT_NUM_EPOCHS_FOR_WRAPPER,
        device=device,
    )
    wrapper.load_state_dict(
        torch.load(job["checkpoint_path"], map_location=device)
    )
    wrapper.eval()

    timeout, abcrown_batch_size = shared.verification_parameters(
        job["dataset"], job["network"]
    )
    print(
        f"Verifying {job['dataset']}/{job['network']}/{job['method']}/"
        f"eps={job['eps']}/{job['config_hash']} in one job"
    )
    return wrapper.evaluate_complete(
        test_loader,
        test_samples=shared.TEST_SAMPLES,
        timeout=timeout,
        no_cores=shared.NO_CORES,
        abcrown_batch_size=abcrown_batch_size,
        abcrown_config_dict=shared.ABCROWN_CONFIG_DICT,
        results_path=str(results_path(job)),
        warm_start=shared.WARM_START,
    )


def main():
    jobs, finished = discover_jobs()
    pending_jobs = len(jobs)
    print(
        f"Discovered {pending_jobs} unfinished final-front checkpoints; "
        f"skipped {finished} complete checkpoints."
    )
    if MAX_JOBS_TO_SUBMIT is not None:
        if MAX_JOBS_TO_SUBMIT < 0:
            raise ValueError(
                "MAX_JOBS_TO_SUBMIT must be None or a non-negative integer"
            )
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
        submitted = [executor.submit(run_complete, job) for job in jobs]
    print(f"Submitted {len(submitted)} of {pending_jobs} unfinished checkpoints.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fronts-root",
        type=Path,
        default=shared.PARETO_FRONTS_ROOT,
        help=(
            "Directory containing final Pareto-front CSVs or exact "
            "_subselected0.05.txt artifacts (default: main-paper fronts)."
        ),
    )
    parser.add_argument(
        "--hpo-root",
        type=Path,
        default=shared.HPO_RESULTS_ROOT,
        help="Directory containing study folders and their nets/ checkpoints.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=VERIFICATION_RESULTS_ROOT,
        help="Directory for complete-verification results.",
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Submit the displayed jobs instead of performing a dry run.",
    )
    cli_args = parser.parse_args()
    shared.PARETO_FRONTS_ROOT = cli_args.fronts_root.resolve()
    shared.HPO_RESULTS_ROOT = cli_args.hpo_root.resolve()
    VERIFICATION_RESULTS_ROOT = cli_args.results_root.resolve()
    if cli_args.submit:
        DRY_RUN = False
    main()
