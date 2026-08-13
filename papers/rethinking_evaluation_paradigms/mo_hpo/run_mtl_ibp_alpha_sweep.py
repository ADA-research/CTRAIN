import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from CTRAIN.util import seed_ctrain
from run_hpo import build_loaders, build_model


EVAL_DIR = Path(__file__).resolve().parents[1] / "eval"
sys.path.insert(0, str(EVAL_DIR))
from util import hypervolume_2d, pareto_front  # noqa: E402
from front_utils import configuration_hash  # noqa: E402


# Sweep setup. Change these values in code instead of adding CLI flags.
NETWORK = "cnn7"
N_ALPHA_VALUES = 20
MIN_ALPHA = 1e-6
MAX_ALPHA = 1.0

EPS_DEFAULTS = {
    2 / 255: {
        "seeds": [42],
        "batch_size": 128,
        "epochs": 160,
        "learning_rate": 5e-4,
        "warm_up_epochs": 1,
        "ramp_up_epochs": 80,
        "lr_decay_kwargs": {"milestones": [120, 140], "gamma": 0.2},
        "l1_reg_weight": 3e-6,
        "shi_reg_weight": 0.5,
        "train_eps_factor": 1,
        "pgd_steps": 8,
        "pgd_alpha": 0.25,
        "pgd_eps_factor": 2.1,
        "default_mtl_ibp_alpha": 0.004,
    },
    8 / 255: {
        "seeds": [42],
        "batch_size": 128,
        "epochs": 260,
        "learning_rate": 5e-4,
        "warm_up_epochs": 1,
        "ramp_up_epochs": 80,
        "lr_decay_kwargs": {"milestones": [180, 220], "gamma": 0.2},
        "l1_reg_weight": 1e-7,
        "shi_reg_weight": 0.5,
        "train_eps_factor": 1,
        "pgd_steps": 1,
        "pgd_alpha": 10,
        "pgd_eps_factor": 1,
        "default_mtl_ibp_alpha": 0.5,
    },
}

EVAL_SAMPLES = 10_000
VAL_SPLIT = False
DATA_ROOT = "data"
OUTPUT_ROOT = Path("papers/rethinking_evaluation_paradigms/results/hpo/alpha_sweep")
REFERENCE_POINT = (0.0, 0.0)

SHI_REG_DECAY = True
PGD_RESTARTS = 1
LOCK_STALE_AFTER_SECONDS = 60 * 60


def parse_eps(value):
    aliases = {
        "2/255": 2 / 255,
        "8/255": 8 / 255,
    }
    if value in aliases:
        return aliases[value]
    return float(value)


def eps_tag(eps):
    numerator = round(eps * 255)
    if abs(eps - numerator / 255) < 1e-12:
        return f"{numerator}_255"
    return f"{eps:.6g}".replace(".", "_")


def defaults_for_eps(eps):
    for configured_eps, defaults in EPS_DEFAULTS.items():
        if abs(eps - configured_eps) < 1e-12:
            return defaults
    raise ValueError(f"No defaults configured for eps={eps}. Add them to EPS_DEFAULTS.")


def alpha_values(default_alpha, n_values=N_ALPHA_VALUES):
    if not 0 < default_alpha <= MAX_ALPHA:
        raise ValueError(f"default_alpha must be in (0, {MAX_ALPHA}], got {default_alpha}")
    if n_values < 3:
        raise ValueError("n_values must be at least 3 to include lower, default, and upper alpha values.")

    n_lower = (n_values - 1) // 2
    n_upper = n_values - 1 - n_lower
    lower = np.geomspace(MIN_ALPHA, default_alpha, n_lower + 1)[:-1]
    upper = np.geomspace(default_alpha, MAX_ALPHA, n_upper + 1)
    return sorted(set([*lower.tolist(), *upper.tolist()]))


def write_csv(path, records):
    fieldnames = [
        "eps",
        "seed",
        "alpha",
        "nat_acc",
        "cert_acc",
        "adv_acc",
        "config_hash",
        "checkpoint_path",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.{os.getpid()}.tmp")
    with tmp_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({name: record.get(name) for name in fieldnames})
    tmp_path.replace(path)


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.{os.getpid()}.tmp")
    with tmp_path.open("w") as handle:
        json.dump(payload, handle, indent=2)
    tmp_path.replace(path)


def read_completed_records(run_dir):
    records_dir = run_dir / "records"
    if not records_dir.exists():
        return []

    records = []
    for record_path in sorted(records_dir.glob("*.json")):
        with record_path.open() as handle:
            records.append(json.load(handle))
    return records


def acquire_lock(lock_path, poll_interval=1.0):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if time.time() - lock_path.stat().st_mtime > LOCK_STALE_AFTER_SECONDS:
                release_lock(lock_path)
                continue
            time.sleep(poll_interval)
            continue
        with os.fdopen(fd, "w") as handle:
            handle.write(str(os.getpid()))
        return lock_path


def release_lock(lock_path):
    try:
        lock_path.unlink()
    except FileNotFoundError:
        pass


def config_for_hash(eps, seed, alpha, defaults):
    return {
        "dataset": "cifar10",
        "network": NETWORK,
        "method": "mtl_ibp_alpha_sweep",
        "eps": eps,
        "seed": seed,
        "epochs": defaults["epochs"],
        "alpha": alpha,
        "batch_size": defaults["batch_size"],
        "train_eps_factor": defaults["train_eps_factor"],
        "learning_rate": defaults["learning_rate"],
        "warm_up_epochs": defaults["warm_up_epochs"],
        "ramp_up_epochs": defaults["ramp_up_epochs"],
        "lr_decay_kwargs": defaults["lr_decay_kwargs"],
        "l1_reg_weight": defaults["l1_reg_weight"],
        "shi_reg_weight": defaults["shi_reg_weight"],
        "pgd_steps": defaults["pgd_steps"],
        "pgd_alpha": defaults["pgd_alpha"],
        "pgd_restarts": PGD_RESTARTS,
        "pgd_eps_factor": defaults["pgd_eps_factor"],
    }


def build_alpha_wrapper(model, input_shape, eps, alpha, defaults, device):
    from CTRAIN.model_wrappers import MTLIBPModelWrapper

    return MTLIBPModelWrapper(
        model=model,
        input_shape=input_shape,
        eps=eps,
        num_epochs=defaults["epochs"],
        device=device,
        train_eps_factor=defaults["train_eps_factor"],
        lr=defaults["learning_rate"],
        warm_up_epochs=defaults["warm_up_epochs"],
        ramp_up_epochs=defaults["ramp_up_epochs"],
        lr_decay_kwargs=defaults["lr_decay_kwargs"],
        l1_reg_weight=defaults["l1_reg_weight"],
        shi_reg_weight=defaults["shi_reg_weight"],
        shi_reg_decay=SHI_REG_DECAY,
        pgd_steps=defaults["pgd_steps"],
        pgd_alpha=defaults["pgd_alpha"],
        pgd_restarts=PGD_RESTARTS,
        pgd_eps_factor=defaults["pgd_eps_factor"],
        mtl_ibp_alpha=alpha,
    )


def run_point(eps, seed, alpha, defaults, train_loader, eval_loader, input_shape, n_classes, device, run_dir):
    seed_ctrain(seed=seed)
    model = build_model(NETWORK, input_shape, n_classes)
    wrapper = build_alpha_wrapper(model, input_shape, eps, alpha, defaults, device)

    wrapper.train_model(train_loader=train_loader)
    wrapper.eval()
    nat_acc, cert_acc, adv_acc = wrapper.evaluate(eval_loader, test_samples=EVAL_SAMPLES)

    config = config_for_hash(eps, seed, alpha, defaults)
    config_hash = configuration_hash(config)
    checkpoint_path = run_dir / "nets" / f"{config_hash}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(wrapper.state_dict(), checkpoint_path)

    return {
        "eps": float(eps),
        "seed": int(seed),
        "alpha": float(alpha),
        "nat_acc": float(nat_acc),
        "cert_acc": float(cert_acc),
        "adv_acc": float(adv_acc),
        "config_hash": config_hash,
        "checkpoint_path": str(checkpoint_path),
    }


def update_outputs(run_dir, eps, defaults, alphas):
    lock_path = acquire_lock(run_dir / ".aggregate.lock")
    try:
        all_records = read_completed_records(run_dir)
        write_csv(run_dir / "alpha_sweep.csv", all_records)

        objective_values = lambda record: (record["nat_acc"], record["cert_acc"])
        front = sorted(
            pareto_front(all_records, objective_values),
            key=objective_values,
            reverse=True,
        )
        hv = hypervolume_2d(front, objective_values, REFERENCE_POINT)
        write_csv(run_dir / "pareto_front.csv", front)

        summary = {
            "eps": float(eps),
            "epochs": defaults["epochs"],
            "seeds": defaults["seeds"],
            "alphas": [float(alpha) for alpha in alphas],
            "default_mtl_ibp_alpha": defaults["default_mtl_ibp_alpha"],
            "num_points": len(all_records),
            "num_expected_points": len(alphas) * len(defaults["seeds"]),
            "num_pareto_points": len(front),
            "reference_point": REFERENCE_POINT,
            "hypervolume": hv,
            "alpha_sweep_csv": str(run_dir / "alpha_sweep.csv"),
            "pareto_front_csv": str(run_dir / "pareto_front.csv"),
        }
        write_json(run_dir / "summary.json", summary)
    finally:
        release_lock(lock_path)
    return all_records, front, hv


def main():
    parser = argparse.ArgumentParser(
        description="Run a fixed MTL-IBP alpha sweep on CIFAR-10 and report Pareto-front hypervolume."
    )
    parser.add_argument("eps", help="Perturbation radius. Use 2/255, 8/255, or a float.")
    parser.add_argument(
        "--alpha-index",
        type=int,
        default=None,
        help="Run only this alpha index from the configured alpha grid.",
    )
    args = parser.parse_args()

    eps = parse_eps(args.eps)
    defaults = defaults_for_eps(eps)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders, input_shape, n_classes = build_loaders(
        "cifar10", NETWORK, defaults["batch_size"], DATA_ROOT, val_split=VAL_SPLIT
    )
    if VAL_SPLIT:
        train_loader, eval_loader, _ = loaders
    else:
        train_loader, eval_loader = loaders

    run_dir = OUTPUT_ROOT / f"cifar10_{NETWORK}_mtl_ibp_alpha_sweep_{eps_tag(eps)}"
    alphas = alpha_values(defaults["default_mtl_ibp_alpha"])
    if args.alpha_index is not None:
        if not 0 <= args.alpha_index < len(alphas):
            raise ValueError(
                f"--alpha-index must be in [0, {len(alphas) - 1}], got {args.alpha_index}"
            )
        selected_alphas = [alphas[args.alpha_index]]
    else:
        selected_alphas = alphas

    print(f"Configured alpha sweep with {len(alphas)} values: {alphas}")
    print(f"Running {len(selected_alphas)} alpha value(s): {selected_alphas}")
    for seed in defaults["seeds"]:
        for alpha in selected_alphas:
            print(f"Running CIFAR-10 {NETWORK} MTL-IBP eps={eps} seed={seed} alpha={alpha}")
            record = run_point(
                eps,
                seed,
                alpha,
                defaults,
                train_loader,
                eval_loader,
                input_shape,
                n_classes,
                device,
                run_dir,
            )
            write_json(run_dir / "records" / f"{record['config_hash']}.json", record)

    all_records, front, hv = update_outputs(run_dir, eps, defaults, alphas)
    print(
        f"eps={eps}: completed points={len(all_records)}/{len(alphas) * len(defaults['seeds'])}, "
        f"Pareto points={len(front)}, hypervolume={hv:.6f}"
    )


if __name__ == "__main__":
    main()
