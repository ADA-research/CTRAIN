"""Evaluate predictive performance and robustness using attacks and verification.

Adversarial attacks search for counterexamples, incomplete methods certify
samples through bounds, and complete systems resolve harder queries within
a configured budget. Classification accuracies are returned as fractions.
"""
import copy
import hashlib
import json
import tempfile
from pathlib import Path
import os
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset, SequentialSampler
import numpy as np

import shutil

from auto_LiRPA import BoundedModule, PerturbationLpNorm
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from CTRAIN.bound import bound_ibp, bound_crown, bound_crown_ibp, bound_alpha_crown, bound_forward
from CTRAIN.eval.autoattack import eval_autoattack
from CTRAIN.attacks import pgd_attack
from CTRAIN.complete_verification.abCROWN.util import (
    instances_to_vnnlib,
    get_abcrown_standard_conf,
)
from CTRAIN.complete_verification.abCROWN.verify import (
    limited_abcrown_eval,
    abcrown_eval,
)
from CTRAIN.complete_verification.neuralsat import neuralsat_eval, validate_neuralsat_command, validate_neuralsat_options
from CTRAIN.util import export_onnx
from CTRAIN.util.util import preserve_model_state, validate_test_samples

from contextlib import redirect_stdout, contextmanager

import gc


class _BoxPerturbation(PerturbationLpNorm):
    """Preserve the input dtype when LiRPA constructs sparse box coefficients."""
    def init_sparse_linf(self, x, x_L, x_U):
        # Installed LiRPA creates sparse coefficients in the default dtype.
        linear, center, aux = super().init_sparse_linf(x, x_L, x_U)
        linear.lw, linear.uw = linear.lw.to(x), linear.uw.to(x)
        return linear, center, aux


@contextmanager
def suppress_stdout():
    """Temporarily suppress standard output while retaining exceptions."""
    with open(os.devnull, "w") as null:
        with redirect_stdout(null):
            yield

@contextmanager
def redirect_stdout_to_file(filename):
    """Redirect standard output to the specified file for the duration of the context."""
    with open(filename, "w") as file:
        with redirect_stdout(file):
            yield

def write_json_atomic(filename, payload):
    """Write JSON through a temporary file and atomically replace the destination."""
    tmp_filename = f"{filename}.tmp"
    with open(tmp_filename, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_filename, filename)

def _canonical_methods(methods):
    """Normalise method spelling and resolve ZONOTOPE to the FORWARD evaluation method."""
    if isinstance(methods, str):
        method = methods.replace("_", "-")
        return 'FORWARD' if method == 'ZONOTOPE' else method
    return [_canonical_methods(method) if isinstance(method, str) else method for method in methods]

def _to_device_tensor(value, device):
    """Move an existing tensor or construct a value tensor on the requested device."""
    return value.to(device) if torch.is_tensor(value) else torch.as_tensor(value, device=device)

def eval_acc(model, test_loader, test_samples=np.inf, device=None):
    """Compute natural classification accuracy on a prefix of the loader.

    Args:
        model (torch.nn.Module): Classifier returning a batch of class logits.
        test_loader (DataLoader): Loader providing inputs and integer class labels.
        test_samples (int or float): Positive sample budget; infinity evaluates all inputs.
        device (str or torch.device, optional): Evaluation device; None chooses an available
            accelerator or CPU.

    Returns:
        (float): Fraction of inputs classified correctly.

    Evaluation preserves caller module modes and gradient flags, but moves the
    model to the chosen device.
    """
    test_samples = validate_test_samples(test_samples)
    with preserve_model_state(model):
        device = torch.device(device) if device is not None else torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

        model = model.to(device)  # Move model to device
        model.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in test_loader:
                if total >= test_samples:
                    break

                remaining_samples = test_samples - total
                if remaining_samples < len(target):
                    data = data[:remaining_samples]
                    target = target[:remaining_samples]

                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = torch.max(output, 1)

                total += target.size(0)
                correct += (predicted == target).sum().item()

        if not total:
            raise ValueError('Evaluation loader must contain samples')
        return correct / total

def eval_ibp(model, eps, data_loader, n_classes=10, test_samples=np.inf, device="cuda"):
    """Certify local robustness using IBP bounds.

    Interval arithmetic provides inexpensive output enclosures.
    An input is certified when every true-class margin has a positive lower
    bound. Uncertified inputs remain unresolved by this bounding method.

    Args:
        model (auto_LiRPA.BoundedModule): Network to evaluate.
        eps (float or torch.Tensor): Perturbation radius in model input coordinates.
        data_loader (DataLoader): Inputs, labels and feasible input-domain limits.
        n_classes (int): Number of output classes.
        test_samples (int or float): Positive sample budget; infinity evaluates all inputs.
        device (str or torch.device): Evaluation device.

    Returns:
        (tuple): (certified_count, evaluated_count, results), where results maps
            loader-order input indices to outcomes, method names and running times.
    """
    certified, total, _, results = eval_adaptive(
        model, eps, data_loader, n_classes, test_samples, device, methods=['IBP'])
    return int(certified), total, results

def eval_crown_ibp(
    model, eps, data_loader, n_classes=10, test_samples=np.inf, device="cuda"
):
    """Certify local robustness using CROWN-IBP bounds.

    CROWN backward relaxations use IBP intermediate bounds.
    An input is certified when every true-class margin has a positive lower
    bound. Uncertified inputs remain unresolved by this bounding method.

    Args:
        model (auto_LiRPA.BoundedModule): Network to evaluate.
        eps (float or torch.Tensor): Perturbation radius in model input coordinates.
        data_loader (DataLoader): Inputs, labels and feasible input-domain limits.
        n_classes (int): Number of output classes.
        test_samples (int or float): Positive sample budget; infinity evaluates all inputs.
        device (str or torch.device): Evaluation device.

    Returns:
        (tuple): (certified_count, evaluated_count, results), where results maps
            loader-order input indices to outcomes, method names and running times.
    """
    certified, total, _, results = eval_adaptive(
        model, eps, data_loader, n_classes, test_samples, device, methods=['CROWN-IBP'])
    return int(certified), total, results

def eval_crown(
    model, eps, data_loader, n_classes=10, test_samples=np.inf, device="cuda"
):
    """Certify local robustness using CROWN bounds.

    Linear relaxations bound the classification margins throughout the network.
    An input is certified when every true-class margin has a positive lower
    bound. Uncertified inputs remain unresolved by this bounding method.

    Args:
        model (auto_LiRPA.BoundedModule): Network to evaluate.
        eps (float or torch.Tensor): Perturbation radius in model input coordinates.
        data_loader (DataLoader): Inputs, labels and feasible input-domain limits.
        n_classes (int): Number of output classes.
        test_samples (int or float): Positive sample budget; infinity evaluates all inputs.
        device (str or torch.device): Evaluation device.

    Returns:
        (tuple): (certified_count, evaluated_count, results), where results maps
            loader-order input indices to outcomes, method names and running times.
    """
    certified, total, _, results = eval_adaptive(
        model, eps, data_loader, n_classes, test_samples, device, methods=['CROWN'])
    return int(certified), total, results

def eval_alpha_crown(
    model,
    eps,
    data_loader,
    n_classes=10,
    test_samples=np.inf,
    device="cuda",
    alpha_iterations=20,
):
    """Certify local robustness using alpha-CROWN bounds.

    Gradient-based slope optimisation tightens the CROWN relaxations.
    An input is certified when every true-class margin has a positive lower
    bound. Uncertified inputs remain unresolved by this bounding method.

    Args:
        model (auto_LiRPA.BoundedModule): Network to evaluate.
        eps (float or torch.Tensor): Perturbation radius in model input coordinates.
        data_loader (DataLoader): Inputs, labels and feasible input-domain limits.
        n_classes (int): Number of output classes.
        test_samples (int or float): Positive sample budget; infinity evaluates all inputs.
        device (str or torch.device): Evaluation device.
        alpha_iterations (int): Number of optimisation iterations for alpha-CROWN relaxation
            slopes.

    Returns:
        (tuple): (certified_count, evaluated_count, results), where results maps
            loader-order input indices to outcomes, method names and running times.
    """
    certified, total, _, results = eval_adaptive(
        model, eps, data_loader, n_classes, test_samples, device,
        methods=['ALPHA-CROWN'], alpha_iterations=alpha_iterations)
    return int(certified), total, results


def eval_forward(model, eps, data_loader, n_classes=10, test_samples=np.inf,
                 device='cuda', propagation_batch_size=1):
    """Certify local robustness using forward LiRPA bounds.

    Same-slope forward relaxations bound the complete clipped input box.
    An input is certified when every true-class margin has a positive lower
    bound. Uncertified inputs remain unresolved by this bounding method.

    Args:
        model (auto_LiRPA.BoundedModule): Network to evaluate; forward LiRPA also accepts
            native models.
        eps (float or torch.Tensor): Perturbation radius in model input coordinates.
        data_loader (DataLoader): Inputs, labels and feasible input-domain limits.
        n_classes (int): Number of output classes.
        test_samples (int or float): Positive sample budget; infinity evaluates all inputs.
        device (str or torch.device): Evaluation device.
        propagation_batch_size (int): Maximum number of inputs propagated together to limit
            bound-computation memory.

    Returns:
        (tuple): (certified_count, evaluated_count, results), where results maps
            loader-order input indices to outcomes, method names and running times.
    """
    certified, total, _, results = eval_adaptive(
        model, eps, data_loader, n_classes, test_samples, device,
        methods=['FORWARD'], propagation_batch_size=propagation_batch_size)
    return int(certified), total, results


def eval_complete(
    model,
    eps_std,
    data_loader,
    n_classes=10,
    input_shape=[1, 28, 28],
    test_samples=np.inf,
    timeout=1000,
    no_cores=28,
    abcrown_batch_size=512,
    abcrown_config_dict=None,
    separate_abcrown_process=False,
    device="cuda",
    results_path="./abCROWN_results",
    warm_start=False,
    start_idx=0,
    end_idx=None,
    results_filename="results.json",
    engine="abcrown",
    neuralsat_command=None,
    neuralsat_batch_size=1000,
):
    """Evaluate robustness with incomplete screening followed by a complete verifier.

    CTRAIN first applies PGD and incomplete bounds to the same captured inputs.
    Unresolved queries are exported as ONNX networks and VNNLIB specifications
    for alpha-beta-CROWN or NeuralSAT. A timeout leaves a query unresolved.
    The loader must be sequential with drop_last=False. Captured CPU tensors
    keep all stages consistent even when dataset transformations are stochastic.

    Results and an identity manifest are saved under results_path. Warm-start
    reuse requires matching model bytes, inputs, labels, input limits and radius;
    legacy result files without a matching manifest require a fresh run.
    Caller modes and gradient flags are restored, including on failure.

    Args:
        model (auto_LiRPA.BoundedModule): The neural network model to be evaluated.
        eps_std (float): Perturbation radius in model input coordinates.
        data_loader (torch.utils.data.DataLoader): DataLoader for the dataset to be
            evaluated.
        n_classes (int, optional): Number of classes in the dataset.
        input_shape (list, optional): Shape of the input data.
        test_samples (int, optional): Number of samples to test in order of the test loader.
        timeout (int, optional): Timeout for the ABCROWN evaluation in seconds.
        no_cores (int, optional): Number of cores to use for MIP solving during the ABCROWN
            evaluation (if configured).
        abcrown_batch_size (int, optional): Batch size for the ABCROWN evaluation.
        abcrown_config_dict (dict, optional): Optional alpha-beta-CROWN configuration overrides following the
            verifier schema.
        separate_abcrown_process (bool, optional): Whether to run ABCROWN in a separate
            process.
        device (str, optional): Device to run the evaluation on.
        results_path (str, optional): Path to save the results of the ABCROWN evaluation.
        warm_start (bool, optional): Reuse results only when the identity manifest matches.
        start_idx (int): First loader-order index included in the verification slice.
        end_idx (int, optional): Exclusive ending index of the verification slice; None uses the available
            inputs.
        results_filename (str): JSON result filename within results_path.
        engine (str): "abcrown" (default) or "neuralsat".
        neuralsat_command (list): NeuralSAT argv prefix; None uses the bundled solver.
        neuralsat_batch_size (int): NeuralSAT maximum parallel branches (default 1000).

    Returns:
        (tuple): A tuple containing the certified accuracy and the adversarial accuracy.

    Requires a sequential loader with drop_last=False. Captures the evaluated
    inputs in CPU memory so every screening/solver stage uses identical tensors.
    A sidecar manifest binds cached records to exported model bytes, transformed
    inputs, labels, classes, clipping bounds and radius. Legacy caches need a
    fresh run. Caller modes and gradient flags are restored on failure.
    """

    if engine not in ("abcrown", "neuralsat"):
        raise ValueError("engine must be 'abcrown' or 'neuralsat'")
    if engine == "neuralsat":
        neuralsat_command = validate_neuralsat_command(neuralsat_command)
        validate_neuralsat_options(timeout, device, neuralsat_batch_size)
        if abcrown_config_dict is not None:
            raise ValueError("abcrown_config_dict cannot be used with NeuralSAT")
    method_name = "abCROWN" if engine == "abcrown" else "neuralsat"

    with tempfile.TemporaryDirectory(prefix=f'ctrain-{method_name}-') as tmp_root, preserve_model_state(model):
        if not isinstance(data_loader.sampler, SequentialSampler) or data_loader.drop_last or data_loader.batch_size is None:
            raise ValueError('Complete verification requires a sequential loader with a batch size and drop_last=False')
        test_samples = validate_test_samples(test_samples)
        eps_std = torch.as_tensor(eps_std, device=device)
        if not torch.isfinite(eps_std).all() or (eps_std < 0).any():
            raise ValueError('Verification radius must be finite and nonnegative')
        input_shape = list(input_shape)
        if len(input_shape) == 3:
            input_shape = [1, *input_shape]
        model.eval()

        os.makedirs(results_path, exist_ok=True)
        os.makedirs(f"{results_path}/{method_name}_logs/", exist_ok=True)
        test_limit = min(int(test_samples), len(data_loader.dataset)) if test_samples < np.inf else len(data_loader.dataset)
        start_idx = max(0, int(start_idx))
        end_idx = test_limit if end_idx is None else min(test_limit, int(end_idx))
        if start_idx >= end_idx:
            raise ValueError(f"Invalid verification chunk [{start_idx}, {end_idx}) for test_limit={test_limit}")
        results_file = os.path.join(results_path, results_filename)
        print(f"Complete verification chunk: [{start_idx}, {end_idx}) -> {results_file}")

        # Freeze transformed inputs so every stage certifies exactly the same region.
        samples, labels, seen = [], [], 0
        for data, targets in data_loader:
            count = min(len(targets), test_limit - seen)
            samples.append(data[:count].detach().cpu())
            labels.append(targets[:count].detach().cpu())
            seen += count
            if seen == test_limit:
                break
        if seen != test_limit:
            raise ValueError('Verification loader did not yield the requested samples')
        captured = TensorDataset(torch.cat(samples), torch.cat(labels))
        del samples, labels
        metadata = {name: getattr(data_loader, name) for name in
                    ('mean', 'std', 'min', 'max', 'normalised') if hasattr(data_loader, name)}
        data_loader = DataLoader(captured, batch_size=data_loader.batch_size, shuffle=False)
        for name, value in metadata.items():
            setattr(data_loader, name, value)

        model_onnx_path = os.path.join(tmp_root, 'model.onnx')
        export_onnx(model=model, file_name=model_onnx_path, batch_size=1, input_shape=input_shape)
        digest = hashlib.sha256()
        for value in (*captured.tensors, eps_std, data_loader.min, data_loader.max, data_loader.std):
            value = torch.as_tensor(value).detach().cpu().contiguous()
            digest.update(str((value.dtype, tuple(value.shape))).encode())
            digest.update(value.reshape(-1).view(torch.uint8).numpy())
        manifest = {'version': 1, 'engine': engine, 'n_classes': n_classes,
                    'inputs_sha256': digest.hexdigest(),
                    'model_sha256': hashlib.sha256(Path(model_onnx_path).read_bytes()).hexdigest()}
        manifest_file = results_file + '.manifest.json'

        if warm_start and os.path.exists(results_file):
            if not os.path.exists(manifest_file):
                raise ValueError('Legacy verification results have no identity manifest; rerun with warm_start=False')
            with open(manifest_file) as file:
                previous = json.load(file)
            if previous.get('engine') != engine:
                raise ValueError('Warm-start results belong to a different verification engine')
            if previous != manifest:
                raise ValueError('Warm-start results do not match the model, inputs, labels or perturbation region')
            with open(results_file, "r") as f:
                results = json.load(f)

            print(f"Loaded {len(results)} results from {results_file}")
            results = {int(k): v for k, v in results.items()}
            for idx, item in results.items():
                if item.get("method") not in (None, method_name, "clean_classification", "IBP", "CROWN-IBP", "CROWN", "PGD"):
                    raise ValueError("Warm-start results belong to a different verification engine")
                if item.get('running_time') is None:
                    item['running_time'] = 0
        else:
            results = {}
            # Clear old certificates before binding this file to a new identity.
            write_json_atomic(results_file, results)
            write_json_atomic(manifest_file, manifest)

        for idx in range(start_idx, end_idx):
            results.setdefault(idx, {"result": None, "method": None, "running_time": 0})

        adv_sample_found = torch.zeros(test_limit, device=device, dtype=torch.bool)
        certified = torch.zeros(test_limit, device=device, dtype=torch.bool)
        for idx, item in results.items():
            if idx < 0 or idx >= test_limit:
                continue
            if item.get("result") == "sat":
                adv_sample_found[idx] = True
            elif item.get("result") == "unsat":
                certified[idx] = True
        no_certified = torch.sum(certified).item()

        write_json_atomic(results_file, results)

        total_images = 0

        batch_size = data_loader.batch_size

        def subset_loader(indices):
            loader = DataLoader(
                Subset(data_loader.dataset, indices),
                batch_size=batch_size,
                shuffle=False,
                num_workers=getattr(data_loader, "num_workers", 0),
            )
            for name in ("mean", "std", "min", "max", "normalised"):
                if hasattr(data_loader, name):
                    setattr(loader, name, getattr(data_loader, name))
            return loader

        model.eval()
        pending = [
            idx for idx in range(start_idx, end_idx)
            if results[idx].get("result") not in ("sat", "unsat", "timeout")
        ]
        for idx in pending:
            results[idx]["result"] = None
        clean_pending = []
        seen = 0
        with torch.no_grad():
            for data, targets in subset_loader(pending):
                correct = model(data.to(device)).argmax(1).cpu() == targets
                for local_idx, is_correct in enumerate(correct):
                    idx = pending[seen + local_idx]
                    if is_correct:
                        clean_pending.append(idx)
                    else:
                        results[idx] = {"result": "sat", "method": "clean_classification", "running_time": 0}
                        adv_sample_found[idx] = True
                seen += len(targets)

        if clean_pending:
            _, _, incomplete_certified, incomplete_results = eval_adaptive(
                model=model,
                eps=eps_std,
                data_loader=subset_loader(clean_pending),
                n_classes=n_classes,
                test_samples=len(clean_pending),
                device=device,
                methods=["IBP", "CROWN"],
            )
            for local_idx, idx in enumerate(clean_pending):
                record = incomplete_results[local_idx]
                results[idx] = {
                    "result": "unsat" if incomplete_certified[local_idx] else None,
                    "method": record.get("method"),
                    "running_time": record.get("running_time") or 0,
                }
                if incomplete_certified[local_idx]:
                    certified[idx] = True

        pgd_pending = [idx for idx in clean_pending if not certified[idx]]
        if pgd_pending:
            _, pgd_found, pgd_results = eval_adversarial(
                model=model,
                data_loader=subset_loader(pgd_pending),
                eps=eps_std,
                n_classes=n_classes,
                device=device,
                test_samples=len(pgd_pending),
                return_adv_indices=True,
                early_stopping=True,
            )
            for local_idx, idx in enumerate(pgd_pending):
                pgd_record = pgd_results[local_idx]
                running_time = results[idx]["running_time"] + (pgd_record.get("running_time") or 0)
                if pgd_found[local_idx]:
                    results[idx] = {"result": "sat", "method": "PGD", "running_time": running_time}
                    adv_sample_found[idx] = True
                else:
                    results[idx]["running_time"] = running_time

        write_json_atomic(results_file, results)
        no_certified = torch.sum(certified).item()
        unresolved = [
            idx for idx in range(start_idx, end_idx)
            if results[idx].get("result") not in ("sat", "unsat", "timeout")
        ]

        std_config = get_abcrown_standard_conf(timeout=timeout, no_cores=no_cores) if engine == "abcrown" else {}
        if engine == "abcrown":
            std_config["solver"]["batch_size"] = abcrown_batch_size
        if abcrown_config_dict is not None:
            def update_config(base_config, custom_config):
                for key, value in custom_config.items():
                    if isinstance(value, dict) and key in base_config:
                        update_config(base_config[key], value)
                    else:
                        base_config[key] = value

            update_config(std_config, abcrown_config_dict)
        if engine == "abcrown":
            std_config["bab"]["hugetensor_allocator"] = False

        for batch_idx, (data, targets) in tqdm(enumerate(data_loader if unresolved else ())):
            batch_start = batch_idx * batch_size
            batch_end = batch_start + len(targets)
            if batch_start >= end_idx:
                break
            if batch_end <= start_idx:
                continue
            local_positions = [
                i for i in range(len(targets))
                if start_idx <= batch_start + i < end_idx
            ]
            global_indices = [batch_start + i for i in local_positions]
            if not local_positions:
                continue
            data = data[local_positions]
            targets = targets[local_positions]
            total_images += len(targets)
            targets = targets.to(device)
            data = data.to(device)

            print(f"BATCH {batch_idx}, global indices {global_indices[0]}:{global_indices[-1] + 1}")

            vnnlib_path = f"{tmp_root}/vnnlib_{batch_idx}"
            os.makedirs(vnnlib_path, exist_ok=True)

            eps_std = eps_std.to("cpu")

            vnnlib_batch = instances_to_vnnlib(
                indices=[
                    i
                    for i in range(len(targets))
                    if not certified[global_indices[i]]
                    and not adv_sample_found[global_indices[i]]
                ],
                data=[(img, target) for img, target in zip(data, targets)],
                vnnlib_path=vnnlib_path,
                experiment_name="Experiment",
                eps=eps_std * data_loader.std,
                eps_temp=eps_std,
                data_min=data_loader.min,
                data_max=data_loader.max,
                no_classes=n_classes,
            )
            vnnlib_indices = [
                global_indices[i]
                for i in range(len(targets))
                if not certified[global_indices[i]]
                and not adv_sample_found[global_indices[i]]
            ]
            print(vnnlib_indices)
            print(adv_sample_found[global_indices[0] : global_indices[-1] + 1])
            print(certified[global_indices[0] : global_indices[-1] + 1])
            for idx, vnn_instance in zip(vnnlib_indices, vnnlib_batch):
                if results[idx]["result"] in ("sat", "unsat", "timeout"):
                    print(f"Skipping instance {idx} as it already has a result: {results[idx]['result']}")
                    continue
                if idx >= test_limit:
                    print(f"Skipping instance {idx} as it exceeds test_samples {test_samples}")
                    continue
                if engine == "neuralsat":
                    running_time, result = neuralsat_eval(
                        model_onnx_path, vnn_instance, neuralsat_command,
                        timeout=timeout, device=device, batch_size=neuralsat_batch_size,
                        log_path=f"{results_path}/{method_name}_logs/{idx}.log",
                    )
                elif separate_abcrown_process:
                    running_time, result = limited_abcrown_eval(
                        config=std_config,
                        seed=42,
                        instance=vnn_instance,
                        vnnlib_path=vnnlib_path,
                        model_path=None,
                        model_name=None,
                        model_onnx_path=model_onnx_path,
                        input_shape=[-1, *input_shape[1:]],
                        timeout=timeout,
                        no_cores=no_cores,
                        par_factor=1,
                    )
                else:
                    torch.cuda.empty_cache()
                    gc.collect()
                    with redirect_stdout_to_file(f"{results_path}/abCROWN_logs/{idx}.log"):
                        running_time, result = abcrown_eval(
                            config=std_config,
                            seed=42,
                            instance=vnn_instance,
                            vnnlib_path=vnnlib_path,
                            model_path=None,
                            model_name=None,
                            model_onnx_path=model_onnx_path,
                            input_shape=[-1, *input_shape[1:]],
                            timeout=timeout,
                            no_cores=no_cores,
                            par_factor=1,
                        )
                torch.cuda.empty_cache()
                gc.collect()
                print(f"Instance {idx} finished with result {result} in {running_time:.2f} seconds")
                if result == "unsat":
                    no_certified += 1
                    certified[idx] = True
                if result == "sat":
                    adv_sample_found[idx] = True

                results[idx] = {
                    "result": result,
                    "method": method_name,
                    "running_time": running_time + results[idx]["running_time"],
                }
                write_json_atomic(results_file, results)

            shutil.rmtree(vnnlib_path)

        write_json_atomic(results_file, results)

        chunk_results = {idx: results[idx] for idx in range(start_idx, end_idx)}
        no_certified = sum(1 for item in chunk_results.values() if item["result"] == "unsat")
        no_counterexample = sum(1 for item in chunk_results.values() if item["result"] == "sat")

        print(certified)
        print(no_certified)

        chunk_total = len(chunk_results)
        certified_acc = no_certified / chunk_total
        adv_acc = (chunk_total - no_counterexample) / chunk_total
        if torch.is_tensor(adv_acc):
            adv_acc = adv_acc.item()

        return certified_acc, adv_acc

def eval_complete_abcrown(*args, **kwargs):
    """Evaluate robustness with alpha-beta-CROWN using the arguments of eval_complete."""
    return eval_complete(*args, **kwargs)

def eval_adaptive(
    model,
    eps,
    data_loader,
    n_classes=10,
    test_samples=np.inf,
    device="cuda",
    methods=["IBP", "CROWN-IBP", "CROWN"],
    alpha_iterations=20,
    propagation_batch_size=1,
):
    """Certify inputs progressively using the requested incomplete methods.

    Methods are applied in increasing computational-cost order, and samples
    certified by an earlier method are omitted from later bound computations.
    Thus, inexpensive bounds solve easy cases while tighter methods are reserved
    for unresolved samples.

    Args:
        model (auto_LiRPA.BoundedModule): The neural network model to be evaluated.
        eps (float): Perturbation radius in model input coordinates.
        data_loader (torch.utils.data.DataLoader): DataLoader for the dataset to be
            evaluated.
        n_classes (int, optional): Number of classes in the dataset.
        test_samples (int, optional): Number of samples to test in order of the test loader.
        device (str, optional): Device to run the evaluation on.
        methods (list): Incomplete verification methods to apply progressively to unresolved
            inputs.
        alpha_iterations (int): Number of optimisation iterations for alpha-CROWN relaxation
            slopes.
        propagation_batch_size (int): Maximum number of inputs propagated together to limit
            bound-computation memory.

    Returns:
        (tuple): Certified count, evaluated count, per-instance certification mask, and result records.
    """
    eps = _to_device_tensor(eps, device)
    if not torch.isfinite(eps).all() or (eps < 0).any():
        raise ValueError('eps must be nonnegative and finite')
    methods = _canonical_methods(methods)
    if isinstance(methods, str):
        methods = [methods]
    if not methods or any(method not in ('IBP', 'CROWN-IBP', 'CROWN', 'ALPHA-CROWN', 'FORWARD') for method in methods):
        raise ValueError('Adaptive methods must contain IBP, CROWN-IBP, CROWN, ALPHA-CROWN, or FORWARD')
    if type(propagation_batch_size) is not int or propagation_batch_size <= 0:
        raise ValueError('propagation_batch_size must be a positive integer')
    if 'ALPHA-CROWN' in methods and (type(alpha_iterations) is not int or alpha_iterations <= 0):
        raise ValueError('alpha_iterations must be a positive integer')
    test_samples = validate_test_samples(test_samples)
    limit = min(len(data_loader.dataset), int(test_samples)) if test_samples < np.inf else len(data_loader.dataset)
    if not limit or not len(data_loader):
        raise ValueError('Evaluation loader must contain samples')
    certified = torch.zeros(limit, dtype=torch.bool, device=device)
    results = {i: {"result": None, "method": None, "running_time": 0.} for i in range(limit)}
    seen = 0
    data_min = _to_device_tensor(data_loader.min, device)
    data_max = _to_device_tensor(data_loader.max, device)
    forward_model = model if isinstance(model, BoundedModule) else None
    with preserve_model_state(model):
        model.eval()
        for data, targets in data_loader:
            if seen >= limit:
                break
            count = min(len(targets), limit - seen)
            data, targets = data[:count].to(device), targets[:count].to(device)
            if n_classes is None:
                with torch.no_grad():
                    n_classes = model(data[:1]).shape[1]
            for method, bound in [('IBP', bound_ibp), ('CROWN-IBP', bound_crown_ibp), ('CROWN', bound_crown), ('ALPHA-CROWN', bound_alpha_crown), ('FORWARD', bound_forward)]:
                if method not in methods:
                    continue
                pending = torch.where(~certified[seen:seen + count])[0]
                groups = (pending.split(propagation_batch_size) if method == 'FORWARD'
                          else pending.split(1) if method in ('CROWN', 'ALPHA-CROWN') else [pending])
                for indices in groups:
                    if not len(indices):
                        continue
                    inputs, labels = data[indices], targets[indices]
                    if method == 'FORWARD' and forward_model is None:
                        # Conversion must not touch the native model's BN buffers.
                        with torch.no_grad():
                            forward_model = BoundedModule(copy.deepcopy(model), inputs, device=device)
                            forward_model.eval()
                    perturbation = _BoxPerturbation if method == 'FORWARD' else PerturbationLpNorm
                    ptb = perturbation(eps=eps, norm=np.inf,
                        x_L=torch.clamp(inputs - eps, data_min, data_max),
                        x_U=torch.clamp(inputs + eps, data_min, data_max))
                    start = time.monotonic()
                    lb, _ = bound(model=forward_model if method == 'FORWARD' else model,
                                  ptb=ptb, data=inputs, target=labels,
                                  n_classes=n_classes, reuse_input=False,
                                  **({'alpha_iterations': alpha_iterations} if method == 'ALPHA-CROWN' else {}))
                    elapsed = (time.monotonic() - start) / len(indices)
                    verified = (lb > 0).all(dim=1)
                    certified[seen + indices] = verified
                    for index, success in zip(indices.tolist(), verified.tolist()):
                        record = results[seen + index]
                        record.update(result='unsat' if success else None, method=method,
                                      running_time=record['running_time'] + elapsed)
            seen += count
    return certified[:seen].sum(), seen, certified[:seen], results

# TODO: can we maybe spare no_classes?
def eval_certified(
    model, data_loader, eps, n_classes=10, test_samples=np.inf, method="IBP", device=None,
    alpha_iterations=20, propagation_batch_size=1
):
    """Evaluate certified local robustness using the selected bounding method.

    Choose one bound-propagation method, a list for adaptive verification, or
    COMPLETE for solver-backed evaluation. Radii are already in model input
    coordinates; wrapper.evaluate performs loader-based normalisation for users.

    Args:
        model (auto_LiRPA.BoundedModule): The neural network model to be evaluated.
        data_loader (torch.utils.data.DataLoader): DataLoader for the dataset to be
            evaluated.
        eps (float or Tensor): Perturbation radius in model input coordinates.
        n_classes (int, optional): Number of classes in the dataset.
        test_samples (int or float, optional): Number of test samples to evaluate.
        method (str or list, optional): The certification method to use. Options are
            'FORWARD', 'IBP', 'CROWN', 'ALPHA-CROWN',
            'CROWN-IBP', 'ADAPTIVE', 'COMPLETE', or a list of
            methods (which results in an ADAPTIVE evaluation
            using these methods).
        device (str or torch.device, optional): Device on which to run the model and tensor computations.
        alpha_iterations (int): Number of optimisation iterations for alpha-CROWN relaxation
            slopes.
        propagation_batch_size (int): Maximum forward-bound chunk size; default 1.

    Returns:
        (tuple): (certified_accuracy, results), with accuracy as a fraction and
            per-input verification outcomes and running times in results. The legacy
            COMPLETE branch returns an aggregate placeholder record.
    """
    device = torch.device(device) if device is not None else torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    with preserve_model_state(model):
        model.eval()
        certified = 0
        total_images = 0

        method = _canonical_methods(method)

        if method == "FORWARD":
            certified, total_images, results = eval_forward(
                model, eps, data_loader, n_classes, test_samples, device, propagation_batch_size)
        elif method == "CROWN":
            certified, total_images, results = eval_crown(
                model, eps, data_loader, n_classes, test_samples, device
            )
        elif method == "ALPHA-CROWN":
            certified, total_images, results = eval_alpha_crown(
                model, eps, data_loader, n_classes, test_samples, device, alpha_iterations
            )
        elif method == "IBP":
            certified, total_images, results = eval_ibp(
                model, eps, data_loader, n_classes, test_samples, device
            )
        elif method == "CROWN-IBP":
            certified, total_images, results = eval_crown_ibp(
                model, eps, data_loader, n_classes, test_samples, device
            )
        elif method == "ADAPTIVE":
            certified, total_images, certified_tensor, results = eval_adaptive(
                model, eps, data_loader, n_classes, test_samples, device
            )
        elif isinstance(method, list):
            certified, total_images, certified_tensor, results = eval_adaptive(
                model, eps, data_loader, n_classes, test_samples, device, methods=method,
                alpha_iterations=alpha_iterations, propagation_batch_size=propagation_batch_size
            )
        elif method == "COMPLETE":
            # TODO: Infer input shape or pass it, pass timeout and no_cores!
            cert_acc, adv_acc = eval_complete_abcrown(
                model,
                eps,
                data_loader,
                n_classes,
                input_shape=(1, 28, 28),
                test_samples=test_samples,
                timeout=1000,
                no_cores=28,
                device=device,
            )
            return cert_acc, {"result": None, "method": "COMPLETE", "running_time": None}
        elif isinstance(method, (list, tuple, np.ndarray)):
            certified, total_images, certified_tensor, results = eval_adaptive(
                model, eps, data_loader, n_classes, test_samples, device, methods=method,
                alpha_iterations=alpha_iterations, propagation_batch_size=propagation_batch_size
            )
        else:
            assert False, "UNKNOWN BOUNDING METHOD!"

        return (
            (certified / total_images).cpu().item()
            if torch.is_tensor(certified)
            else certified / total_images
        ), results

def eval_adversarial(
    model,
    data_loader,
    eps,
    return_adv_indices=False,
    restarts=5,
    step_size=0.1,
    n_steps=40,
    early_stopping=False,
    n_classes=10,
    device="cuda",
    test_samples=np.inf,
    method="PGD",
    autoattack_kwargs=None,
):
    """Search for classification counterexamples using PGD or AutoAttack.

    The returned adversarial accuracy is the fraction of inputs for which the
    attack found no misclassification. It is an empirical upper bound on robust
    accuracy; failing to find a counterexample does not establish robustness.

    Args:
        model (torch.nn.Module): The model to evaluate.
        data_loader (torch.utils.data.DataLoader): DataLoader providing the dataset.
        eps (torch.Tensor): Perturbation radius in model input coordinates.
        return_adv_indices (bool, optional): Whether to return a Boolean failure mask and
            per-input result records.
        restarts (int, optional): Number of random restarts for the PGD attack.
        step_size (float, optional): Step size for the PGD attack.
        n_steps (int, optional): Number of steps for the PGD attack.
        early_stopping (bool, optional): Whether to stop early if an adversarial example is
            found.
        n_classes (int, optional): Number of classes in the dataset.
        device (str, optional): Device to perform computations on.
        test_samples (int or float, optional): Number of samples to test.
        method (str): PGD (default) or AUTOATTACK.
        autoattack_kwargs (dict, optional): AutoAttack version, seed and verbose settings.

    Returns:
        (float or tuple): Adversarial accuracy as a fraction. With return_adv_indices=True,
            return (accuracy, failure_mask, results), where the Boolean mask marks
            inputs with a found misclassification and results holds per-input outcomes.
    """
    test_samples = validate_test_samples(test_samples)
    with preserve_model_state(model):
        if method.upper().replace('-', '').replace('_', '') == 'AUTOATTACK':
            return eval_autoattack(model, data_loader, eps, return_adv_indices,
                                   device, test_samples, **(autoattack_kwargs or {}))
        if method != 'PGD':
            raise ValueError('Adversarial method must be PGD or AUTOATTACK')
        model.eval()
        adv_preds = np.array([])
        labels = np.array([])
        data_min = _to_device_tensor(data_loader.min, device)
        data_max = _to_device_tensor(data_loader.max, device)
        results = {}

        for batch_idx, (data, targets) in enumerate(tqdm(data_loader)):
            if len(labels) >= test_samples:
                break

            batch_indices = int(min(len(targets), test_samples - len(labels)))
            batch_results = {
                batch_idx * data_loader.batch_size + i: {"result": None, "method": "PGD", "running_time": None}
                for i in range(batch_indices)
            }
            data = data[:batch_indices]
            targets = targets[:batch_indices]

            data, targets = data.to(device), targets.to(device)
            eps = _to_device_tensor(eps, device)

            pgd_start_time = time.time()
            x_test_adv = pgd_attack(
                model=model,
                data=data,
                target=targets,
                x_L=torch.clamp(data - eps, data_min, data_max).to(device),
                x_U=torch.clamp(data + eps, data_min, data_max).to(device),
                restarts=restarts,
                step_size=step_size,
                n_steps=n_steps,
                early_stopping=early_stopping,
                device=device,
            )
            pgd_end_time = time.time()

            adv_predictions_batch = model(x_test_adv)
            adv_predictions_batch = torch.argmax(adv_predictions_batch, dim=1).cpu().numpy()
            if len(labels) + len(targets) > test_samples:
                too_many_samples_no = (len(labels) + len(targets)) - test_samples
                adv_predictions_batch = adv_predictions_batch[:-too_many_samples_no]
                targets = targets[:-too_many_samples_no]

            adv_preds = np.append(adv_preds, adv_predictions_batch)
            labels = np.append(labels, targets.cpu())
            for i in range(len(targets)):
                instance = batch_idx * data_loader.batch_size + i
                batch_results[instance]["result"] = 'sat' if adv_predictions_batch[i] != targets[i].item() else None
                batch_results[instance]["running_time"] = (pgd_end_time - pgd_start_time) / len(targets)

            results.update(batch_results)

        test_samples = min(test_samples, len(labels))

        if not len(labels):
            raise ValueError('Evaluation loader must contain samples')
        adv_accuracy = accuracy_score(labels, adv_preds)

        if return_adv_indices:
            adv_sample_found = labels != adv_preds
            return adv_accuracy, adv_sample_found, results

        return adv_accuracy

def eval_model(
    model,
    data_loader,
    eps,
    n_classes=10,
    test_samples=np.inf,
    method="ADAPTIVE",
    device="cuda",
    adversarial_method="PGD",
    alpha_iterations=20,
    autoattack_kwargs=None,
    propagation_batch_size=1,
):
    """Evaluate natural, certified and adversarial classification accuracy.

    Sound bounds establish a lower bound on robust accuracy, while adversarial
    attacks give an empirical upper bound. The same sample budget is used for
    all three measurements, with eps expressed in model input coordinates.

    Args:
        model (auto_LiRPA.BoundedModule): The model to be evaluated.
        data_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        eps (float): Perturbation radius in model input coordinates.
        n_classes (int, optional): Number of classes in the dataset.
        test_samples (int or float, optional): Number of samples to test.
        method (str or list, optional): The certification method to use. Options are
            'FORWARD', 'IBP', 'CROWN', 'ALPHA-CROWN',
            'CROWN-IBP', 'ADAPTIVE', 'COMPLETE', or a list of
            methods (which results in an ADAPTIVE evaluation
            using these methods).
        device (str, optional): Device to perform adversarial evaluation on.
        adversarial_method (str): PGD (default) or AUTOATTACK.
        alpha_iterations (int): Alpha-CROWN optimisation iterations; default 20.
        autoattack_kwargs (dict, optional): AutoAttack version, seed and verbose settings.
        propagation_batch_size (int): Maximum forward-bound chunk size; default 1.

    Returns:
        (tuple): (std_acc (float): Standard accuracy of the model, cert_acc (float): Certified accuracy of the model, adv_acc (float): Adversarial accuracy of the model)
    """
    test_samples = validate_test_samples(test_samples)
    with preserve_model_state(model):
        std_acc = eval_acc(model, test_loader=data_loader, test_samples=test_samples, device=device)
        cert_acc, _ = eval_certified(
            model=model,
            data_loader=data_loader,
            n_classes=n_classes,
            eps=eps,
            test_samples=test_samples,
            method=method,
            alpha_iterations=alpha_iterations,
            propagation_batch_size=propagation_batch_size,
            device=device,
        )
        adv_acc = eval_adversarial(
            model=model,
            data_loader=data_loader,
            eps=eps,
            n_classes=n_classes,
            device=device,
            test_samples=test_samples,
            method=adversarial_method,
            autoattack_kwargs=autoattack_kwargs,
            restarts=30,
            n_steps=100,
            step_size=.1,
            early_stopping=True,
        )

        return std_acc, cert_acc, adv_acc

def eval_epoch(
    model,
    data_loader,
    eps,
    n_classes,
    device="cuda",
    test_samples=1000,
    verification_method="IBP",
    results_path="./results",
):
    """Evaluate and record a model's natural, certified and adversarial accuracies.

    The measurements are written as JSON under results_path and returned in
    the common (standard, certified, adversarial) order.

    Args:
        model (torch.nn.Module): The model to be evaluated.
        data_loader (torch.utils.data.DataLoader): DataLoader for the dataset to evaluate.
        eps (torch.Tensor): Perturbation radius in model input coordinates.
        n_classes (int): Number of classes in the dataset.
        device (str, optional): Device to run the evaluation on.
        test_samples (int, optional): Number of samples to use for evaluation.
        verification_method (str, optional): Method to use for certification.
        results_path (str, optional): Path to save the evaluation results.

    Returns:
        (tuple): A tuple containing standard accuracy, certified accuracy, and adversarial accuracy.
    """
    test_samples = validate_test_samples(test_samples)
    with preserve_model_state(model):
        os.makedirs(results_path, exist_ok=True)
        model.eval()
        std_acc = eval_acc(model, test_loader=data_loader, test_samples=test_samples, device=device)
        if (torch.as_tensor(eps) == 0.0).all():
            cert_acc = adv_acc = std_acc
        else:
            with torch.no_grad():
                cert_acc, results = eval_certified(
                    model=model,
                    data_loader=data_loader,
                    n_classes=n_classes,
                    eps=eps,
                    test_samples=test_samples,
                    method=verification_method,
                    device=device,
                )
                adv_acc = eval_adversarial(
                    model=model,
                    data_loader=data_loader,
                    n_classes=n_classes,
                    eps=eps,
                    device=device,
                    test_samples=test_samples,
                )

        with open(f"{results_path}/stats.json", "w") as f:
            json.dump({"acc": std_acc, "cert_acc": cert_acc, "adv_acc": adv_acc}, f)
        return std_acc, cert_acc, adv_acc
