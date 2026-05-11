import json
import os
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

import shutil

from auto_LiRPA import PerturbationLpNorm, BoundedTensor
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from CTRAIN.bound import bound_ibp, bound_crown, bound_crown_ibp
from CTRAIN.attacks import pgd_attack
from CTRAIN.complete_verification.abCROWN.util import (
    instances_to_vnnlib,
    get_abcrown_standard_conf,
)
from CTRAIN.complete_verification.abCROWN.verify import (
    limited_abcrown_eval,
    abcrown_eval,
)
from CTRAIN.util import export_onnx, construct_c

from contextlib import redirect_stdout, contextmanager

import gc

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as null:
        with redirect_stdout(null):
            yield

@contextmanager
def redirect_stdout_to_file(filename):
    with open(filename, "w") as file:
        with redirect_stdout(file):
            yield


def write_json_atomic(filename, payload):
    tmp_filename = f"{filename}.tmp"
    with open(tmp_filename, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_filename, filename)


def _canonical_methods(methods):
    if isinstance(methods, str):
        return methods.replace("_", "-")
    return [method.replace("_", "-") if isinstance(method, str) else method for method in methods]


def _to_device_tensor(value, device):
    return value.to(device) if torch.is_tensor(value) else torch.as_tensor(value, device=device)


def eval_acc(model, test_loader, test_samples=np.inf):
    device = torch.device(
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

    return correct / total


def eval_ibp(model, eps, data_loader, n_classes=10, test_samples=np.inf, device="cuda"):
    """
    Evaluate a model using Interval Bound Propagation (IBP) for certification.

    Args:
        model (auto_LiRPA.BoundedModule): The neural network model to be evaluated.
        eps (float): The l_inf perturbation bound for for certification.
        data_loader (torch.utils.data.DataLoader): DataLoader providing the dataset to evaluate.
        n_classes (int, optional): Number of classes in the classification task. Default is 10.
        test_samples (int, optional): Number of samples to test in order of the test loader. Default is np.inf (test all samples).
        device (str, optional): Device to run the evaluation on ('cuda', 'mps', 'cpu'). Default is 'cuda'.

    Returns:
        (tuple): A tuple containing the number of certified samples, the total number of images evaluated and per-instance result and running time.
    """
    certified = 0
    total_images = 0
    results = {}
    for batch_idx, (data, targets) in tqdm(enumerate(data_loader)):
        if total_images >= test_samples:
            break

        batch_indices = int(min(len(targets), test_samples - total_images))
        data = data[:batch_indices]
        targets = targets[:batch_indices]
        batch_results = {
            batch_idx * data_loader.batch_size + i: {"result": None, "method": "IBP", "running_time": None}
            for i in range(batch_indices)
        }
        data, targets = data.to(device), targets.to(device)
        eps_device = _to_device_tensor(eps, device)
        data_min, data_max = data_loader.min.to(device), data_loader.max.to(device)

        ptb = PerturbationLpNorm(
            eps=eps_device,
            norm=np.inf,
            x_L=torch.clamp(data - eps_device, data_min, data_max),
            x_U=torch.clamp(data + eps_device, data_min, data_max),
        )

        start_time_ibp = time.time()
        lb, ub = bound_ibp(
            model=model,
            ptb=ptb,
            data=data,
            target=targets,
            n_classes=n_classes,
            reuse_input=False,
        )
        end_time_ibp = time.time()
        no_certified = torch.sum((lb > 0).all(dim=1)).item()
        # no_falsified = torch.sum((ub < 0).any(dim=1)).item()
        certified += no_certified

        total_images += len(targets)

        instance_running_time = (end_time_ibp - start_time_ibp) / len(targets)
        for i in range(len(targets)):
            instance = batch_idx * data_loader.batch_size + i
            batch_results[instance]["result"] = 'unsat' if (lb[i] > 0).all().item() else None
            batch_results[instance]["running_time"] = instance_running_time

        results.update(batch_results)

    return certified, total_images, results


def eval_crown_ibp(
    model, eps, data_loader, n_classes=10, test_samples=np.inf, device="cuda"
):
    """
    Evaluate the model using the CROWN-IBP method.

    Parameters:
        model (auto_LiRPA.BoundedModule): The neural network model to be evaluated.
        eps (float): The perturbation bound.
        data_loader (torch.utils.data.DataLoader): DataLoader for the dataset to be evaluated.
        n_classes (int, optional): Number of classes in the dataset. Default is 10.
        test_samples (int, optional): Number of samples to test in order of the test loader. Default is np.inf (test all samples).
        device (str, optional): Device to run the evaluation on. Default is 'cuda'.

    Returns:
        (tuple): A tuple containing the number of certified samples, the total number of images evaluated and per-instance result and running time.
    """
    certified = 0
    total_images = 0
    results = {}
    for batch_idx, (data, targets) in tqdm(enumerate(data_loader)):
        if total_images >= test_samples:
            break
        batch_indices = int(min(len(targets), test_samples - total_images))
        data = data[:batch_indices]
        targets = targets[:batch_indices]
        batch_results = {
            batch_idx * data_loader.batch_size + i: {"result": None, "method": "CROWN-IBP", "running_time": None}
            for i in range(batch_indices)
        }
        data, targets = data.to(device), targets.to(device)
        eps_device = _to_device_tensor(eps, device)
        data_min, data_max = data_loader.min.to(device), data_loader.max.to(device)

        ptb = PerturbationLpNorm(
            eps=eps_device,
            norm=np.inf,
            x_L=torch.clamp(data - eps_device, data_min, data_max),
            x_U=torch.clamp(data + eps_device, data_min, data_max),
        )

        start_time_crown_ibp = time.time()
        lb, ub = bound_crown_ibp(
            model=model,
            ptb=ptb,
            data=data,
            target=targets,
            n_classes=n_classes,
            reuse_input=False,
        )
        end_time_crown_ibp = time.time()

        no_certified = torch.sum((lb > 0).all(dim=1)).item()
        # no_falsified = torch.sum((ub < 0).any(dim=1)).item()
        certified += no_certified

        total_images += len(targets)

        for i in range(len(targets)):
            instance = batch_idx * data_loader.batch_size + i
            batch_results[instance]["result"] = 'unsat' if (lb[i] > 0).all().item() else None
            batch_results[instance]["running_time"] = (
                end_time_crown_ibp - start_time_crown_ibp
            ) / len(targets)

        results.update(batch_results)

    return certified, total_images, results


def eval_crown(
    model, eps, data_loader, n_classes=10, test_samples=np.inf, device="cuda"
):
    """
    Evaluate the model using the CROWN method.

    Parameters:
        model (auto_LiRPA.BoundedModule): The neural network model to be evaluated.
        eps (float): The perturbation bound.
        data_loader (torch.utils.data.DataLoader): DataLoader for the dataset to be evaluated.
        n_classes (int, optional): Number of classes in the dataset. Default is 10.
        test_samples (int, optional): Number of samples to test in order of the test loader. Default is np.inf (test all samples).
        device (str, optional): Device to run the evaluation on. Default is 'cuda'.

    Returns:
        (tuple): A tuple containing the number of certified samples, the total number of images evaluated and per-instance result and running time.
    """
    # IMPORTANT: Data Loader Batch Size must match Bounding Batch Size when using CROWN for evaluation (not important for IBP)
    crown_data_loader = DataLoader(data_loader.dataset, batch_size=1, shuffle=False)
    crown_data_loader.max, crown_data_loader.min, crown_data_loader.std = (
        data_loader.max.to(device),
        data_loader.min.to(device),
        data_loader.std.to(device),
    )
    certified = 0
    total_images = 0
    results = {
        i: {"result": None, "method": "CROWN", "running_time": None}
        for i in range(min(len(crown_data_loader.dataset), test_samples))
    }
    for batch_idx, (data, targets) in tqdm(enumerate(crown_data_loader)):
        if total_images >= test_samples:
            continue
        data, targets = data.to(device), targets.to(device)
        eps_device = _to_device_tensor(eps, device)
        ptb = PerturbationLpNorm(
            eps=eps_device,
            norm=np.inf,
            x_L=torch.clamp(data - eps_device, crown_data_loader.min, crown_data_loader.max),
            x_U=torch.clamp(data + eps_device, crown_data_loader.min, crown_data_loader.max),
        )
        start_time_crown = time.time()
        lb, ub = bound_crown(
            model=model,
            ptb=ptb,
            data=data,
            target=targets,
            n_classes=n_classes,
            reuse_input=False,
        )
        end_time_crown = time.time()
        no_certified = torch.sum((lb > 0).all(dim=1)).item()
        # no_falsified = torch.sum((ub < 0).any(dim=1)).item()
        certified += no_certified

        total_images += len(targets)

        results[batch_idx] = {
            "result": 'unsat' if (lb > 0).all(dim=1).cpu().item() else None,
            "method": "CROWN",
            "running_time": (end_time_crown - start_time_crown) / len(targets),
        }

    return certified, total_images, results


def eval_alpha_crown(
    model,
    eps,
    data_loader,
    n_classes=10,
    test_samples=np.inf,
    device="cuda",
    alpha_iterations=20,
):
    """
    Evaluate the model using alpha-CROWN (optimized CROWN).
    """
    alpha_data_loader = DataLoader(data_loader.dataset, batch_size=1, shuffle=False)
    alpha_data_loader.max, alpha_data_loader.min, alpha_data_loader.std = (
        data_loader.max.to(device),
        data_loader.min.to(device),
        data_loader.std.to(device),
    )
    certified = 0
    total_images = 0
    results = {
        i: {"result": None, "method": "ALPHA-CROWN", "running_time": None}
        for i in range(min(len(alpha_data_loader.dataset), test_samples))
    }
    model.set_bound_opts({"optimize_bound_args": {"iteration": alpha_iterations}})
    for batch_idx, (data, targets) in tqdm(enumerate(alpha_data_loader)):
        if total_images >= test_samples:
            continue
        data, targets = data.to(device), targets.to(device)
        eps_device = _to_device_tensor(eps, device)
        ptb = PerturbationLpNorm(
            eps=eps_device,
            norm=np.inf,
            x_L=torch.clamp(data - eps_device, alpha_data_loader.min, alpha_data_loader.max),
            x_U=torch.clamp(data + eps_device, alpha_data_loader.min, alpha_data_loader.max),
        )
        bounded = BoundedTensor(data, ptb=ptb)
        c = construct_c(bounded, targets, n_classes)
        start_time = time.time()
        lb, ub = model.compute_bounds(
            x=(bounded,),
            IBP=False,
            method="CROWN-Optimized",
            C=c,
            bound_upper=False,
        )
        end_time = time.time()
        no_certified = torch.sum((lb > 0).all(dim=1)).item()
        certified += no_certified
        total_images += len(targets)
        results[batch_idx] = {
            "result": "unsat" if (lb > 0).all(dim=1).cpu().item() else None,
            "method": "ALPHA-CROWN",
            "running_time": (end_time - start_time) / len(targets),
        }

    return certified, total_images, results


def eval_complete_abcrown(
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
):
    """
    Evaluate the model using the complete ABCROWN method. Attention, this evaluation may be very costly!

    Parameters:
        model (auto_LiRPA.BoundedModule): The neural network model to be evaluated.
        eps_std (float): The standard deviation of the perturbation bound.
        data_loader (torch.utils.data.DataLoader): DataLoader for the dataset to be evaluated.
        n_classes (int, optional): Number of classes in the dataset. Default is 10.
        input_shape (list, optional): Shape of the input data. Default is [1, 28, 28].
        test_samples (int, optional): Number of samples to test in order of the test loader. Default is np.inf (test all samples).
        timeout (int, optional): Timeout for the ABCROWN evaluation in seconds. Default is 1000.
        no_cores (int, optional): Number of cores to use for MIP solving during the ABCROWN evaluation (if configured). Default is 28.
        abcrown_batch_size (int, optional): Batch size for the ABCROWN evaluation. Default is 512.
        separate_abcrown_process (bool, optional): Whether to run ABCROWN in a separate process. Default is False.
        device (str, optional): Device to run the evaluation on. Default is 'cuda'.
        results_path (str, optional): Path to save the results of the ABCROWN evaluation. Default is "./abCROWN_results".
        warm_start (bool, optional): Whether to skip verification for instances where the results file contains an result already.

    Returns:
        (tuple): A tuple containing the certified accuracy and the adversarial accuracy.
    """

    eps_std = eps_std.to(device)

    os.makedirs(results_path, exist_ok=True)
    os.makedirs(f"{results_path}/abCROWN_logs/", exist_ok=True)
    test_limit = min(int(test_samples), len(data_loader.dataset)) if test_samples < np.inf else len(data_loader.dataset)
    start_idx = max(0, int(start_idx))
    end_idx = test_limit if end_idx is None else min(test_limit, int(end_idx))
    if start_idx >= end_idx:
        raise ValueError(f"Invalid verification chunk [{start_idx}, {end_idx}) for test_limit={test_limit}")
    results_file = os.path.join(results_path, results_filename)
    print(f"Complete verification chunk: [{start_idx}, {end_idx}) -> {results_file}")

    if warm_start and os.path.exists(results_file):
        with open(results_file, "r") as f:
            results = json.load(f)

        print(f"Loaded {len(results)} results from {results_file}")
        results = {int(k): v for k, v in results.items()}
        for idx, item in results.items():
            if item.get('running_time') is None:
                item['running_time'] = 0
    else:
        results = {}

    for idx in range(start_idx, end_idx):
        results.setdefault(idx, {"result": None, "method": None, "running_time": 0})

    adv_sample_found = torch.zeros(test_limit, device=device, dtype=torch.bool)
    certified = torch.zeros(test_limit, device=device, dtype=torch.bool)
    for idx, item in results.items():
        if idx >= test_limit:
            continue
        if item.get("result") == "sat":
            adv_sample_found[idx] = True
        elif item.get("result") == "unsat":
            certified[idx] = True
    no_certified = torch.sum(certified).item()

    write_json_atomic(results_file, results)

    total_images = 0

    batch_size = data_loader.batch_size

    std_config = get_abcrown_standard_conf(timeout=timeout, no_cores=no_cores)
    std_config["solver"]["batch_size"] = abcrown_batch_size
    if abcrown_config_dict is not None:
        def update_config(base_config, custom_config):
            for key, value in custom_config.items():
                if isinstance(value, dict) and key in base_config:
                    update_config(base_config[key], value)
                else:
                    base_config[key] = value

        update_config(std_config, abcrown_config_dict)

    tmp_root = f"/tmp/abCROWN_{os.getpid()}_{start_idx}_{end_idx}"
    os.makedirs(tmp_root, exist_ok=True)
    model_onnx_path = f"{tmp_root}/model.onnx"

    export_onnx(
        model=model,
        file_name=model_onnx_path,
        batch_size=1,
        input_shape=input_shape,
    )

    for batch_idx, (data, targets) in tqdm(enumerate(data_loader)):
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

        # clean_pred = torch.argmax(model(data), dim=1)
        # clean_correct = clean_pred == targets

        # for pred_idx, correct_pred in enumerate(clean_correct):
        #     if not correct_pred:
        #         results[batch_idx * batch_size + pred_idx] = {
        #             "result": 'sat',
        #             "method": "clean_classification",
        #             "running_time": 0,
        #         }
        #         adv_sample_found[batch_idx * batch_size + pred_idx] = True

        vnnlib_path = f"{tmp_root}/vnnlib_{batch_idx}"
        os.makedirs(vnnlib_path, exist_ok=True)

        eps_std = eps_std.to("cpu")

        vnnlib_batch = instances_to_vnnlib(
            indices=[
                i
                for i in range(len(targets))
                if not certified[global_indices[i]]
                and not adv_sample_found[global_indices[i]]
                # and clean_correct[i]
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
            # and clean_correct[i]
        ]
        print(vnnlib_indices)
        print(adv_sample_found[global_indices[0] : global_indices[-1] + 1])
        print(certified[global_indices[0] : global_indices[-1] + 1])
        for idx, vnn_instance in zip(vnnlib_indices, vnnlib_batch):
            if results[idx]["result"] is not None:
                print(f"Skipping instance {idx} as it already has a result: {results[idx]['result']}")
                continue
            if idx >= test_limit:
                print(f"Skipping instance {idx} as it exceeds test_samples {test_samples}")
                continue
            if separate_abcrown_process:
                running_time, result = limited_abcrown_eval(
                    # work_dir='/tmp/abCROWN',
                    config=std_config,
                    seed=42,
                    instance=vnn_instance,
                    vnnlib_path=vnnlib_path,
                    model_path=None,
                    model_name=None,
                    model_onnx_path=model_onnx_path,
                    input_shape=[-1] + input_shape[1:4],
                    timeout=timeout,
                    no_cores=no_cores,
                    par_factor=1,
                )
            else:
                # rand_no = np.random.rand()
                # if rand_no < 0.5:
                #     running_time, result = 10.0, 'sat'  # Placeholder for actual result
                # elif rand_no < 0.9:
                #     running_time, result = 600.0, 'timeout'
                # else:
                #     running_time, result = 10.0, 'unsat'
                torch.cuda.empty_cache()
                gc.collect()
                with redirect_stdout_to_file(f"{results_path}/abCROWN_logs/{idx}.log"):
                    running_time, result = abcrown_eval(
                        # work_dir='/tmp/abCROWN',
                        config=std_config,
                        seed=42,
                        instance=vnn_instance,
                        vnnlib_path=vnnlib_path,
                        model_path=None,
                        model_name=None,
                        model_onnx_path=model_onnx_path,
                        input_shape=[-1] + input_shape[1:4],
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
                "method": "abCROWN",
                "running_time": running_time + results[idx]["running_time"],
            }
            write_json_atomic(results_file, results)



        shutil.rmtree(vnnlib_path)

    # with torch.no_grad():
    #     for batch_idx, (data, targets) in tqdm(enumerate(data_loader)):
    #         targets = targets.to(device)
    #         data = data.to(device)

    #         # print(f"BATCH {batch_idx}")

    #         clean_pred = torch.argmax(model(data), dim=1)
    #         clean_correct = clean_pred == targets
    #         # print(clean_correct)

    #         for pred_idx, correct_pred in enumerate(clean_correct):
    #             if not correct_pred:
    #                 results[batch_idx * batch_size + pred_idx] = {
    #                     "result": 'sat',
    #                     "method": "clean_classification",
    #                     "running_time": 0,
    #                 }
    #                 adv_sample_found[batch_idx * batch_size + pred_idx] = True

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

    shutil.rmtree(tmp_root, ignore_errors=True)

    return certified_acc, adv_acc

def eval_adaptive(
    model,
    eps,
    data_loader,
    n_classes=10,
    test_samples=np.inf,
    device="cuda",
    methods=["IBP", "CROWN-IBP", "CROWN"],
):
    """
    Evaluate the model in terms of certified accuracy in an adaptive method. This means, that all methods
    passed in the methods parameter are used to certify the samples in ascending order of computational complexity (IBP < CROWN-IBP < CROWN).
    If a sample is certified by one method, it is not evaluated by the following methods.

    Parameters:
        model (auto_LiRPA.BoundedModule): The neural network model to be evaluated.
        eps (float): The perturbation bound.
        data_loader (torch.utils.data.DataLoader): DataLoader for the dataset to be evaluated.
        n_classes (int, optional): Number of classes in the dataset. Default is 10.
        test_samples (int, optional): Number of samples to test in order of the test loader. Default is np.inf (test all samples).
        device (str, optional): Device to run the evaluation on. Default is 'cuda'.

    Returns:
        (tuple): A tuple containing the number of certified samples, total number of images evaluated, and a tensor holding per-instance certification results.
    """
    eps = _to_device_tensor(eps, device)
    methods = _canonical_methods(methods)
    assert methods is not None and len(methods) >= 1, (
        "Please provide at least one bounding method!"
    )

    certified = torch.tensor([], device=device)
    total_images = 0

    crown_data_loader = DataLoader(data_loader.dataset, batch_size=1, shuffle=False)
    crown_data_loader.max, crown_data_loader.min, crown_data_loader.std = (
        data_loader.max,
        data_loader.min,
        data_loader.std,
    )

    results = {
        i: {"result": None, "method": None, "running_time": None}
        for i in range(min(len(data_loader.dataset), test_samples))
    }

    for batch_idx, (data, targets) in tqdm(enumerate(data_loader)):
        certified_idx = torch.zeros(len(data), device=device, dtype=torch.bool)
        data, targets = data.to(device), targets.to(device)
        data_min, data_max = data_loader.min.to(device), data_loader.max.to(device)

        ptb = PerturbationLpNorm(
            eps=eps,
            norm=np.inf,
            x_L=torch.clamp(data - eps, data_min, data_max),
            x_U=torch.clamp(data + eps, data_min, data_max),
        )

        if batch_idx * data_loader.batch_size >= test_samples:
            continue

        total_images += len(targets)

        if "IBP" in methods:
            start_time = time.time()
            lb, ub = bound_ibp(
                model=model,
                ptb=ptb,
                data=data,
                target=targets,
                n_classes=n_classes,
                reuse_input=False,
            )
            end_time = time.time()
            certified_idx[(lb > 0).all(dim=1)] = True

            results.update({
                batch_idx * data_loader.batch_size + i: {
                    "result": 'unsat' if (lb[i] > 0).all().item() else None,
                    "method": "IBP",
                    "running_time": (end_time - start_time) / len(targets),
                }
                for i in range(len(targets))
            })


        ptb = PerturbationLpNorm(
            eps=eps,
            norm=np.inf,
            x_L=torch.clamp(
                data[~certified_idx] - eps, data_min, data_max
            ),
            x_U=torch.clamp(
                data[~certified_idx] + eps, data_min, data_max
            ),
        )
        data = data.to(device)
        certified_idx = certified_idx.to(device)

        if torch.sum(~certified_idx) > 0 and "CROWN-IBP" in methods:
            start_time = time.time()
            lb, ub = bound_crown_ibp(
                model=model,
                ptb=ptb,
                data=data[~certified_idx],
                target=targets[~certified_idx],
                n_classes=n_classes,
                reuse_input=False,
            )
            end_time = time.time()
            uncertified_indices = torch.where(~certified_idx)[0]

            # Update certification status for the uncertified samples
            certified_idx[~certified_idx] = (lb > 0).all(dim=1)

            # Update results only for the samples that were actually evaluated by CROWN-IBP
            crown_ibp_results = {}
            for lb_idx, original_idx in enumerate(uncertified_indices):
                instance_id = batch_idx * data_loader.batch_size + original_idx.item()
                crown_ibp_results[instance_id] = {
                    "result": 'unsat' if (lb[lb_idx] > 0).all().item() else None,
                    "method": "CROWN-IBP",
                    "running_time": (end_time - start_time) / len(targets),
                }

            results.update(crown_ibp_results)

        certified = torch.cat((certified, certified_idx))

    print(
        f"certified {torch.sum(certified).item()} / {len(certified)} using IBP",
        flush=True,
    )

    for batch_idx, (data, targets) in tqdm(enumerate(crown_data_loader)):
        if batch_idx >= test_samples or not ("CROWN" in methods):
            break
        if certified[batch_idx]:
            continue

        data = data.to(device)
        ptb = PerturbationLpNorm(
            eps=eps,
            norm=np.inf,
            x_L=torch.clamp(data - eps, data_loader.min.to(device), data_loader.max.to(device)),
            x_U=torch.clamp(data + eps, data_loader.min.to(device), data_loader.max.to(device)),
        )
        data, targets = data.to(device), targets.to(device)
        start_time = time.time()
        lb, ub = bound_crown(
            model=model,
            ptb=ptb,
            data=data,
            target=targets,
            n_classes=n_classes,
            reuse_input=False,
        )
        end_time = time.time()
        instance_certified = (lb > 0).all(dim=1).item()
        certified[batch_idx] = instance_certified

        results[batch_idx] = {
            "result": 'unsat' if instance_certified else None,
            "method": "CROWN",
            "running_time": (end_time - start_time) / len(targets),
        }

    if test_samples < np.inf:
        certified = certified[:test_samples]
    no_certified = torch.sum(certified)
    total_images = len(certified)

    if "CROWN" in methods:
        print(
            f"certified {torch.sum(certified).item()} / {len(certified)} after using CROWN",
            flush=True,
        )

    return no_certified, total_images, certified, results


# TODO: can we maybe spare no_classes?
def eval_certified(
    model, data_loader, eps, n_classes=10, test_samples=np.inf, method="IBP"
):
    """
    Evaluate the certified robustness of a model using a given verification method.

    Parameters:
        model (auto_LiRPA.BoundedModule): The neural network model to be evaluated.
        data_loader (torch.utils.data.DataLoader): DataLoader for the dataset to be evaluated.
        n_classes (int, optional): Number of classes in the dataset. Default is 10.
        eps (float): Perturbation radius for certification.
        test_samples (int or float, optional): Number of test samples to evaluate. Default is np.inf (all samples).
        method (str or list, optional): The certification method to use. Options are 'IBP', 'CROWN', 'ALPHA-CROWN', 'CROWN-IBP', 'ADAPTIVE', 'COMPLETE', or a list of methods (which results in an ADAPTIVE evaluation using these methods). Default is 'IBP'.

    Returns:
        (float): The certified accuracy of the model on the test examples for the given epsilon.
        (dict): A dictionary containing per-instance certification results and running times.
    """
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    model.eval()
    certified = 0
    total_images = 0

    method = _canonical_methods(method)

    if method == "CROWN":
        certified, total_images, results = eval_crown(
            model, eps, data_loader, n_classes, test_samples, device
        )
    elif method == "ALPHA-CROWN":
        certified, total_images, results = eval_alpha_crown(
            model, eps, data_loader, n_classes, test_samples, device
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
            model, eps, data_loader, n_classes, test_samples, device, methods=method
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
            model, eps, data_loader, n_classes, test_samples, device, methods=method
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
):
    """
    Evaluate the adversarial robustness of a model using the PGD attack.

    Parameters:
        model (torch.nn.Module): The model to evaluate.
        data_loader (torch.utils.data.DataLoader): DataLoader providing the dataset.
        eps (torch.Tensor): The perturbation radius for the PGD attack.
        return_adv_indices (bool, optional): Whether to return the indices of adversarial samples. Default is False.
        restarts (int, optional): Number of random restarts for the PGD attack. Default is 5.
        step_size (float, optional): Step size for the PGD attack. Default is 0.1.
        n_steps (int, optional): Number of steps for the PGD attack. Default is 40.
        early_stopping (bool, optional): Whether to stop early if an adversarial example is found. Default is False.
        n_classes (int, optional): Number of classes in the dataset. Default is 10.
        device (str, optional): Device to perform computations on. Default is 'cuda'.
        test_samples (int or float, optional): Number of samples to test. Default is np.inf.

    Returns:
        (float): Adversarial accuracy of the model.
        (np.ndarray): Indices of unsafe indices if return_adv_indices is True.
    """
    model.eval()
    adv_preds = np.array([])
    labels = np.array([])
    data_min = data_loader.min.to(device)
    data_max = data_loader.max.to(device)
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
):
    """
    Evaluate the model on standard, certified, and adversarial accuracy.

    Args:
        model (auto_LiRPA.BoundedModule): The model to be evaluated.
        data_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        eps (float): Perturbation size for adversarial and certified evaluation.
        n_classes (int, optional): Number of classes in the dataset. Default is 10.
        test_samples (int or float, optional): Number of samples to test. Default is np.inf (all samples).
        method (str or list, optional): The certification method to use. Options are 'IBP', 'CROWN', 'CROWN-IBP', 'ADAPTIVE', 'COMPLETE', or a list of methods (which results in an ADAPTIVE evaluation using these methods). Default is 'ADAPTIVE'.
        device (str, optional): Device to perform adversarial evaluation on. Default is 'cuda'.

    Returns:
        tuple: (std_acc (float): Standard accuracy of the model, cert_acc (float): Certified accuracy of the model, adv_acc (float): Adversarial accuracy of the model)
    """
    std_acc = eval_acc(model, test_loader=data_loader, test_samples=test_samples)
    cert_acc, _ = eval_certified(
        model=model,
        data_loader=data_loader,
        n_classes=n_classes,
        eps=eps,
        test_samples=test_samples,
        method=method,
    )
    adv_acc = eval_adversarial(
        model=model,
        data_loader=data_loader,
        eps=eps,
        n_classes=n_classes,
        device=device,
        test_samples=test_samples,
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
    """
    Evaluate the model during training. It computes the standard accuracy, certified accuracy,
    and adversarial accuracy of the model. The results are saved to a specified path in JSON format.

    Parameters:
        model (torch.nn.Module): The model to be evaluated.
        data_loader (torch.utils.data.DataLoader): DataLoader for the dataset to evaluate.
        eps (torch.Tensor): Perturbation radius used during evaluation of standard, certified and adversarial accuracy.
        n_classes (int): Number of classes in the dataset.
        device (str, optional): Device to run the evaluation on. Default is 'cuda'.
        test_samples (int, optional): Number of samples to use for evaluation. Default is 1000.
        verification_method (str, optional): Method to use for certification. Default is "IBP".
        results_path (str, optional): Path to save the evaluation results. Default is "./results".

    Returns:
        tuple: A tuple containing standard accuracy, certified accuracy, and adversarial accuracy.
    """
    os.makedirs(results_path, exist_ok=True)
    model.eval()
    std_acc = eval_acc(model, test_loader=data_loader, test_samples=test_samples)
    if (eps == 0.0).all():
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
    model.train()

    return std_acc, cert_acc, adv_acc
