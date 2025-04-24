import json
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

import shutil

from auto_LiRPA import PerturbationLpNorm
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from CTRAIN.bound import bound_ibp, bound_crown, bound_crown_ibp
from CTRAIN.attacks import pgd_attack
from CTRAIN.complete_verification.abCROWN.util import instances_to_vnnlib, get_abcrown_standard_conf
from CTRAIN.complete_verification.abCROWN.verify import limited_abcrown_eval, abcrown_eval
from CTRAIN.util import export_onnx

def eval_acc(model, test_loader, test_samples=np.inf):
    """
    Evaluate the accuracy of a given model on a test dataset.
    
    Args:
        model (torch.nn.Module): The model to be evaluated.
        test_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        test_samples (int, optional): Number of samples to test in order of the test loader. Default is np.inf (test all samples).
    
    Returns:
        (float): The accuracy of the model on the test dataset.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            if total >= test_samples:
                break
            batch_indices = min(len(target), test_samples - total)
            data = data[:batch_indices]
            target = target[:batch_indices]
            
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)

            correct += (predicted == target).sum().item()

    test_samples = min(test_samples, total)
    # print(f'Accuracy of the standard model on the first {test_samples} test images: {correct / test_samples:.4f}')
    return correct / test_samples

def eval_ibp(model, eps, data_loader, n_classes=10, test_samples=np.inf, device='cuda'):
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
        (tuple): A tuple containing the number of certified samples and the total number of images evaluated.
    """
    certified = 0
    total_images = 0
    for batch_idx, (data, targets) in tqdm(enumerate(data_loader)):
        if total_images >= test_samples:
            continue
        
        ptb = PerturbationLpNorm(eps=eps, norm=np.inf, x_L=torch.clamp(data - eps, data_loader.min, data_loader.max).to(device), x_U=torch.clamp(data + eps, data_loader.min, data_loader.max).to(device))
        data, targets = data.to(device), targets.to(device)

        lb, ub = bound_ibp(
            model=model,
            ptb=ptb,
            data=data,
            target=targets,
            n_classes=n_classes,
            reuse_input=False
        )
        no_certified = torch.sum((lb > 0).all(dim=1)).item()
        # no_falsified = torch.sum((ub < 0).any(dim=1)).item()
        certified += no_certified
        
        total_images += len(targets)
    
    return certified, total_images

def eval_crown_ibp(model, eps, data_loader, n_classes=10, test_samples=np.inf, device='cuda'):
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
        (tuple): A tuple containing the number of certified samples and the total number of images evaluated.
    """
    certified = 0
    total_images = 0
    for batch_idx, (data, targets) in tqdm(enumerate(data_loader)):
        if total_images >= test_samples:
            continue
        
        ptb = PerturbationLpNorm(eps=eps, norm=np.inf, x_L=torch.clamp(data - eps, data_loader.min, data_loader.max).to(device), x_U=torch.clamp(data + eps, data_loader.min, data_loader.max).to(device))
        data, targets = data.to(device), targets.to(device)
        
        lb, ub = bound_crown_ibp(
            model=model,
            ptb=ptb,
            data=data,
            target=targets,
            n_classes=n_classes,
            reuse_input=False
        )
        no_certified = torch.sum((lb > 0).all(dim=1)).item()
        # no_falsified = torch.sum((ub < 0).any(dim=1)).item()
        certified += no_certified
        
        total_images += len(targets)
    
    return certified, total_images


def eval_crown(model, eps, data_loader, n_classes=10, test_samples=np.inf, device='cuda'):
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
        (tuple): A tuple containing the number of certified samples and the total number of images evaluated.
    """
    # IMPORTANT: Data Loader Batch Size must match Bounding Batch Size when using CROWN for evaluation (not important for IBP)
    crown_data_loader = DataLoader(data_loader.dataset, batch_size=1, shuffle=False)
    crown_data_loader.max, crown_data_loader.min, crown_data_loader.std = data_loader.max, data_loader.min, data_loader.std
    certified = 0
    total_images = 0
    for batch_idx, (data, targets) in tqdm(enumerate(crown_data_loader)):
        if total_images >= test_samples:
            continue
        ptb = PerturbationLpNorm(eps=eps, norm=np.inf, x_L=torch.clamp(data - eps, data_loader.min, data_loader.max).to(device), x_U=torch.clamp(data + eps, data_loader.min, data_loader.max).to(device))
        data, targets = data.to(device), targets.to(device)
        lb, ub = bound_crown(
            model=model,
            ptb=ptb,
            data=data,
            target=targets,
            n_classes=n_classes,
            reuse_input=False
        )
        no_certified = torch.sum((lb > 0).all(dim=1)).item()
        # no_falsified = torch.sum((ub < 0).any(dim=1)).item()
        certified += no_certified
        
        total_images += len(targets)
    
    return certified, total_images


def eval_complete_abcrown(model, eps_std, data_loader, n_classes=10, input_shape=[1, 28, 28], test_samples=np.inf, timeout=1000, no_cores=28, abcrown_batch_size=512, abcrown_config_dict=dict(), separate_abcrown_process=False, device='cuda'):
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
        abcrown_config_dict (dict, optional): Configuration dictionary for the ABCROWN verification system. Default is an empty dictionary.
        separate_abcrown_process (bool, optional): Whether to run ABCROWN in a separate process. Default is False.
        device (str, optional): Device to run the evaluation on. Default is 'cuda'.

    Returns:
        (tuple): A tuple containing the certified accuracy and the adversarial accuracy.
    """
    no_certified, total_images, certified = eval_adaptive(model=model, eps=eps_std, data_loader=data_loader, n_classes=n_classes, test_samples=test_samples, device=device)
    adv_acc, adv_sample_found = eval_adversarial(model=model, data_loader=data_loader, eps=eps_std, n_classes=n_classes, device=device, test_samples=test_samples, return_adv_indices=True)
    adv_sample_found = torch.tensor(adv_sample_found)
    total_images = 0
        
    batch_size = data_loader.batch_size
    
    std_config = get_abcrown_standard_conf(timeout=timeout, no_cores=no_cores)
    std_config['solver']['batch_size'] = abcrown_batch_size
    
    def update_config(base_config, custom_config):
        for key, value in custom_config.items():
            if isinstance(value, dict) and key in base_config:
                update_config(base_config[key], value)
            else:
                base_config[key] = value

    update_config(std_config, abcrown_config_dict)
    
    
    os.makedirs('/tmp/abCROWN/', exist_ok=True)
    
    export_onnx(
        model=model,
        file_name='/tmp/abCROWN_model.onnx',
        batch_size=1, 
        input_shape=input_shape
    )
    
    for batch_idx, (data, targets) in tqdm(enumerate(data_loader)):
        if total_images >= test_samples:
            break
        
        batch_indices = min(len(targets), test_samples - total_images)
        data = data[:batch_indices]
        targets = targets[:batch_indices]
        total_images += len(targets)
        
        print(f"BATCH {batch_idx}")
        
        clean_pred = torch.argmax(model(data.to(device)), dim=1)
        clean_correct = clean_pred.cpu() == targets
        
        if certified[batch_idx * batch_size:(batch_idx + 1) * batch_size].all():
            continue
        
        os.makedirs(f'/tmp/vnnlib_{batch_idx}/', exist_ok=True)
        
        vnnlib_batch = instances_to_vnnlib(
            indices=[i for i in range(len(targets)) if not certified[batch_idx * batch_size + i] and not adv_sample_found[batch_idx * batch_size + i] and clean_correct[i]],
            data=[(img, target) for img, target in zip(data, targets)],
            vnnlib_path=f'/tmp/vnnlib_{batch_idx}/',
            experiment_name='Experiment',
            eps=eps_std * data_loader.std,
            eps_temp=eps_std,
            data_min=data_loader.min,
            data_max=data_loader.max,
            no_classes=n_classes
        )
        vnnlib_indices = [batch_idx * batch_size + i for i in range(len(targets)) if not certified[batch_idx * batch_size + i] and not adv_sample_found[batch_idx * batch_size + i] and clean_correct[i]]
        print(vnnlib_indices)
        for idx, vnn_instance in zip(vnnlib_indices, vnnlib_batch):
            if separate_abcrown_process:
                running_time, result = limited_abcrown_eval(
                    # work_dir='/tmp/abCROWN',
                    config=std_config,
                    seed=42,
                    instance=vnn_instance,
                    vnnlib_path=f'/tmp/vnnlib_{batch_idx}',
                    model_path=None,
                    model_name=None,
                    model_onnx_path='/tmp/abCROWN_model.onnx',
                    input_shape=[-1] + input_shape[1:4],
                    timeout=timeout,
                    no_cores=no_cores,
                    par_factor=1
                )
            else:
                running_time, result = abcrown_eval(
                    # work_dir='/tmp/abCROWN',
                    config=std_config,
                    seed=42,
                    instance=vnn_instance,
                    vnnlib_path=f'/tmp/vnnlib_{batch_idx}',
                    model_path=None,
                    model_name=None,
                    model_onnx_path='/tmp/abCROWN_model.onnx',
                    input_shape=[-1] + input_shape[1:4],
                    timeout=timeout,
                    no_cores=no_cores,
                    par_factor=1
                )
            print(running_time, result)
            if result == 'unsat':
                no_certified += 1
                certified[idx] = True
            if result == 'sat':
                adv_sample_found[idx] = True
        
        shutil.rmtree(f"/tmp/vnnlib_{batch_idx}/")
    
    if test_samples < np.inf:
        certified = certified[:test_samples]
    no_certified = torch.sum(certified)    
    no_counterexample = torch.sum(adv_sample_found)
            
    certified_acc = (no_certified / test_samples).cpu().item() if torch.is_tensor(certified) else certified / test_samples
    adv_acc = (test_samples  - no_counterexample) / test_samples
    if torch.is_tensor(adv_acc):
        adv_acc = adv_acc.item()
    
    return certified_acc, adv_acc

def eval_adaptive(model, eps, data_loader, n_classes=10, test_samples=np.inf, device='cuda', methods=["IBP", "CROWN_IBP", "CROWN"]):
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
    assert methods is not None and len(methods) > 1, "Please provide at least one bounding method!"

    certified = torch.tensor([], device=device)
    total_images = 0
    
    crown_data_loader = DataLoader(data_loader.dataset, batch_size=1, shuffle=False)
    crown_data_loader.max, crown_data_loader.min, crown_data_loader.std = data_loader.max, data_loader.min, data_loader.std
    
    for batch_idx, (data, targets) in tqdm(enumerate(data_loader)):
        certified_idx = torch.zeros(len(data), device=device, dtype=torch.bool)
        
        ptb = PerturbationLpNorm(eps=eps, norm=np.inf, x_L=torch.clamp(data - eps, data_loader.min, data_loader.max).to(device), x_U=torch.clamp(data + eps, data_loader.min, data_loader.max).to(device))
        data, targets = data.to(device), targets.to(device)
        if batch_idx * data_loader.batch_size >= test_samples:
            continue
        
        total_images += len(targets)
        
        if "IBP" in methods:
            lb, ub = bound_ibp(
                model=model,
                ptb=ptb,
                data=data,
                target=targets,
                n_classes=n_classes,
                reuse_input=False
            )
            certified_idx[(lb > 0).all(dim=1)] = True
        
        data = data.to('cpu')
        certified_idx = certified_idx.to("cpu")
        ptb = PerturbationLpNorm(eps=eps, norm=np.inf, x_L=torch.clamp(data[~certified_idx] - eps, data_loader.min, data_loader.max).to(device), x_U=torch.clamp(data[~certified_idx] + eps, data_loader.min, data_loader.max).to(device))
        data = data.to(device)
        certified_idx = certified_idx.to(device)
        
        if len(certified_idx) < len(targets) and "CROWN-IBP" in methods:        
            lb, ub = bound_crown_ibp(
                model=model,
                ptb=ptb,
                data=data[~certified_idx],
                target=targets[~certified_idx],
                n_classes=n_classes,
                reuse_input=False
            )
            certified_idx[~certified_idx] = (lb > 0).all(dim=1)
            
        certified = torch.concatenate((certified, certified_idx))
    
    print(f"certified {torch.sum(certified).item()} / {len(certified)} using IBP", flush=True)
    
    for batch_idx, (data, targets) in tqdm(enumerate(crown_data_loader)):
        if batch_idx >= test_samples:
            continue
        if certified[batch_idx] or not ("CROWN" in methods):
            continue
        
        ptb = PerturbationLpNorm(eps=eps, norm=np.inf, x_L=torch.clamp(data - eps, data_loader.min, data_loader.max).to(device), x_U=torch.clamp(data + eps, data_loader.min, data_loader.max).to(device))
        data, targets = data.to(device), targets.to(device)
        
        lb, ub = bound_crown(
            model=model,
            ptb=ptb,
            data=data,
            target=targets,
            n_classes=n_classes,
            reuse_input=False
        )
        instance_certified = (lb > 0).all(dim=1).item()
        certified[batch_idx] = instance_certified
    
    if test_samples < np.inf:
        certified = certified[:test_samples]
    no_certified = torch.sum(certified)            
    total_images = len(certified)
    
    if 'CROWN' in methods:
        print(f"certified {torch.sum(certified).item()} / {len(certified)} after using CROWN", flush=True)
    
    return no_certified, total_images, certified

# TODO: can we maybe spare no_classes?
def eval_certified(model, data_loader, eps, n_classes=10, test_samples=np.inf, method='IBP'):
    """
    Evaluate the certified robustness of a model using a given verification method.
    
    Parameters:
        model (auto_LiRPA.BoundedModule): The neural network model to be evaluated.
        data_loader (torch.utils.data.DataLoader): DataLoader for the dataset to be evaluated.
        n_classes (int, optional): Number of classes in the dataset. Default is 10.
        eps (float): Perturbation radius for certification.
        test_samples (int or float, optional): Number of test samples to evaluate. Default is np.inf (all samples).
        method (str or list, optional): The certification method to use. Options are 'IBP', 'CROWN', 'CROWN-IBP', 'ADAPTIVE', 'COMPLETE', or a list of methods (which results in an ADAPTIVE evaluation using these methods). Default is 'IBP'.
    
    Returns:
        (float): The certified accuracy of the model on the test examples for the given epsilon.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model.eval()
    certified = 0
    total_images = 0

    if method == "CROWN":
        certified, total_images = eval_crown(model, eps, data_loader, n_classes, test_samples, device)
    elif method == 'IBP':
        certified, total_images = eval_ibp(model, eps, data_loader, n_classes, test_samples, device)
    elif method == 'CROWN-IBP':
        certified, total_images = eval_crown_ibp(model, eps, data_loader, n_classes, test_samples, device)
    elif method == 'ADAPTIVE':
        certified, total_images, cert_results = eval_adaptive(model, eps, data_loader, n_classes, test_samples, device)
    elif isinstance(method, list):
        certified, total_images, cert_results = eval_adaptive(model, eps, data_loader, n_classes, test_samples, device, methods=method)
    elif method == 'COMPLETE':
        # TODO: Infer input shape or pass it, pass timeout and no_cores!
        eval_complete_abcrown(model, eps, data_loader, n_classes, input_shape=(1, 28, 28), test_samples=test_samples, timeout=1000, no_cores=28, device=device)
    elif isinstance(method, (list, tuple, np.ndarray)):
        certified, total_images, cert_results = eval_adaptive(model, eps, data_loader, n_classes, test_samples, device, methods=method)
    else:
        assert False, "UNKNOWN BOUNDING METHOD!"
    
    return (certified / total_images).cpu().item() if torch.is_tensor(certified) else certified / total_images


def eval_adversarial(model, data_loader, eps, return_adv_indices=False, restarts=5, step_size=0.1, n_steps=40, early_stopping=False, n_classes=10, device='cuda', test_samples=np.inf,):
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
    
    for batch_idx, (data, targets) in tqdm(enumerate(data_loader)):
        if len(labels) >= test_samples:
            break
        
        batch_indices = min(len(targets), test_samples - len(labels))
        data = data[:batch_indices]
        targets = targets[:batch_indices]
        
        data, targets = data.to(device), targets.to(device)
        eps = eps.to(device)

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
            device=device
        )
        
        adv_predictions_batch = model(x_test_adv)
        adv_predictions_batch = torch.argmax(adv_predictions_batch, dim=1).cpu().numpy()
        if len(labels) + len(targets) > test_samples:
            too_many_samples_no = (len(labels) + len(targets)) % test_samples
            adv_predictions_batch = adv_predictions_batch[:-too_many_samples_no]
            targets = targets[:-too_many_samples_no]

        adv_preds = np.append(adv_preds, adv_predictions_batch)
        labels = np.append(labels, targets.cpu())

    test_samples = min(test_samples, len(labels))

    adv_accuracy = accuracy_score(labels, adv_preds)
    
    if return_adv_indices:
        adv_sample_found = labels != adv_preds
        return adv_accuracy, adv_sample_found

    return adv_accuracy

def eval_model(model, data_loader, eps, n_classes=10, test_samples=np.inf, method='ADAPTIVE', device='cuda'):
    """
    Evaluate the model on standard, certified, and adversarial accuracy.
    
    Parameters:
        model (auto_LiRPA.BoundedModule): The model to be evaluated.
        data_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        n_classes (int, optional): Number of classes in the dataset. Default is 10.
        eps (float, optional): Perturbation size for adversarial and certified evaluation.
        test_samples (int or float, optional): Number of samples to test. Default is np.inf (all samples).
        method (str or list, optional): The certification method to use. Options are 'IBP', 'CROWN', 'CROWN-IBP', 'ADAPTIVE', 'COMPLETE', or a list of methods (which results in an ADAPTIVE evaluation using these methods). Default is 'IBP'.
        device (str, optional): Device to perform adversarial evaluation on. Default is 'cuda'.
        
    Returns:
        tuple: A tuple containing std_acc (float): Standard accuracy of the model, cert_acc (float): Certified accuracy of the model, adv_acc (float): Adversarial accuracy of the model.
    """
    std_acc = eval_acc(model, test_loader=data_loader, test_samples=test_samples)
    cert_acc = eval_certified(model=model, data_loader=data_loader, n_classes=n_classes, eps=eps, test_samples=test_samples, method=method)
    adv_acc = eval_adversarial(model=model, data_loader=data_loader, eps=eps, n_classes=n_classes, device=device, test_samples=test_samples, restarts=30, n_steps=100, step_size=.1, early_stopping=True)
    
    return std_acc, cert_acc, adv_acc
    
def eval_epoch(model, data_loader, eps, n_classes, device='cuda', test_samples=1000, verification_method="IBP", results_path="./results"):
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
    if (eps == 0.).all():
        cert_acc = adv_acc = std_acc
    else:
        with torch.no_grad():
            cert_acc = eval_certified(model=model, data_loader=data_loader, n_classes=n_classes, eps=eps, test_samples=test_samples, method=verification_method)
            adv_acc = eval_adversarial(model=model, data_loader=data_loader, n_classes=n_classes, eps=eps, device=device, test_samples=test_samples)
    
    with open(f"{results_path}/stats.json", "w") as f:
        json.dump(
            {"acc": std_acc, "cert_acc": cert_acc, "adv_acc": adv_acc}, f
        )
    model.train()

    return std_acc, cert_acc, adv_acc 
    
     
