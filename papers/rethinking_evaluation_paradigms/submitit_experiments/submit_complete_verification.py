import json
import os
import submitit
import torch
import numpy as np
from CTRAIN.model_definitions import CNN7_Shi, CNN5_Mao, CNN9_Mao
from torchvision.models import resnet18
from CTRAIN.model_wrappers import *
from CTRAIN.data_loaders import load_cifar10
from CTRAIN.util import seed_ctrain

PAPER_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_ROOT = os.environ.get("CTRAIN_DATA_ROOT", os.path.abspath(os.path.join(PAPER_ROOT, "..", "data")))
RESULTS_ROOT = os.environ.get("CTRAIN_PAPER_RESULTS_ROOT", os.path.join(PAPER_ROOT, "results"))

def parse_results_file(file_path):
    hashes = []
    
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    for line in lines:
        if not "Config hash" in line:
            continue
    
        hash = line.split("Config hash: ")[1].strip()
        print(f"Config hash: {hash}") 
    
        hashes.append(hash)
    
    return hashes


def get_networks(nets_folder_prefix, hashes):
    networks = {}
    
    for hash in hashes:
        if os.path.exists(f"{nets_folder_prefix}/{hash}.pt"):            
            network_path = f"{nets_folder_prefix}/{hash}.pt"
            print(f"Found network at: {network_path}")
            networks[hash] = network_path
            
    return networks

def get_model_wrapper(method):
    if method == "shi":
        from CTRAIN.model_wrappers import ShiIBPModelWrapper
        return ShiIBPModelWrapper
    elif method == "mtl_ibp":
        from CTRAIN.model_wrappers import MTLIBPModelWrapper
        return MTLIBPModelWrapper
    elif method == "sabr":
        from CTRAIN.model_wrappers import SABRModelWrapper
        return SABRModelWrapper
    elif method in ['crown_ibp', 'crown_ibp_nofusion']:
        from CTRAIN.model_wrappers import CrownIBPModelWrapper
        return CrownIBPModelWrapper
    else:
        raise ValueError(f"Unknown certification method: {method}")

def eval_complete(results_path, model_path, cert_train_method, eps, dataset="cifar10", architecture='cnn7'):
    seed_ctrain(seed=42)

    if dataset == "cifar10":
        train_loader, test_loader = load_cifar10(
            batch_size=512, val_split=False, data_root=DATA_ROOT
        )
        in_shape = [3, 32, 32]
        n_classes = 10
    elif dataset == "mnist":
        from CTRAIN.data_loaders import load_mnist
        train_loader, test_loader = load_mnist(
            batch_size=512, val_split=False, data_root=DATA_ROOT
        )
        in_shape = [1, 28, 28]
        n_classes = 10
    elif dataset == 'tinyimagenet':
        from CTRAIN.data_loaders import load_tinyimagenet
        train_loader, test_loader = load_tinyimagenet(
            batch_size=64, val_split=False, data_root=DATA_ROOT
        )
        in_shape = [3, 64, 64]
        n_classes = 200
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    
    if architecture == "cnn7":
        model = CNN7_Shi(in_shape=in_shape, n_classes=n_classes)
        abcrown_batch_size = 1024
    elif architecture == "wide_cnn7":
        model = CNN7_Shi(in_shape=in_shape, n_classes=n_classes, width=128)
        abcrown_batch_size = 512
    elif architecture == "narrow_cnn7":
        model = CNN7_Shi(in_shape=in_shape, n_classes=n_classes, width=32)
        abcrown_batch_size = 1024
    elif architecture == "cnn5":
        model = CNN5_Mao(in_shape=in_shape, n_classes=n_classes)
        abcrown_batch_size = 1024
    elif architecture == "cnn9":
        model = CNN9_Mao(in_shape=in_shape, n_classes=n_classes)
        abcrown_batch_size = 512
    elif architecture == "resnet18":
        model = resnet18(num_classes=n_classes)
        model.conv1 = torch.nn.Conv2d(
            in_shape[0], 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        model.maxpool = torch.nn.Identity()
        abcrown_batch_size = 256
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    
    if architecture == 'cnn7' and dataset in ['cifar10', 'tinyimagenet']:
        timeout = 1000
    else:
        timeout = 300

    wrapped_model = get_model_wrapper(cert_train_method)(
        model=model,
        input_shape=in_shape,
        device=torch.device("cuda"),
        eps=eps,
        num_epochs=160,
    )

    wrapped_model.load_state_dict(
        torch.load(
            model_path
        ) 
    )
    wrapped_model.eval()
    print(
        wrapped_model.evaluate_complete(test_loader, timeout=timeout, abcrown_batch_size=abcrown_batch_size, test_samples=10_000, results_path=results_path, warm_start=True)
    )

if __name__ == "__main__":
    executor = submitit.AutoExecutor(folder="./submitit_logs")
    executor.update_parameters(
        timeout_min=60 * 24 * 50,
        slurm_partition="CLUSTER",
        gpus_per_node=1,
        slurm_array_parallelism=13,
        cpus_per_task=14,
        mem_gb=15.7 * 14,
        slurm_additional_parameters={"qos": "gpu", },
        slurm_job_name="MOCTRAIN",
        slurm_setup=['module load CUDA/12.1.1', 'module load Python/3.11']
    )
    
    with executor.batch():
        for method in ["mtl_ibp", "sabr", "shi", "crown_ibp_nofusion", "crown_ibp"]:
            datasets = ['cifar10', 'tinyimagenet', 'mnist']
            for network in [
                "cnn7",
                "wide_cnn7",
                "cnn5",
                "cnn9",
                "narrow_cnn7",
            ]:
                for dataset in datasets:
                    for eps in [1 / 255, 2 / 255, 8 / 255, 0.3]:
                            if os.path.exists(f'{RESULTS_ROOT}/hpo/pareto_fronts/pareto_front_{method}_{network}_{dataset}_{eps}_subselected0.05.txt'):
                                results_path_subselected = f'{RESULTS_ROOT}/hpo/pareto_fronts/pareto_front_{method}_{network}_{dataset}_{eps}_subselected0.05.txt'
                            elif os.path.exists(f'{RESULTS_ROOT}/hpo/pareto_fronts/pareto_front_{method}_{network}_{dataset}_{eps}.txt'):
                                results_path_subselected = f'{RESULTS_ROOT}/hpo/pareto_fronts/pareto_front_{method}_{network}_{dataset}_{eps}.txt'
                            else:
                                print(f"File for {method}_{network}_{dataset}_{eps} not found, skipping.")
                                continue
                            print("Processing file:", results_path_subselected)
                            hashes = parse_results_file(results_path_subselected)
                            print(f"Found {len(hashes)} hashes for {method} on {network} with eps {eps} on {dataset}")
                            
                            nets_folder_prefix = f"{RESULTS_ROOT}/hpo/{dataset}_{network}_{method}{eps}"
                            
                            networks = get_networks(nets_folder_prefix, hashes)
                            print(f"Found {len(networks)} networks for {method} on {network} with eps {eps} on {dataset}")
                            for hash, model_path in networks.items():
                                print(f"Evaluating {method} on {network} with hash {hash} and eps {eps}")
                                results_path = f'{RESULTS_ROOT}/verification/{dataset}/{network}/{eps}/{method}/{hash}'
                                if os.path.exists(f"{results_path}/results.json"):
                                    with open(f"{results_path}/results.json", "r") as f:
                                        results = json.load(f)
                                    if len([i for i in results if results[i]['result'] is None]) == 0:
                                        print(f"Results already exist at {results_path}, skipping.")
                                        continue
                                # executor.submit(eval_complete, results_path, model_path, method, eps, dataset, network)
                                print("THIS WOULD BE A SUBMITIT JOB, BUT WE ARE NOT SUBMITTING IT NOW")
