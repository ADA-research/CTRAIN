import csv
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from CTRAIN.model_definitions import CNN7_Shi, CNN5_Mao, CNN9_Mao
from CTRAIN.data_loaders import load_cifar10, load_mnist, load_tinyimagenet
from CTRAIN.model_wrappers import ShiIBPModelWrapper, SABRModelWrapper, CrownIBPModelWrapper, MTLIBPModelWrapper

import torch

PAPER_ROOT = Path(__file__).resolve().parents[1]
PARETO_FRONTS_PATH = PAPER_ROOT / "results" / "hpo" / "validation" / "pareto_fronts"
HPO_STUDY_ROOT = Path(
    os.environ.get(
        "CTRAIN_PAPER_HPO_ROOT",
        PAPER_ROOT / "results" / "hpo" / "validation" / "optuna_studies",
    )
)
NAT_ACC_RESULTS_PATH = PAPER_ROOT / "results" / "verification" / "clean_accuracy"
DATA_ROOT = os.environ.get(
    "CTRAIN_DATA_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data")),
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EXPERIMENTS = [
    # (architecture, dataset, eps)
    ("cnn7", "cifar10", 8/255),
    ("cnn7", "cifar10", 2/255),
    ("wide_cnn7", "cifar10", 2/255),
    ("narrow_cnn7", "cifar10", 2/255),
    ("cnn5", "cifar10", 2/255),
    ("cnn9", "cifar10", 2/255),
    ("cnn7", "tinyimagenet", 1/255),
    ("cnn7", "mnist", 0.3),
]

METHODS = [
    "shi",
    "crown_ibp",
    "crown_ibp_nofusion",
    "sabr",
    "mtl_ibp"
]

def parse_front(file_path):
    networks = {}
    with file_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("subselected", "true").lower() not in {"true", "1"}:
                continue
            config_hash = row["config_hash"]
            network_path = HPO_STUDY_ROOT / row["source_study"] / "nets" / f"{config_hash}.pt"
            if network_path.exists():
                print(f"Found network at: {network_path}")
                networks[config_hash] = network_path
            else:
                print(f"Checkpoint not found: {network_path}")
    return networks

def wrap_model(model, method, in_shape, eps):
    if method == "shi":
        return ShiIBPModelWrapper(model, input_shape=in_shape, eps=eps, num_epochs=160, device=DEVICE)
    elif method == "crown_ibp":
        return CrownIBPModelWrapper(model, loss_fusion=True, input_shape=in_shape, eps=eps, num_epochs=160, device=DEVICE)
    elif method == "crown_ibp_nofusion":
        return CrownIBPModelWrapper(model, loss_fusion=False, input_shape=in_shape, eps=eps, num_epochs=160, device=DEVICE)
    elif method == "sabr":
        return SABRModelWrapper(model, input_shape=in_shape, eps=eps, num_epochs=160, device=DEVICE)
    elif method == "mtl_ibp":
        return MTLIBPModelWrapper(model, input_shape=in_shape, eps=eps, num_epochs=160, device=DEVICE)
    else:
        raise ValueError(f"Unknown method: {method}")
    

def get_model(architecture, in_shape, n_classes):
    if architecture == "cnn7":
        model = CNN7_Shi(in_shape=in_shape, n_classes=n_classes)
    elif architecture == "wide_cnn7":
        model = CNN7_Shi(in_shape=in_shape, n_classes=n_classes, width=128)
    elif architecture == "narrow_cnn7":
        model = CNN7_Shi(in_shape=in_shape, n_classes=n_classes, width=32)
    elif architecture == "cnn5":
        model = CNN5_Mao(in_shape=in_shape, n_classes=n_classes)
    elif architecture == "cnn9":
        model = CNN9_Mao(in_shape=in_shape, n_classes=n_classes)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    
    return model


def get_nat_acc(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    results = {}
    image_idx = 0
    
    for images, labels in data_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Add to results dict with image index as key
        for p, l in zip(predicted, labels):
            results[image_idx] = bool(p == l)
            image_idx += 1
        
    return correct / total, results


def eval_nat_acc():
    NAT_ACC_RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    for architecture, dataset, eps in EXPERIMENTS:
        for method in METHODS:
            pareto_front_file = PARETO_FRONTS_PATH / f"pareto_front_{method}_{architecture}_{dataset}_{eps}.csv"
            if not pareto_front_file.exists():
                print(f"Pareto front file not found: {pareto_front_file}, skipping.")
                continue
            networks = parse_front(pareto_front_file)
            print(f"Evaluating natural accuracy for {method}_{architecture}_{dataset}_{eps}")
            if dataset == "cifar10":
                _, test_loader = load_cifar10(batch_size=1024, data_root=DATA_ROOT, val_split=False)
                in_shape = (3, 32, 32)
                n_classes = 10
            elif dataset == "mnist":
                _, test_loader = load_mnist(batch_size=512, data_root=DATA_ROOT, val_split=False)
                in_shape = (1, 28, 28)
                n_classes = 10
            elif dataset == "tinyimagenet":
                _, test_loader = load_tinyimagenet(batch_size=256, data_root=DATA_ROOT, val_split=False)
                in_shape = (3, 64, 64)
                n_classes = 200
            else:
                raise ValueError(f"Unknown dataset: {dataset}")
            
            for hash, network_path in networks.items():
                model = get_model(architecture, in_shape, n_classes)
                model = wrap_model(model, method, in_shape, eps)
                model.load_state_dict(torch.load(network_path, map_location=DEVICE))
                model.eval()
                nat_acc, results_json = get_nat_acc(model, test_loader)
                print(f"Hash: {hash}, Natural Accuracy: {nat_acc}")
                
                output_path = NAT_ACC_RESULTS_PATH / f'{dataset}_{architecture}_{method}{eps}_{hash}_nat_acc.json'
                with output_path.open('w') as f:
                    import json
                    json.dump({
                        "std_acc": nat_acc,
                        "results": results_json
                    }, f, indent=4)

if __name__ == "__main__":
    eval_nat_acc()
