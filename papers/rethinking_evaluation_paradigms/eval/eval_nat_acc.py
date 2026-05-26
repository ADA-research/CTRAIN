import os
from CTRAIN.model_definitions import CNN7_Shi, CNN5_Mao, CNN9_Mao
from CTRAIN.data_loaders import load_cifar10, load_mnist, load_tinyimagenet
from CTRAIN.model_wrappers import ShiIBPModelWrapper, SABRModelWrapper, CrownIBPModelWrapper, MTLIBPModelWrapper

import torch

HPO_RESULTS_PATH = "../results/hpo"
VERIFICATION_RESULTS_PATH = "../results/verification"
NAT_ACC_RESULTS_PATH = "../results/clean_classification"
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
            continue
            
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
    os.makedirs(NAT_ACC_RESULTS_PATH, exist_ok=True)
    for architecture, dataset, eps in EXPERIMENTS:
        for method in METHODS:
            pareto_front_file =   f"{HPO_RESULTS_PATH}/pareto_fronts/pareto_front_{method}_{architecture}_{dataset}_{eps}.txt"
            if not os.path.exists(pareto_front_file):
                print(f"Pareto front file not found: {pareto_front_file}, skipping.")
                continue
            hashes = parse_results_file(pareto_front_file)
            nets_folder_prefix = f"{HPO_RESULTS_PATH}/{dataset}_{architecture}_{method}{eps}"
            networks = get_networks(nets_folder_prefix, hashes)
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
                model.load_state_dict(torch.load(network_path))
                model.eval()
                nat_acc, results_json = get_nat_acc(model, test_loader)
                print(f"Hash: {hash}, Natural Accuracy: {nat_acc}")
                
                with open(f'{NAT_ACC_RESULTS_PATH}/{dataset}_{architecture}_{method}{eps}_{hash}_nat_acc.json', 'w') as f:
                    import json
                    json.dump({
                        "std_acc": nat_acc,
                        "results": results_json
                    }, f, indent=4)

if __name__ == "__main__":
    eval_nat_acc()
