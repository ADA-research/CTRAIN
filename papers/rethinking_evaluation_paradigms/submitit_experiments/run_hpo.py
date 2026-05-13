import os

import numpy as np
import submitit
import torch
from torchvision.models import resnet18

from CTRAIN.data_loaders import load_cifar10, load_gtsrb, load_mnist, load_tinyimagenet
from CTRAIN.model_definitions import (
    CNN7_Shi,
    CNN3_Mao,
    CNN5_Mao,
    CNN9_Mao,
    CNN11_Mao,
) 
from CTRAIN.model_wrappers import (
    CrownIBPModelWrapper,
    MTLIBPModelWrapper,
    SABRModelWrapper,
    ShiIBPModelWrapper,
)
from CTRAIN.util import seed_ctrain


PAPER_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_ROOT = os.environ.get("CTRAIN_DATA_ROOT", os.path.abspath(os.path.join(PAPER_ROOT, "..", "data")))
RESULTS_ROOT = os.environ.get("CTRAIN_PAPER_RESULTS_ROOT", os.path.join(PAPER_ROOT, "results"))


epochs_map = {
    ("cifar10", "cnn7", "mtl_ibp", 2 / 255): 160,
    ("cifar10", "cnn7", "sabr", 2 / 255): 160,
    ("cifar10", "cnn7", "shi", 2 / 255): 160,
    ("cifar10", "cnn7", "crown_ibp", 2 / 255): 160,
    ("cifar10", "cnn7", "crown_ibp_nofusion", 2 / 255): 160,
    ("cifar10", "wide_cnn7", "mtl_ibp", 2 / 255): 160,
    ("cifar10", "wide_cnn7", "sabr", 2 / 255): 160,
    ("cifar10", "wide_cnn7", "shi", 2 / 255): 160,
    ("cifar10", "wide_cnn7", "crown_ibp", 2 / 255): 160,
    ("cifar10", "wide_cnn7", "crown_ibp_nofusion", 2 / 255): 160,
    ("cifar10", "cnn3", "mtl_ibp", 2 / 255): 160,
    ("cifar10", "cnn3", "sabr", 2 / 255): 160,
    ("cifar10", "cnn3", "shi", 2 / 255): 160,
    ("cifar10", "cnn3", "crown_ibp", 2 / 255): 160,
    ("cifar10", "cnn3", "crown_ibp_nofusion", 2 / 255): 160,
    ("cifar10", "cnn5", "mtl_ibp", 2 / 255): 160,
    ("cifar10", "cnn5", "sabr", 2 / 255): 160,
    ("cifar10", "cnn5", "shi", 2 / 255): 160,
    ("cifar10", "cnn5", "crown_ibp", 2 / 255): 160,
    ("cifar10", "cnn5", "crown_ibp_nofusion", 2 / 255): 160,
    ("cifar10", "cnn9", "mtl_ibp", 2 / 255): 160,
    ("cifar10", "cnn9", "sabr", 2 / 255): 160,
    ("cifar10", "cnn9", "shi", 2 / 255): 160,
    ("cifar10", "cnn9", "crown_ibp", 2 / 255): 160,
    ("cifar10", "cnn9", "crown_ibp_nofusion", 2 / 255): 160,
    ("cifar10", "cnn11", "mtl_ibp", 2 / 255): 160,
    ("cifar10", "cnn11", "sabr", 2 / 255): 160,
    ("cifar10", "cnn11", "shi", 2 / 255): 160,
    ("cifar10", "cnn11", "crown_ibp", 2 / 255): 160,
    ("cifar10", "cnn11", "crown_ibp_nofusion", 2 / 255): 160,
    ("cifar10", "narrow_cnn7", "mtl_ibp", 2 / 255): 160,
    ("cifar10", "narrow_cnn7", "sabr", 2 / 255): 160,
    ("cifar10", "narrow_cnn7", "shi", 2 / 255): 160,
    ("cifar10", "narrow_cnn7", "crown_ibp", 2 / 255): 160,
    ("cifar10", "narrow_cnn7", "crown_ibp_nofusion", 2 / 255): 160,
    ("cifar10", "cnn7", "mtl_ibp", 8 / 255): 260,
    ("cifar10", "cnn7", "sabr", 8 / 255): 260,
    ("cifar10", "cnn7", "shi", 8 / 255): 260,
    ("cifar10", "cnn7", "crown_ibp", 8 / 255): 260,
    ("cifar10", "cnn7", "crown_ibp_nofusion", 8 / 255): 260,
    ("cifar10", "wide_cnn7", "mtl_ibp", 8 / 255): 260,
    ("cifar10", "wide_cnn7", "sabr", 8 / 255): 260,
    ("cifar10", "wide_cnn7", "shi", 8 / 255): 260,
    ("cifar10", "wide_cnn7", "crown_ibp", 8 / 255): 260,
    ("cifar10", "wide_cnn7", "crown_ibp_nofusion", 8 / 255): 260,
    ("cifar10", "resnet18", "mtl_ibp", 2 / 255): 160,
    ("cifar10", "resnet18", "sabr", 2 / 255): 160,
    ("cifar10", "resnet18", "shi", 2 / 255): 160,
    ("cifar10", "resnet18", "crown_ibp", 2 / 255): 160,
    ("cifar10", "resnet18", "crown_ibp_nofusion", 2 / 255): 160,
    ("cifar10", "resnet18", "mtl_ibp", 8 / 255): 260,
    ("cifar10", "resnet18", "sabr", 8 / 255): 260,
    ("cifar10", "resnet18", "shi", 8 / 255): 260,
    ("cifar10", "resnet18", "crown_ibp", 8 / 255): 260,
    ("cifar10", "resnet18", "crown_ibp_nofusion", 8 / 255): 260,
    ("mnist", "resnet18", "mtl_ibp", 0.3): 70,
    ("mnist", "resnet18", "sabr", 0.3): 70,
    ("mnist", "resnet18", "shi", 0.3): 70,
    ("mnist", "resnet18", "crown_ibp", 0.3): 70,
    ("mnist", "cnn7", "mtl_ibp", 0.3): 70,
    ("mnist", "cnn7", "sabr", 0.3): 70,
    ("mnist", "cnn7", "shi", 0.3): 70,
    ("mnist", "cnn7", "crown_ibp", 0.3): 70,
    ("mnist", "wide_cnn7", "mtl_ibp", 0.3): 70,
    ("mnist", "wide_cnn7", "sabr", 0.3): 70,
    ("mnist", "wide_cnn7", "shi", 0.3): 70,
    ("mnist", "wide_cnn7", "crown_ibp", 0.3): 70,
    ("gtsrb", "resnet18", "mtl_ibp", 2 / 255): 160,
    ("gtsrb", "resnet18", "sabr", 2 / 255): 160,
    ("gtsrb", "resnet18", "shi", 2 / 255): 160,
    ("gtsrb", "resnet18", "crown_ibp", 2 / 255): 160,
    ("gtsrb", "cnn7", "mtl_ibp", 2 / 255): 160,
    ("gtsrb", "cnn7", "sabr", 2 / 255): 160,
    ("gtsrb", "cnn7", "shi", 2 / 255): 160,
    ("gtsrb", "cnn7", "crown_ibp", 2 / 255): 160,
    ("gtsrb", "wide_cnn7", "mtl_ibp", 2 / 255): 160,
    ("gtsrb", "wide_cnn7", "sabr", 2 / 255): 160,
    ("gtsrb", "wide_cnn7", "shi", 2 / 255): 160,
    ("gtsrb", "wide_cnn7", "crown_ibp", 2 / 255): 160,
    ("tinyimagenet", "resnet18", "mtl_ibp", 0.3): 160,
    ("tinyimagenet", "resnet18", "sabr", 0.3): 160,
    ("tinyimagenet", "resnet18", "shi", 0.3): 160,
    ("tinyimagenet", "resnet18", "crown_ibp", 0.3): 160,
    ("tinyimagenet", "cnn7", "mtl_ibp", 0.3): 160,
    ("tinyimagenet", "cnn7", "sabr", 0.3): 160,
    ("tinyimagenet", "cnn7", "shi", 0.3): 160,
    ("tinyimagenet", "cnn7", "crown_ibp", 0.3): 160,
    ("tinyimagenet", "wide_cnn7", "mtl_ibp", 0.3): 160,
    ("tinyimagenet", "wide_cnn7", "sabr", 0.3): 160,
    ("tinyimagenet", "wide_cnn7", "shi", 0.3): 160,
    ("tinyimagenet", "wide_cnn7", "crown_ibp", 0.3): 160,
}

min_accs_map = {  # certified accuracy, natural accuracy
    ("cifar10", "cnn7", "mtl_ibp", 2 / 255): (0.4, 0.6),
    ("cifar10", "cnn7", "sabr", 2 / 255): (0.4, 0.6),
    ("cifar10", "cnn7", "shi", 2 / 255): (0.4, 0.6),
    ("cifar10", "cnn7", "crown_ibp", 2 / 255): (0.4, 0.6),
    ("cifar10", "cnn7", "crown_ibp_nofusion", 2 / 255): (0.4, 0.6),
    ("cifar10", "wide_cnn7", "mtl_ibp", 2 / 255): (0.4, 0.6),
    ("cifar10", "wide_cnn7", "sabr", 2 / 255): (0.4, 0.6),
    ("cifar10", "wide_cnn7", "shi", 2 / 255): (0.4, 0.6),
    ("cifar10", "wide_cnn7", "crown_ibp", 2 / 255): (0.4, 0.6),
    ("cifar10", "wide_cnn7", "crown_ibp_nofusion", 2 / 255): (0.4, 0.6),
    ("cifar10", "cnn3", "mtl_ibp", 2 / 255): (0.4, 0.6),
    ("cifar10", "cnn3", "sabr", 2 / 255): (0.4, 0.6),
    ("cifar10", "cnn3", "shi", 2 / 255): (0.4, 0.6),
    ("cifar10", "cnn3", "crown_ibp", 2 / 255): (0.4, 0.6),
    ("cifar10", "cnn3", "crown_ibp_nofusion", 2 / 255): (0.4, 0.6),
    ("cifar10", "cnn5", "mtl_ibp", 2 / 255): (0.4, 0.6),
    ("cifar10", "cnn5", "sabr", 2 / 255): (0.4, 0.6),
    ("cifar10", "cnn5", "shi", 2 / 255): (0.4, 0.6),
    ("cifar10", "cnn5", "crown_ibp", 2 / 255): (0.4, 0.6),
    ("cifar10", "cnn5", "crown_ibp_nofusion", 2 / 255): (0.4, 0.6),
    ("cifar10", "cnn9", "mtl_ibp", 2 / 255): (0.4, 0.6),
    ("cifar10", "cnn9", "sabr", 2 / 255): (0.4, 0.6),
    ("cifar10", "cnn9", "shi", 2 / 255): (0.4, 0.6),
    ("cifar10", "cnn9", "crown_ibp", 2 / 255): (0.4, 0.6),
    ("cifar10", "cnn9", "crown_ibp_nofusion", 2 / 255): (0.4, 0.6),
    ("cifar10", "cnn11", "mtl_ibp", 2 / 255): (0.4, 0.6),
    ("cifar10", "cnn11", "sabr", 2 / 255): (0.4, 0.6),
    ("cifar10", "cnn11", "shi", 2 / 255): (0.4, 0.6),
    ("cifar10", "cnn11", "crown_ibp", 2 / 255): (0.4, 0.6),
    ("cifar10", "cnn11", "crown_ibp_nofusion", 2 / 255): (0.4, 0.6),
    ("cifar10", "narrow_cnn7", "mtl_ibp", 2 / 255): (0.4, 0.6),
    ("cifar10", "narrow_cnn7", "sabr", 2 / 255): (0.4, 0.6),
    ("cifar10", "narrow_cnn7", "shi", 2 / 255): (0.4, 0.6),
    ("cifar10", "narrow_cnn7", "crown_ibp", 2 / 255): (0.4, 0.6),
    ("cifar10", "narrow_cnn7", "crown_ibp_nofusion", 2 / 255): (0.4, 0.6),
    ("cifar10", "cnn7", "mtl_ibp", 8 / 255): (0.25, 0.4),
    ("cifar10", "cnn7", "sabr", 8 / 255): (0.25, 0.4),
    ("cifar10", "cnn7", "shi", 8 / 255): (0.25, 0.4),
    ("cifar10", "cnn7", "crown_ibp", 8 / 255): (0.25, 0.4),
    ("cifar10", "cnn7", "crown_ibp_nofusion", 8 / 255): (0.25, 0.4),
    ("cifar10", "wide_cnn7", "mtl_ibp", 8 / 255): (0.25, 0.4),
    ("cifar10", "wide_cnn7", "sabr", 8 / 255): (0.25, 0.4),
    ("cifar10", "wide_cnn7", "shi", 8 / 255): (0.25, 0.4),
    ("cifar10", "wide_cnn7", "crown_ibp", 8 / 255): (0.25, 0.4),
    ("cifar10", "wide_cnn7", "crown_ibp_nofusion", 8 / 255): (0.25, 0.4),
    ("cifar10", "resnet18", "mtl_ibp", 2 / 255): (0.4, 0.6),
    ("cifar10", "resnet18", "sabr", 2 / 255): (0.4, 0.6),
    ("cifar10", "resnet18", "shi", 2 / 255): (0.4, 0.6),
    ("cifar10", "resnet18", "crown_ibp", 2 / 255): (0.4, 0.6),
    ("cifar10", "resnet18", "crown_ibp_nofusion", 2 / 255): (0.4, 0.6),
    ("cifar10", "resnet18", "mtl_ibp", 8 / 255): (0.25, 0.4),
    ("cifar10", "resnet18", "sabr", 8 / 255): (0.25, 0.4),
    ("cifar10", "resnet18", "shi", 8 / 255): (0.25, 0.4),
    ("cifar10", "resnet18", "crown_ibp", 8 / 255): (0.25, 0.4),
    ("cifar10", "resnet18", "crown_ibp_nofusion", 8 / 255): (0.25, 0.4),
    ("mnist", "resnet18", "mtl_ibp", 0.3): (0.9, 0.95),
    ("mnist", "resnet18", "sabr", 0.3): (0.9, 0.95),
    ("mnist", "resnet18", "shi", 0.3): (0.9, 0.95),
    ("mnist", "resnet18", "crown_ibp", 0.3): (0.9, 0.95),
    ("mnist", "cnn7", "mtl_ibp", 0.3): (0.9, 0.95),
    ("mnist", "cnn7", "sabr", 0.3): (0.9, 0.95),
    ("mnist", "cnn7", "shi", 0.3): (0.9, 0.95),
    ("mnist", "cnn7", "crown_ibp", 0.3): (0.9, 0.95),
    ("mnist", "wide_cnn7", "mtl_ibp", 0.3): (0.9, 0.95),
    ("mnist", "wide_cnn7", "sabr", 0.3): (0.9, 0.95),
    ("mnist", "wide_cnn7", "shi", 0.3): (0.9, 0.95),
    ("mnist", "wide_cnn7", "crown_ibp", 0.3): (0.9, 0.95),
    ("tinyimagenet", "resnet18", "mtl_ibp", 0.3): (0.15, 0.2),
    ("tinyimagenet", "resnet18", "sabr", 0.3): (0.15, 0.2),
    ("tinyimagenet", "resnet18", "shi", 0.3): (0.15, 0.2),
    ("tinyimagenet", "resnet18", "crown_ibp", 0.3): (0.15, 0.2),
    ("tinyimagenet", "cnn7", "mtl_ibp", 0.3): (0.15, 0.2),
    ("tinyimagenet", "cnn7", "sabr", 0.3): (0.15, 0.2),
    ("tinyimagenet", "cnn7", "shi", 0.3): (0.15, 0.2),
    ("tinyimagenet", "cnn7", "crown_ibp", 0.3): (0.15, 0.2),
    ("tinyimagenet", "wide_cnn7", "mtl_ibp", 0.3): (0.15, 0.2),
    ("tinyimagenet", "wide_cnn7", "sabr", 0.3): (0.15, 0.2),
    ("tinyimagenet", "wide_cnn7", "shi", 0.3): (0.15, 0.2),
    ("tinyimagenet", "wide_cnn7", "crown_ibp", 0.3): (0.15, 0.2),
}


def _mirror_mtl_ibp_entries(mapping):
    for (dataset, network, method, eps), value in list(mapping.items()):
        if method == "mtl_ibp":
            mapping[(dataset, network, "cc_ibp", eps)] = value
            mapping[(dataset, network, "exp_ibp", eps)] = value


_mirror_mtl_ibp_entries(epochs_map)
_mirror_mtl_ibp_entries(min_accs_map)


def run(method, dataset, epochs, network, eps, seed, min_cert_acc, min_nat_acc, complete_verify=False):
    print(
        f"Running {method} on {dataset} with {network} and eps={eps} and seed={seed} for {epochs} epochs with min cert acc {min_cert_acc} and min nat acc {min_nat_acc}"
    )
    batch_size = 512
    seed_ctrain(seed=seed)

    if dataset == "cifar10":
        train_loader, test_loader = load_cifar10(
            batch_size=batch_size,
            val_split=False,
            data_root=DATA_ROOT,
        )
        in_shape = [3, 32, 32]
        n_classes = 10
    elif dataset == "mnist":
        train_loader, test_loader = load_mnist(
            batch_size=batch_size,
            val_split=False,
            data_root=DATA_ROOT,
        )
        in_shape = [1, 28, 28]
        n_classes = 10
    elif dataset == "tinyimagenet":
        train_loader, test_loader = load_tinyimagenet(
            batch_size=batch_size,
            val_split=False,
            data_root=DATA_ROOT,
        )
        in_shape = [3, 64, 64]
        n_classes = 200

    if network == "cnn7":
        model = CNN7_Shi(in_shape=in_shape, width=64, n_classes=n_classes)
    elif network == "resnet18":
        model = resnet18(num_classes=n_classes)
        model.conv1 = torch.nn.Conv2d(
            in_shape[0], 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        model.maxpool = torch.nn.Identity()
    elif network == "wide_cnn7":
        model = CNN7_Shi(in_shape=in_shape, width=128, n_classes=n_classes)
    elif network == "narrow_cnn7":
        model = CNN7_Shi(in_shape=in_shape, width=32, n_classes=n_classes)
    elif network == "cnn3":
        model = CNN3_Mao(in_shape=in_shape, width=64, n_classes=n_classes)
    elif network == "cnn5":
        model = CNN5_Mao(in_shape=in_shape, width=64, n_classes=n_classes)
    elif network == "cnn9":
        model = CNN9_Mao(in_shape=in_shape, width=64, n_classes=n_classes)
    elif network == "cnn11":
        model = CNN11_Mao(in_shape=in_shape, width=64, n_classes=n_classes)
    else:
        raise ValueError(f"Unknown network: {network}")

    if method == "mtl_ibp":
        wrapped_model = MTLIBPModelWrapper(
            model=model,
            input_shape=in_shape,
            device=torch.device("cuda"),
            eps=eps,
            num_epochs=epochs,
        )
    elif method in {"cc_ibp", "exp_ibp"}:
        raise NotImplementedError(
            f"{method} was an archived publication-fork variant and is not part "
            "of the current CTRAIN codebase."
        )
    elif method == "sabr":
        wrapped_model = SABRModelWrapper(
            model=model,
            input_shape=in_shape,
            device=torch.device("cuda"),
            eps=eps,
            num_epochs=epochs,
        )
    elif method == "shi":
        wrapped_model = ShiIBPModelWrapper(
            model=model,
            input_shape=in_shape,
            device=torch.device("cuda"),
            eps=eps,
            num_epochs=epochs,
        )
    elif method == "crown_ibp":
        wrapped_model = CrownIBPModelWrapper(
            model=model,
            input_shape=in_shape,
            device=torch.device("cuda"),
            eps=eps,
            num_epochs=epochs,
        )
    elif method == "crown_ibp_nofusion":
        wrapped_model = CrownIBPModelWrapper(
            model=model,
            input_shape=in_shape,
            device=torch.device("cuda"),
            eps=eps,
            num_epochs=epochs,
            loss_fusion=False,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    wrapped_model.hpo(
        train_loader=train_loader,
        val_loader=test_loader,
        eval_samples=1_000_000,
        defaults={},
        output_dir=f"{RESULTS_ROOT}/hpo/{dataset}_{network}_{method}_{eps}_{seed}_complete_{complete_verify}",
        budget_time=np.inf,
        budget_trials=100,
        seed=seed,
        min_cert_acc=min_cert_acc,
        min_nat_acc=min_nat_acc,
        complete_verify=complete_verify,
    )


if __name__ == "__main__":
    executor = submitit.AutoExecutor(
        "logs", "slurm"
    )
    executor.update_parameters(
        timeout_min=60 * 24 * 50,
        slurm_partition="CLUSTER",
        gpus_per_node=1,
        slurm_array_parallelism=12,
        cpus_per_task=14,
        mem_gb=15.7 * 14,
        slurm_additional_parameters={"qos": "gpu"},
        slurm_job_name="MOCTRAIN",
    )

    with executor.batch():
        for network in [
            "cnn7",
            "wide_cnn7",
            "cnn5",
            "cnn9",
            "narrow_cnn7",
            "resnet18"
        ]:
            for method in ["crown_ibp_nofusion",  "sabr", "shi", "mtl_ibp"]:
                dataset = "cifar10"
                for eps in [ 2 / 255, 8 / 255]:
                    if eps == 8 / 255 and network != "cnn7":
                        continue
                    for seed in [0, 1, 2]:
                        epochs = epochs_map[(dataset, network, method, eps)]
                        min_cert_acc, min_nat_acc = min_accs_map[
                            (dataset, network, method, eps)
                        ]
                        
                        executor.submit(
                            run,
                            method=method,
                            dataset=dataset,
                            epochs=epochs,
                            network=network,
                            eps=eps,
                            seed=seed,
                            min_cert_acc=min_cert_acc,
                            min_nat_acc=min_nat_acc,
                        )


                dataset = "mnist"
                for eps in [ 0.3 ]:
                    if network not in ["cnn7"]:
                        continue
                    for seed in [0, 1, 2]:
                        epochs = epochs_map[(dataset, network, method, eps)]
                        min_cert_acc, min_nat_acc = min_accs_map[
                            (dataset, network, method, eps)
                        ]
                        
                        executor.submit(
                            run,
                            method=method,
                            dataset=dataset,
                            epochs=epochs,
                            network=network,
                            eps=eps,
                            seed=seed,
                            min_cert_acc=min_cert_acc,
                            min_nat_acc=min_nat_acc,
                        )

                dataset = "tinyimagenet"
                for eps in [ 1 / 255 ]:
                    if network not in ["cnn7"]:
                        continue
                    for seed in [0, 1, 2]:
                        epochs = epochs_map[(dataset, network, method, eps)]
                        min_cert_acc, min_nat_acc = min_accs_map[
                            (dataset, network, method, eps)
                        ]
                        
                        if method == "crown_ibp_nofusion":
                            method = "crown_ibp" 
                        executor.submit(
                            run,
                            method=method,
                            dataset=dataset,
                            epochs=epochs,
                            network=network,
                            eps=eps,
                            seed=seed,
                            min_cert_acc=min_cert_acc,
                            min_nat_acc=min_nat_acc,
                        )
                        if method == "crown_ibp":
                            method = "crown_ibp_nofusion" 

    network = "cnn7"
    method = "cc_ibp"
    dataset = "cifar10"
    eps = 2 / 255
    for seed in [0, 1, 2]:
        epochs = epochs_map[(dataset, network, method, eps)]
        min_cert_acc, min_nat_acc = min_accs_map[
            (dataset, network, method, eps)
        ]
        
        executor.submit(
            run,
            method=method,
            dataset=dataset,
            epochs=epochs,
            network=network,
            eps=eps,
            seed=seed,
            min_cert_acc=min_cert_acc,
            min_nat_acc=min_nat_acc,
            complete_verify=False,
        )

    network = "cnn7"
    method = "exp_ibp"
    dataset = "cifar10"
    eps = 2 / 255
    for seed in [0, 1, 2]:
        epochs = epochs_map[(dataset, network, method, eps)]
        min_cert_acc, min_nat_acc = min_accs_map[
            (dataset, network, method, eps)
        ]
        
        executor.submit(
            run,
            method=method,
            dataset=dataset,
            epochs=epochs,
            network=network,
            eps=eps,
            seed=seed,
            min_cert_acc=min_cert_acc,
            min_nat_acc=min_nat_acc,
            complete_verify=False,
        )

    network = "cnn7"
    method = "mtl_ibp"
    dataset = "cifar10"
    eps = 2 / 255
    for seed in [0, 1, 2]:
        epochs = epochs_map[(dataset, network, method, eps)]
        min_cert_acc, min_nat_acc = min_accs_map[
            (dataset, network, method, eps)
        ]
        
        executor.submit(
            run,
            method=method,
            dataset=dataset,
            epochs=epochs,
            network=network,
            eps=eps,
            seed=seed,
            min_cert_acc=min_cert_acc,
            min_nat_acc=min_nat_acc,
            complete_verify=True,
        )
