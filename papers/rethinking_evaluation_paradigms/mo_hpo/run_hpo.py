import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from torchvision.models import resnet18

from CTRAIN.data_loaders import load_cifar10, load_gtsrb, load_mnist, load_tinyimagenet
from CTRAIN.model_definitions import CNN3_Mao, CNN5_Mao, CNN7_Shi, CNN9_Mao, CNN11_Mao
from CTRAIN.util import seed_ctrain


def build_loaders(dataset, network, batch_size, data_root, val_split=False):
    if dataset == "cifar10":
        return load_cifar10(batch_size=batch_size, val_split=val_split, data_root=data_root), [3, 32, 32], 10
    if dataset == "mnist":
        return load_mnist(batch_size=batch_size, val_split=val_split, data_root=data_root), [1, 28, 28], 10
    if dataset == "gtsrb":
        return load_gtsrb(batch_size=batch_size, val_split=val_split, data_root=data_root), [3, 30, 30], 43
    if dataset == "tinyimagenet":
        return load_tinyimagenet(batch_size=batch_size, val_split=val_split, data_root=data_root), [3, 64, 64], 200
    raise ValueError(f"Unknown dataset: {dataset}")


def build_model(network, input_shape, n_classes):
    if network == "cnn7":
        return CNN7_Shi(in_shape=input_shape, width=64, n_classes=n_classes)
    if network == "cnn3":
        return CNN3_Mao(in_shape=input_shape, width=64, n_classes=n_classes)
    if network == "cnn5":
        return CNN5_Mao(in_shape=input_shape, width=64, n_classes=n_classes)
    if network == "cnn9":
        return CNN9_Mao(in_shape=input_shape, width=64, n_classes=n_classes)
    if network == "cnn11":
        return CNN11_Mao(in_shape=input_shape, width=64, n_classes=n_classes)
    if network == "wide_cnn7":
        return CNN7_Shi(in_shape=input_shape, width=128, n_classes=n_classes)
    if network == "narrow_cnn7":
        return CNN7_Shi(in_shape=input_shape, width=32, n_classes=n_classes)
    if network == "resnet18":
        model = resnet18(num_classes=n_classes)
        model.conv1 = torch.nn.Conv2d(input_shape[0], 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = torch.nn.Identity()
        return model
    raise ValueError(f"Unknown network: {network}")


def build_wrapper(method, model, input_shape, eps, epochs, device, *, teacher=None,
                  for_evaluation=False):
    from CTRAIN.model_wrappers import (
        CrownIBPModelWrapper,
        MTLIBPModelWrapper,
        SABRModelWrapper,
        ShiIBPModelWrapper,
        CCIBPModelWrapper,
        CCDistModelWrapper,
    )

    kwargs = {
        "model": model,
        "input_shape": input_shape,
        "eps": eps,
        "num_epochs": epochs,
        "device": device,
    }
    if method == "cc_dist":
        if for_evaluation:
            return CCIBPModelWrapper(**kwargs, warm_up_epochs=0, ramp_up_epochs=0)
        if teacher is None:
            raise ValueError("CC-Dist HPO requires pretrained teacher weights")
        return CCDistModelWrapper(**kwargs, teacher=teacher,
                                 warm_up_epochs=min(1, epochs - 1),
                                 ramp_up_epochs=min(70, epochs - 1))
    if method == "mtl_ibp":
        return MTLIBPModelWrapper(**kwargs)
    if method == "sabr":
        return SABRModelWrapper(**kwargs)
    if method == "shi":
        return ShiIBPModelWrapper(**kwargs)
    if method == "crown_ibp":
        return CrownIBPModelWrapper(**kwargs)
    if method == "crown_ibp_nofusion":
        return CrownIBPModelWrapper(**kwargs, loss_fusion=False)
    raise ValueError(f"Unknown method: {method}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run one CTRAIN multi-objective HPO experiment.")
    parser.add_argument("--dataset", required=True, choices=["cifar10", "mnist", "gtsrb", "tinyimagenet"])
    parser.add_argument("--network", required=True, choices=["cnn3", "cnn5", "cnn7", "cnn9", "cnn11", "wide_cnn7", "narrow_cnn7", "resnet18"])
    parser.add_argument("--method", required=True, choices=["mtl_ibp", "sabr", "shi", "crown_ibp", "crown_ibp_nofusion", "cc_dist"])
    parser.add_argument("--teacher", type=Path, help="Native pretrained teacher checkpoint for CC-Dist.")
    parser.add_argument("--teacher-sha256", help="Expected SHA256 of the teacher checkpoint.")
    parser.add_argument("--eps", required=True, type=float)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--budget-time", type=float, default=np.inf)
    parser.add_argument("--budget-trials", type=float, default=100)
    parser.add_argument("--eval-samples", type=int, default=1_000_000)
    parser.add_argument("--min-cert-acc", type=float, default=0.0)
    parser.add_argument("--min-nat-acc", type=float, default=0.0)
    parser.add_argument("--sampler", default="botorch", choices=["botorch", "nsgaii"])
    parser.add_argument("--complete-verify", action="store_true")
    parser.add_argument("--val-split", action="store_true", help="Use a train/validation split for HPO instead of the test loader.")
    parser.add_argument("--data-root", default="data")
    parser.add_argument(
        "--output-root",
        default="papers/rethinking_evaluation_paradigms/results/hpo/main/optuna_studies",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    if args.method == "cc_dist" and args.teacher is None:
        parser.error("--method cc_dist requires --teacher")
    if args.method != "cc_dist" and (args.teacher or args.teacher_sha256):
        parser.error("Teacher options are only supported for CC-Dist")
    teacher_metadata = None
    if args.teacher is not None:
        args.teacher = args.teacher.resolve()
        digest = hashlib.sha256(args.teacher.read_bytes()).hexdigest()
        if args.teacher_sha256 and digest != args.teacher_sha256:
            raise ValueError("Teacher checkpoint SHA256 mismatch")
        teacher_metadata = dict(path=str(args.teacher), sha256=digest)

    seed_ctrain(seed=args.seed)
    loaders, input_shape, n_classes = build_loaders(
        args.dataset, args.network, args.batch_size, args.data_root, val_split=args.val_split
    )
    if args.val_split:
        train_loader, hpo_loader, _ = loaders
    else:
        train_loader, hpo_loader = loaders
    model = build_model(args.network, input_shape, n_classes)
    teacher = None
    if args.teacher is not None:
        teacher = copy.deepcopy(model)
        saved = torch.load(args.teacher, map_location="cpu", weights_only=True)
        teacher.load_state_dict(saved.get("model_state_dict", saved.get("model", saved)))
    wrapper = build_wrapper(
        args.method,
        model=model,
        input_shape=input_shape,
        eps=args.eps,
        epochs=args.epochs,
        device=torch.device(args.device),
        teacher=teacher,
    )

    output_dir = Path(args.output_root) / (
        f"{args.dataset}_{args.network}_{args.method}_{args.eps}_{args.seed}"
    )
    if teacher_metadata is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest = dict(arguments={key: str(value) if isinstance(value, Path) else value
                                   for key, value in vars(args).items()},
                        teacher=teacher_metadata,
                        selection_split="training_holdout" if args.val_split else "test",
                        objectives=["natural_accuracy", "incomplete_certified_accuracy"
                                    if not args.complete_verify else "complete_certified_accuracy"])
        manifest_path = output_dir / "run_manifest.json"
        if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Existing HPO manifest differs from this run")
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    wrapper.hpo(
        train_loader=train_loader,
        val_loader=hpo_loader,
        eval_samples=args.eval_samples,
        defaults={},
        output_dir=str(output_dir),
        budget_time=args.budget_time,
        budget_trials=args.budget_trials,
        seed=args.seed,
        min_cert_acc=args.min_cert_acc,
        min_nat_acc=args.min_nat_acc,
        sampler=args.sampler,
        complete_verify=args.complete_verify,
    )


if __name__ == "__main__":
    main()
