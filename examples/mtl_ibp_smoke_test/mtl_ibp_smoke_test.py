import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from CTRAIN.model_definitions.models_shi import CNN7_Shi
from CTRAIN.model_wrappers import MTLIBPModelWrapper
from CTRAIN.util import seed_ctrain


def make_synthetic_loader(batch_size=8, n_samples=32, input_shape=(1, 8, 8)):
    x = torch.rand(n_samples, *input_shape)
    # Simple deterministic label rule, so the smoke test is repeatable.
    y = (x.flatten(1).mean(dim=1) > 0.5).long()

    loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)

    mean = torch.tensor([0.0])
    std = torch.tensor([1.0])
    data_min = torch.tensor([0.0]).view(1, 1, 1, 1)
    data_max = torch.tensor([1.0]).view(1, 1, 1, 1)
    loader.mean = mean
    loader.std = std
    loader.min = data_min
    loader.max = data_max
    loader.normalised = False
    return loader


def main():
    parser = argparse.ArgumentParser(description="Small MTL-IBP training smoke test.")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--eps", type=float, default=0.03)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    ramp_steps = args.epochs * ((args.samples + args.batch_size - 1) // args.batch_size)
    if ramp_steps < 8:
        raise ValueError(
            "Use at least 8 total ramp-up batches for this smoke test, e.g. "
            "--epochs 2 --samples 32 --batch-size 8."
        )

    seed_ctrain(0)

    input_shape = (1, 8, 8)
    train_loader = make_synthetic_loader(
        batch_size=args.batch_size,
        n_samples=args.samples,
        input_shape=input_shape,
    )

    model = CNN7_Shi(input_shape, width=2, linear_size=8, n_classes=2)
    wrapper = MTLIBPModelWrapper(
        model=model,
        input_shape=input_shape,
        eps=args.eps,
        num_epochs=args.epochs,
        warm_up_epochs=0,
        ramp_up_epochs=args.epochs,
        lr=1e-3,
        lr_scheduler_func=torch.optim.lr_scheduler.MultiStepLR,
        lr_decay_kwargs={"milestones": [], "gamma": 1.0},
        gradient_clip=10,
        l1_reg_weight=0.0,
        shi_reg_weight=0.0,
        pgd_steps=1,
        pgd_alpha=args.eps,
        pgd_restarts=1,
        pgd_early_stopping=False,
        mtl_ibp_alpha=0.5,
        checkpoint_save_path=None,
        device=torch.device(args.device),
    )

    wrapper.train_model(train_loader=train_loader)

    wrapper.eval()
    with torch.no_grad():
        data, target = next(iter(train_loader))
        logits = wrapper(data.to(wrapper.device))
        pred = logits.argmax(dim=1).cpu()

    print("MTL-IBP smoke test completed.")
    print(f"Device: {wrapper.device}")
    print(f"Example batch accuracy: {(pred == target).float().mean().item():.3f}")


if __name__ == "__main__":
    main()
