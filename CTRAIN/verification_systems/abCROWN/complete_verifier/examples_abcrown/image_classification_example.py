#########################################################################
##   This file is part of the α,β-CROWN (alpha-beta-CROWN) verifier    ##
##                                                                     ##
##   Copyright (C) 2021-2025 The α,β-CROWN Team                        ##
##   Team leaders:                                                     ##
##          Faculty:   Huan Zhang <huan@huan-zhang.com> (UIUC)         ##
##          Student:   Xiangru Zhong <xiangru4@illinois.edu> (UIUC)    ##
##                                                                     ##
##   See CONTRIBUTORS for all current and past developers in the team. ##
##                                                                     ##
##     This program is licensed under the BSD 3-Clause License,        ##
##        contained in the LICENCE file in this directory.             ##
##                                                                     ##
#########################################################################
"""Minimal image-classification verification demo."""

import torch

from abcrown import (
    ABCrownSolver,
    ConfigBuilder,
    VerificationSpec,
    input_vars,
    output_vars,
)


class SimpleConvClassifier(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 16, kernel_size=3, padding=1),
            torch.nn.ReLU(),
        )
        self.head = torch.nn.Linear(16 * 32 * 32, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.view(x.shape[0], 3, 32, 32)
        feats = self.conv(x)
        flat = feats.view(feats.shape[0], -1)
        return self.head(flat)


def main() -> None:
    torch.manual_seed(40)
    base_image = torch.rand(1, 3, 32, 32)
    eps = 0.0039

    x = input_vars((3, 32, 32))
    y = output_vars(10)
    input_constraint = (x >= (base_image - eps)) & (x <= (base_image + eps))
    label = 0
    output_constraint = None
    for i in range(10):
        if i == label:
            continue
        pred = y[i] > y[label]
        output_constraint = pred if output_constraint is None else (output_constraint & pred)
    if output_constraint is None:
        raise ValueError("No output constraints generated.")
    spec = VerificationSpec.build_spec(
        input_vars=x,
        output_vars=y,
        input_constraint=input_constraint,
        output_constraint=output_constraint,
    )

    model = SimpleConvClassifier()
    cfg = ConfigBuilder.from_defaults()
    solver = ABCrownSolver(spec, model, config=cfg)
    result = solver.solve()

    print(f"[info] verifying epsilon={eps:.4f} around a random base image")
    print(f"status={result.status}, success={result.success}")


if __name__ == "__main__":
    main()
