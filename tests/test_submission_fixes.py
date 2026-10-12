"""CPU regressions for submission audit findings 1–15. Run this file directly."""
import copy
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import optuna
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from auto_LiRPA import PerturbationLpNorm

from CTRAIN.attacks import pgd_attack
from CTRAIN.data_loaders import load_mnist, load_cifar10, load_gtsrb, load_tinyimagenet
from CTRAIN.eval.eval import eval_complete, eval_crown
from CTRAIN.model_definitions import CNN7_Shi, GowalConvMed, GowalConvLarge
from CTRAIN.model_wrappers import (ShiIBPModelWrapper, CrownIBPModelWrapper,
                                   TAPSModelWrapper, STAPSModelWrapper, GLSModelWrapper,
                                   SABRModelWrapper, MTLIBPModelWrapper)
from CTRAIN.model_wrappers.configs import get_config_space
from CTRAIN.train.certified.eps_scheduler import LinearScheduler, SmoothedScheduler
from CTRAIN.train.certified.losses.crown_ibp import get_crown_ibp_loss


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Flatten(), nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))

    def forward(self, inputs):
        return self.layers(inputs)


def loader_for(data=None):
    data = torch.tensor([.2, .4, .6, .8]).reshape(4, 1, 1, 1).expand(4, 1, 2, 2).clone() if data is None else data
    loader = DataLoader(TensorDataset(data, torch.zeros(len(data), dtype=torch.long)), batch_size=2)
    loader.mean, loader.std = torch.zeros(1), torch.ones(1)
    loader.min, loader.max, loader.normalised = torch.tensor(0.), torch.tensor(1.), False
    return loader


def raises(action, message):
    try:
        action()
    except ValueError as exc:
        assert message in str(exc), str(exc)
    else:
        raise AssertionError(f'Expected ValueError containing {message!r}')




def attack_counterexamples():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(1.))

        def forward(self, x):
            z = x * self.scale
            return torch.cat((z * 0, .11375 - .1375 * z, -11.24875 + 12.4875 * z), 1)

    model = Model()
    model.scale.grad = torch.tensor(7.)
    data, target = torch.tensor([[.85]]), torch.tensor([0])
    candidates = iter([torch.tensor([[.1]]), torch.tensor([[.9]])])
    with patch('torch.rand', side_effect=lambda *a, **kw: next(candidates)):
        result = pgd_attack(model, data, target, torch.zeros_like(data), torch.ones_like(data),
                            restarts=2, n_steps=1, step_size=1e-6, early_stopping=False, device='cpu')
    assert model(result).argmax(1).item() == 1
    assert model.scale.grad.item() == 7.
    # An already wrong nominal input survives a zero-step random restart.
    wrong = torch.tensor([[.1]])
    with patch('torch.rand', return_value=torch.tensor([[.9]])):
        result = pgd_attack(model, wrong, target, torch.zeros_like(data), torch.ones_like(data),
                            n_steps=0, early_stopping=False, device='cpu')
    assert model(result).argmax(1).item() == 1
    raises(lambda: pgd_attack(model, data, target, data, data, restarts=0, device='cpu'), 'restarts')


if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.manual_seed(0)
    attack_counterexamples()
    print("Shared training and verification regressions passed")
