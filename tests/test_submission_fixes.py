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


def schedules():
    for scheduler_type in (LinearScheduler, SmoothedScheduler):
        for ramp in (0, 1, 2, 4, 10):
            for eps in (0., .2):
                args = dict(num_epochs=12, eps=eps, mean=0., std=1., start_kappa=.8,
                            end_kappa=.2, start_beta=.7, end_beta=.3, eps_schedule=(1, ramp),
                            eps_schedule_unit='epoch', batches_per_epoch=2)
                full = scheduler_type(**args)
                values = []
                for step in range(24):
                    values.append((full.get_cur_eps(False), full.get_cur_kappa(), full.get_cur_beta()))
                    if step % 2 == 0 and step:
                        resumed = scheduler_type(**args, start_epoch=step // 2)
                        torch.testing.assert_close(torch.tensor(values[-1]), torch.tensor([
                            resumed.get_cur_eps(False), resumed.get_cur_kappa(), resumed.get_cur_beta()]))
                    full.batch_step()
                assert all(np.isfinite(value).all() for value in values)
                torch.testing.assert_close(torch.tensor(values[-1]), torch.tensor([eps, .2, .3]))
                assert values[0] == (0., .8, .7)
        immediate = scheduler_type(num_epochs=1, eps=.2, mean=0., std=1.,
                                   eps_schedule=(0, 0), batches_per_epoch=1)
        assert immediate.get_cur_eps(False) == .2
        offset = scheduler_type(num_epochs=2, eps=.2, start_eps=.1, mean=0., std=1.,
                                eps_schedule=(0, 2), batches_per_epoch=2)
        for _ in range(5):
            offset.batch_step()
        assert offset.get_cur_eps(False) == .2


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




def validation_transforms():
    class Dataset:
        def __init__(self, transform, **kwargs):
            self.transform = transform
            self._samples = [(f'00000/{i // 2:05d}_{i:05d}.ppm', 0) for i in range(20)]

        def __len__(self):
            return 20

        def __getitem__(self, index):
            return self.transform(Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))), 0

    def augment(image):
        return Image.fromarray(np.full((32, 32, 3), 255, dtype=np.uint8))
    for name, load in [('MNIST', load_mnist), ('CIFAR10', load_cifar10),
                       ('GTSRB', load_gtsrb), ('ImageFolder', load_tinyimagenet)]:
        with patch(f'CTRAIN.data_loaders.data_loaders.datasets.{name}', Dataset), \
             patch('CTRAIN.data_loaders.data_loaders.download_tinyimagenet'):
            train, val, test = load(normalise=False, train_transforms=[augment], val_split=True)
        assert train.dataset[0][0].mean() == 1
        assert val.dataset[0][0].mean() == test.dataset[0][0].mean() == 0
        assert not set(train.dataset.indices) & set(val.dataset.indices)
        assert train.dataset.dataset.transform is not val.dataset.dataset.transform


if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.manual_seed(0)
    attack_counterexamples()
    schedules()
    validation_transforms()
    print("Shared training and verification regressions passed")
