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


def checkpoint_and_modes():
    loader = loader_for()
    for wrapper_type, extra in [(ShiIBPModelWrapper, {}),
                                (CrownIBPModelWrapper, {'loss_fusion': True})]:
        with tempfile.TemporaryDirectory() as root:
            opts = {'conv_mode': 'patches', 'relu': 'adaptive'}
            args = dict(input_shape=(1, 2, 2), eps=.1, num_epochs=3, device='cpu',
                        warm_up_epochs=0, ramp_up_epochs=2, l1_reg_weight=0, shi_reg_weight=0,
                        optimizer_func=torch.optim.SGD, lr=.1,
                        lr_decay_kwargs={'milestones': (1,), 'gamma': .1}, bound_opts=opts, **extra)
            torch.manual_seed(12)
            full = wrapper_type(Toy(), checkpoint_save_path=root, checkpoint_save_interval=1, **args)
            assert full.eval() is full and full.train(False) is full
            assert all(not module.training for module in full.modules())
            assert full.train() is full and all(module.training for module in full.modules())
            assert full.lr_scheduler.optimizer is full.optimizer
            assert 'loss_fusion' not in opts and 'loss_fusion' not in full.bound_opts
            full.train_model(loader)
            checkpoint = torch.load(Path(root) / '1_checkpoint.pt', weights_only=False)
            assert checkpoint['lr_scheduler_state_dict']['last_epoch'] == 1
            assert abs(checkpoint['optimizer_state_dict']['param_groups'][0]['lr'] - .01) < 1e-9
            resumed = wrapper_type(Toy(), **args)
            resumed.resume_from_checkpoint(Path(root) / '1_checkpoint.pt', loader)
            assert resumed.lr_scheduler.state_dict() == full.lr_scheduler.state_dict()
            for actual, expected in zip(resumed.parameters(), full.parameters()):
                torch.testing.assert_close(actual, expected)
            del checkpoint['lr_scheduler_state_dict']
            legacy = Path(root) / 'legacy.pt'
            torch.save(checkpoint, legacy)
            raises(lambda: resumed.resume_from_checkpoint(legacy, loader), 'legacy checkpoint')

    for wrapper_type in (SABRModelWrapper, MTLIBPModelWrapper, TAPSModelWrapper, STAPSModelWrapper):
        extra = {'block_sizes': (2, 2)} if wrapper_type in (TAPSModelWrapper, STAPSModelWrapper) else {}
        args = dict(input_shape=(1, 2, 2), eps=.1, num_epochs=3, device='cpu',
                    warm_up_epochs=0, ramp_up_epochs=2, l1_reg_weight=0, shi_reg_weight=0,
                    optimizer_func=torch.optim.SGD, lr=.1, pgd_steps=1,
                    lr_decay_kwargs={'milestones': (1,), 'gamma': .1}, **extra)
        with tempfile.TemporaryDirectory() as root:
            torch.manual_seed(12)
            full = wrapper_type(Toy(), checkpoint_save_path=root, checkpoint_save_interval=1, **args)
            full.train_model(loader, end_epoch=1)
            continuation_rng = torch.get_rng_state()
            full.train_model(loader, start_epoch=1)

            resumed = wrapper_type(Toy(), **args)
            # Check the trajectory with identical attack randomness supplied by the test.
            torch.set_rng_state(continuation_rng)
            resumed.resume_from_checkpoint(Path(root) / '1_checkpoint.pt', loader)
            assert resumed.lr_scheduler.state_dict() == full.lr_scheduler.state_dict()
            for actual, expected in zip(resumed.parameters(), full.parameters()):
                torch.testing.assert_close(actual, expected)


def crown_bounds_and_models():
    loader = loader_for()
    data, targets = next(iter(loader))
    for fusion in (False, True):
        wrapper = CrownIBPModelWrapper(Toy(), (1, 2, 2), .1, 2, device='cpu',
                                      warm_up_epochs=0, ramp_up_epochs=1, loss_fusion=fusion)
        for beta in (0., .5, 1.):
            if fusion:
                wrapper.loss_fusion_model(data, targets)
            else:
                wrapper.bounded_model(data)
            ptb = PerturbationLpNorm(norm=np.inf, eps=.1, x_L=data-.1, x_U=data+.1)
            loss, lower, upper = get_crown_ibp_loss(
                wrapper.loss_fusion_model if fusion else wrapper.bounded_model,
                ptb, data, targets, 2, nn.CrossEntropyLoss(), beta,
                loss_fusion=fusion, return_bounds=True, return_stats=False)
            assert torch.isfinite(loss)
            assert lower is not None
            if fusion:
                assert upper is not None and torch.isfinite(upper).all()
            gradients = torch.autograd.grad(loss, tuple(wrapper.parameters()), allow_unused=True)
            assert any(g is not None and torch.isfinite(g).all() for g in gradients)
    for model_type in (GowalConvMed, GowalConvLarge):
        assert model_type(n_classes=7)(torch.zeros(2, 1, 28, 28)).shape == (2, 7)
    for size in (28, 29, 55, 57):
        for downsample in (False, True):
            model = CNN7_Shi((1, size, size), width=2, linear_size=4, n_classes=3,
                             downsample_final=downsample).eval()
            assert model(torch.zeros(2, 1, size, size)).shape == (2, 3)


def hpo_and_gls():
    from optuna_integration.botorch import BoTorchSampler
    loader = loader_for()
    for wrapper_type in (ShiIBPModelWrapper, CrownIBPModelWrapper):
        wrapper = wrapper_type(Toy(), (1, 2, 2), .1, 1, device='cpu', warm_up_epochs=0,
                               ramp_up_epochs=1, start_kappa=.7, end_kappa=.2)
        sampler = wrapper._get_optuna_single_objective_sampler(optuna, 'botorch', 0)
        assert isinstance(sampler, BoTorchSampler) and sampler._device == torch.device('cpu')
        with tempfile.TemporaryDirectory() as root:
            # This runs a real bounded training epoch and real incomplete evaluation.
            result = wrapper.hpo_single_objective(loader, loader, budget_trials=1,
                                                  eval_samples=2, output_dir=root, load_best=False)
            assert np.isfinite(result['objective_value'])
            assert Path(result['checkpoint_path']).is_file()

    # Exercise a GP-guided candidate, beyond BoTorch's initial random trials.
    study = optuna.create_study(sampler=BoTorchSampler(n_startup_trials=2, seed=0,
                                                       device=torch.device('cpu')))
    study.optimize(lambda trial: (trial.suggest_float('x', -1., 1.) - .3) ** 2, n_trials=3)
    assert len(study.trials) == 3 and all(np.isfinite(trial.value) for trial in study.trials)

    for wrapper_type in (TAPSModelWrapper, STAPSModelWrapper):
        wrapper = wrapper_type(Toy(), (1, 2, 2), .1, 1, device='cpu', warm_up_epochs=0,
                               ramp_up_epochs=1, block_sizes=(2, 2))
        config = dict(get_config_space(wrapper, 1, torch.tensor(.1)).get_default_configuration())
        config['taps:block_split_point'] = .99
        config['l1_reg_weight'], config['shi_reg_weight'] = 1e-8, 0.
        with tempfile.TemporaryDirectory() as root:
            (Path(root) / 'nets').mkdir()
            loss, _ = wrapper._hpo_runner(config, 0, 1, loader, loader, root, cert_eval_samples=2)
            assert np.isfinite(loss)

    # With fixed zero noise, the GLS L1 term must affect the actual weight update.
    original = Toy()
    models = []
    for weight in (0., .2):
        wrapper = GLSModelWrapper(copy.deepcopy(original), (1, 2, 2), 0., 1, device='cpu',
                                  warm_up_epochs=0, ramp_up_epochs=0, shi_init=False,
                                  l1_reg_weight=weight, optimizer_func=torch.optim.SGD, lr=.1)
        with patch('torch.randn_like', side_effect=torch.zeros_like):
            wrapper.train_model(loader)
        models.append([p.detach().clone() for p in wrapper.parameters()])
    assert any(not torch.equal(a, b) for a, b in zip(*models))


if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.manual_seed(0)
    attack_counterexamples()
    schedules()
    validation_transforms()
    crown_bounds_and_models()
    checkpoint_and_modes()
    hpo_and_gls()
    print("Shared training and verification regressions passed")
