"""CPU checks of public HPO, evaluation, save/load and trajectory resumption."""
import copy
from pathlib import Path
import tempfile
from unittest.mock import patch

import optuna
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from CTRAIN import IBPRModelWrapper
from CTRAIN.model_wrappers.configs import get_config_space


def check():
    torch.set_num_threads(1)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    data = torch.tensor([.1, .3, .7, .9]).reshape(4, 1, 1, 1)
    loader = DataLoader(TensorDataset(data, torch.zeros(4, dtype=torch.long)), batch_size=2)
    loader.mean, loader.std = torch.tensor([0.]), torch.tensor([1.])
    loader.min, loader.max, loader.normalised = torch.tensor(0.), torch.tensor(1.), False
    model = nn.Sequential(nn.Flatten(), nn.Linear(1, 3), nn.ReLU(), nn.Linear(3, 2))
    with torch.no_grad():
        model[-1].weight.zero_()
        model[-1].bias.copy_(torch.tensor([2., -2.]))
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for scheduler, kwargs in ((torch.optim.lr_scheduler.ExponentialLR, {'gamma': .8}),
                                  (torch.optim.lr_scheduler.MultiStepLR, {'milestones': [1, 2], 'gamma': .5})):
            options = dict(num_epochs=3, ramp_up_epochs=1, pgd_steps=1, device='cpu',
                           lr_scheduler_func=scheduler, lr_decay_kwargs=kwargs,
                           checkpoint_save_interval=1)
            full = IBPRModelWrapper(copy.deepcopy(model), (1, 1, 1), .05, **options)
            run = root / scheduler.__name__
            split = IBPRModelWrapper(copy.deepcopy(model), (1, 1, 1), .05,
                                    checkpoint_save_path=str(run), **options)
            torch.manual_seed(43)
            full.train_model(loader)
            torch.manual_seed(43)
            split.train_model(loader, end_epoch=1)
            rng = torch.get_rng_state()
            resumed = IBPRModelWrapper(copy.deepcopy(model), (1, 1, 1), .05, **options)
            torch.set_rng_state(rng)
            resumed.resume_from_checkpoint(run / '1_checkpoint.pt', loader)
            assert resumed.epoch == full.epoch == 3
            assert resumed.lr_scheduler.state_dict() == full.lr_scheduler.state_dict()
            assert resumed.optimizer.param_groups[0]['lr'] == full.optimizer.param_groups[0]['lr']
            for name, value in full.state_dict().items():
                torch.testing.assert_close(resumed.state_dict()[name], value, rtol=0, atol=0)
        wrapper = IBPRModelWrapper(copy.deepcopy(model), (1, 1, 1), .05,
                                  num_epochs=3, ramp_up_epochs=1, device='cpu')
        # Budgets down to one epoch yield schedules that fit in every sample.
        for epochs in (1, 2, 3, 10, 20, 100):
            space = get_config_space(wrapper, epochs, .05)
            for config in space.sample_configuration(10):
                assert config['warm_up_epochs'] + config['ramp_up_epochs'] <= epochs
        assert get_config_space(wrapper, 3, .05).get_default_configuration()['optimizer_func'] == 'sgd'
        # Exercise actual training, adaptive certification, PGD, SQLite and artifacts.
        best = wrapper.hpo_single_objective(
            loader, loader, budget_trials=1, budget_time=60, eval_samples=2,
            sampler=optuna.samplers.RandomSampler(seed=0), output_dir=str(root / 'single'))
        assert Path(best['checkpoint_path']).is_file()
        assert all(0 <= best['metrics'][key] <= 1 for key in ('nat_acc', 'cert_acc', 'adv_acc'))
        loaded = torch.load(best['checkpoint_path'], map_location='cpu')
        for name, value in wrapper.state_dict().items():
            torch.testing.assert_close(value, loaded[name])
        front = wrapper.hpo(loader, loader, budget_trials=1, budget_time=60,
                            eval_samples=2, sampler='nsgaii', output_dir=str(root / 'multi'))
        assert len(front) == 1 and Path(front[0]['checkpoint_path']).is_file()
        with patch.object(wrapper, '_hpo_runner', side_effect=AssertionError('repeated trial')):
            assert len(wrapper.hpo(loader, loader, budget_trials=1, eval_samples=2,
                                   sampler='nsgaii', output_dir=str(root / 'multi'))) == 1
        incumbent = wrapper.hpo_smac(loader, loader, budget=10, eval_samples=2,
                                    deterministic=True, seed=0, output_dir=str(root / 'smac'))
        assert 'ibp_r:rho' in incumbent
        assert list((root / 'smac' / 'nets').glob('*.pt'))
        wrapper.eval()
        for method in ('IBP', 'ZONOTOPE'):
            metrics = wrapper.evaluate(loader, test_samples=2, eval_method=method)
            assert len(metrics) == 3 and all(0 <= value <= 1 for value in metrics)
        # Verify complete-verifier dispatch without running an external solver.
        with patch('CTRAIN.model_wrappers.model_wrapper.eval_complete', return_value=(1., 1.)) as complete:
            wrapper.evaluate_complete(loader, test_samples=2, results_path=str(root / 'verify'))
        assert complete.call_args.kwargs['engine'] == 'abcrown'
        torch.testing.assert_close(complete.call_args.kwargs['eps_std'].reshape(-1), torch.tensor([.05]))
    print('IBP-R public integration checks passed')


if __name__ == '__main__':
    check()
