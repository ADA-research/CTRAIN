"""Small real Optuna/CC-Dist run through the ICML entrypoint; no dataset downloads."""
import copy
import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import optuna
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from papers.rethinking_evaluation_paradigms.mo_hpo import run_hpo
from papers.rethinking_evaluation_paradigms.mo_hpo.front_utils import trial_config_hash


def check_cuda(teacher_path):
    """Check the submitted CNN7 batch size, trained teacher, loss and CROWN on GPU."""
    from auto_LiRPA import PerturbationLpNorm
    from CTRAIN.bound import bound_crown
    from CTRAIN.model_wrappers.configs import get_config_space
    assert torch.cuda.is_available()
    torch.set_num_threads(1)
    torch.manual_seed(0)
    model = run_hpo.build_model('cnn7', (3, 32, 32), 10)
    teacher = copy.deepcopy(model)
    saved = torch.load(teacher_path, map_location='cpu', weights_only=True)
    teacher.load_state_dict(saved.get('model_state_dict', saved.get('model', saved)))
    wrapper = run_hpo.build_wrapper('cc_dist', model, (3, 32, 32), 2 / 255,
                                    160, 'cuda', teacher=teacher)
    loaders, _, _ = run_hpo.build_loaders('cifar10', 'cnn7', 512,
                                        '/storage/work/kaulen/ctrain_dev/CTRAIN/data')
    loader, _ = loaders
    x, y = next(iter(loader))
    x, y = x.cuda(), y.cuda()
    radius = (2 / 255 / loader.std).view(3, 1, 1).cuda()
    lower = torch.maximum(x - radius, loader.min.cuda())
    upper = torch.minimum(x + radius, loader.max.cuda())
    hp = get_config_space(wrapper, 160, 2 / 255)['cc_dist:distillation_weight']
    wrapper.train()
    loss, = wrapper._loss_func(wrapper.bounded_model, model,
        PerturbationLpNorm(eps=radius, x_L=lower, x_U=upper), x, y, 10,
        nn.CrossEntropyLoss(reduction='none'), .5, distillation_weight=hp.upper,
        n_steps=10, step_size=.25, pgd_eps_factor=3., eps=radius,
        data_min=loader.min.cuda(), data_max=loader.max.cuda(), device='cuda')
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in wrapper.parameters() if p.grad is not None)
    assert any(p.grad is not None and p.grad.abs().max() > 0 for p in wrapper.parameters())
    wrapper.optimizer.step()
    wrapper.eval()
    with torch.no_grad():
        bounds, _ = bound_crown(wrapper.bounded_model,
            PerturbationLpNorm(eps=radius, x_L=lower[:1], x_U=upper[:1]), x[:1], y[:1], 10)
    assert bounds.shape == (1, 9) and torch.isfinite(bounds).all()
    assert all(p.grad is None and not p.requires_grad for p in wrapper.teacher_features.parameters())
    print(json.dumps(dict(check='CC-Dist HPO CUDA batch/loss/gradient/CROWN',
        batch_size=512, crown_batch_size=1, loss=loss.item(),
        max_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        device=torch.cuda.get_device_name()), indent=2))


def main():
    torch.set_num_threads(1)
    torch.manual_seed(4)
    teacher = nn.Sequential(nn.Flatten(), nn.Linear(2, 3), nn.ReLU(), nn.Linear(3, 2))
    x = .2 + .6 * torch.rand(8, 2, 1, 1)
    y = torch.tensor([0, 1] * 4)
    loader = DataLoader(TensorDataset(x, y), batch_size=4)
    loader.normalised = False
    loader.std = torch.ones(2)
    loader.mean = torch.zeros(2)
    loader.min, loader.max = torch.zeros(1, 2, 1, 1), torch.ones(1, 2, 1, 1)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        checkpoint = root / 'teacher.pt'
        torch.save(dict(config={'method': 'pgd'}, model=teacher.state_dict()), checkpoint)
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        argv = ['--dataset', 'cifar10', '--network', 'cnn7', '--method', 'cc_dist',
                '--eps', str(2 / 255), '--epochs', '2', '--budget-trials', '2',
                '--sampler', 'botorch', '--device', 'cpu', '--eval-samples', '8',
                '--teacher', str(checkpoint), '--teacher-sha256', digest,
                '--output-root', str(root / 'studies')]
        with patch.object(run_hpo, 'build_loaders', return_value=((loader, loader), (2, 1, 1), 2)), \
             patch.object(run_hpo, 'build_model', side_effect=lambda *args: copy.deepcopy(teacher)):
            run_hpo.main(argv)
        output = next((root / 'studies').iterdir())
        manifest = json.loads((output / 'run_manifest.json').read_text())
        assert manifest['selection_split'] == 'test'
        assert manifest['teacher']['sha256'] == digest
        study = optuna.load_study(study_name='moctrain',
                                 storage=f'sqlite:///{output}/optuna_study.db')
        assert len(study.trials) == 2
        for trial in study.trials:
            assert trial.state == optuna.trial.TrialState.COMPLETE
            beta = trial.params['cc_dist:distillation_weight']
            assert .1 / 3 <= beta <= 50 / 3
            assert tuple(trial.values) == (trial.user_attrs['metrics']['nat_acc'],
                                           trial.user_attrs['metrics']['cert_acc'])
            path = output / 'nets' / f'{trial_config_hash(trial, "cc_dist", 2 / 255)}.pt'
            state = torch.load(path, weights_only=True)
            assert not any(key.startswith('teacher') for key in state)
            inference = run_hpo.build_wrapper('cc_dist', copy.deepcopy(teacher), (2, 1, 1),
                                              2 / 255, 1, 'cpu', for_evaluation=True)
            inference.load_state_dict(state)
            inference.eval()
            assert torch.isfinite(inference(x)).all()
        with patch.object(run_hpo, 'build_loaders') as build:
            try:
                run_hpo.main([*argv, '--teacher-sha256', '0' * 64])
            except ValueError as error:
                assert 'SHA256' in str(error)
            else:
                raise AssertionError('Changed teacher checkpoint accepted')
            build.assert_not_called()
        try:
            run_hpo.build_wrapper('cc_dist', teacher, (2, 1, 1), 2 / 255, 160, 'cpu')
        except ValueError as error:
            assert 'teacher' in str(error)
        else:
            raise AssertionError('HPO silently lost its teacher')
    print('CC-Dist ICML runner, real BoTorch trials, beta, provenance and inference checks passed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--teacher', type=Path)
    args = parser.parse_args()
    if args.cuda:
        if args.teacher is None:
            parser.error('--cuda requires --teacher')
        check_cuda(args.teacher)
    else:
        main()
