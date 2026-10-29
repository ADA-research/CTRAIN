"""CPU regressions for shared trainer contracts and property decoupling."""
import copy
import tempfile
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from CTRAIN import GLSModelWrapper, SetBasedModelWrapper, PropertyModelWrapper
from CTRAIN.bound.properties import PropertyBounds
from CTRAIN.eval.properties import initial_partition, partition_state
from CTRAIN.properties import Box, Outputs, PropertySpec
from CTRAIN.train.certified import PropertyTrainingState, property_train_model
from CTRAIN.train.certified import (
    shi_train_model, crown_ibp_train_model, sabr_train_model,
    mtl_ibp_train_model, taps_train_model, staps_train_model,
)
from CTRAIN.train.certified.eps_scheduler import linear_schedule_fraction
from CTRAIN.train.certified.losses.property_methods import (
    get_property_training_loss, compute_property_training_loss,
)
from CTRAIN.train.certified.util import EpochMetrics, validate_training_run, resolve_validation_eps
from CTRAIN.util import save_checkpoint


def rejects(error, function):
    try:
        function()
    except error:
        return
    raise AssertionError(f'Expected {error.__name__}')


def classification_loader():
    x = torch.linspace(.1, .9, 20).reshape(5, 1, 2, 2)
    loader = DataLoader(TensorDataset(x, torch.tensor([0, 1, 0, 1, 0])), batch_size=3)
    loader.mean, loader.std = torch.zeros(1), torch.ones(1)
    loader.min, loader.max, loader.normalised = torch.tensor(0.), torch.tensor(1.), False
    return loader


def shared_helpers():
    loader = classification_loader()
    assert validate_training_run(loader, 3, 1, None, 10) == 3
    assert validate_training_run(loader, 3, 3, 3, 10) == 3
    for start, end, interval in ((True, 2, 1), (2, 1, 1), (0, 4, 1), (0, 2, 0)):
        rejects(ValueError, lambda: validate_training_run(loader, 3, start, end, interval))
    rejects(ValueError, lambda: validate_training_run(loader, 3, 0, 3, 1, history={}))
    assert resolve_validation_eps(.1, None) == resolve_validation_eps(None, .1) == .1
    rejects(ValueError, lambda: resolve_validation_eps(.1, .2))
    metrics = EpochMetrics()
    metrics.update(3, loss=2.)
    metrics.update(1, loss=10.)
    assert metrics.mean('loss') == 4.
    for batches in (1, 3, 7):
        for warm in (0, 1, 2):
            for ramp in (0, 1, 3):
                for epoch in range(6):
                    for batch in range(batches):
                        elapsed = epoch + batch / batches - warm
                        begin = min(1., max(0., elapsed / ramp)) if ramp else float(elapsed >= 0)
                        end = (0. if epoch < warm else
                               min(1., (epoch - warm + (batch + 1) / batches) / ramp) if ramp else 1.)
                        assert linear_schedule_fraction(epoch, batch, batches, warm, ramp) == begin
                        assert linear_schedule_fraction(epoch, batch, batches, warm, ramp, at_batch_end=True) == end
    # Empty/invalid ranges must not trigger Shi initialization in any classic trainer.
    for trainer in (shi_train_model, crown_ibp_train_model, sabr_train_model,
                    mtl_ibp_train_model, taps_train_model, staps_train_model):
        model = nn.Sequential(nn.Flatten(), nn.Linear(4, 2))
        before = copy.deepcopy(model.state_dict())
        options = dict(num_epochs=1, eps_schedule=(0, 0, 1),
                       optimizer=torch.optim.SGD(model.parameters(), lr=.01),
                       results_path=None, device='cpu')
        assert trainer(model, model, loader, end_epoch=0, **options) is model
        rejects(ValueError, lambda: trainer(model, model, loader, end_epoch=2, **options))
        for key, value in before.items():
            torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)


def explicit_properties():
    spec = PropertySpec('positive', Outputs()[0] >= 1., [Box([-1.], [1.])])
    initial = nn.Linear(1, 1).double()
    initial.weight.data.fill_(.2)
    initial.bias.data.fill_(-.1)
    x = torch.tensor([[-1.], [0.], [1.]], dtype=torch.double)
    loader = DataLoader(TensorDataset(x, torch.ones_like(x)), batch_size=2)
    wrapped = PropertyModelWrapper(copy.deepcopy(initial), (1,), properties=[spec],
        task_loss=nn.MSELoss(), num_epochs=2, training_method='IBP', property_batch_size=2,
        lr=.01, refinement=True, split_count=1)
    model = copy.deepcopy(initial)
    bounds = PropertyBounds(model, (1,), [spec])
    state = PropertyTrainingState(initial_partition([spec]))
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    rng = torch.get_rng_state()
    property_train_model(model, loader, bounds=bounds, task_loss=nn.MSELoss(),
        optimizer=optimizer, num_epochs=2, state=state, training_method='IBP',
        property_batch_size=2, refinement=True, split_count=1)
    torch.set_rng_state(rng)
    wrapped.train_model(loader)
    assert state.history == wrapped.history
    assert partition_state(state.partition) == partition_state(wrapped.partition)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, wrapped.original_model.state_dict()[key], rtol=0, atol=0)
    assert property_train_model(wrapped, loader, end_epoch=2) is wrapped.original_model
    assert property_train_model(wrapper=wrapped, train_loader=loader, end_epoch=2) is wrapped.original_model
    old_loss = get_property_training_loss(wrapped, 0, spec.domains)
    new_loss = compute_property_training_loss(wrapped.bounds, 0, spec.domains,
        verification_model=wrapped.verification_model, training_method=wrapped.training_method,
        training_options=wrapped.training_options, margin=wrapped.margin, reference=wrapped.reference)
    torch.testing.assert_close(old_loss, new_loss)
    old_grads = torch.autograd.grad(old_loss, tuple(wrapped.original_model.parameters()))
    new_grads = torch.autograd.grad(new_loss, tuple(wrapped.original_model.parameters()))
    for actual, expected in zip(new_grads, old_grads):
        torch.testing.assert_close(actual, expected)
    with tempfile.TemporaryDirectory() as root:
        path = Path(root) / 'epoch.pt'
        wrapped.save_checkpoint(path, loader)
        saved = torch.load(path, weights_only=False)
        assert {'model_state_dict', 'optimizer_state_dict', 'selection_state', 'history',
                'configuration', 'partition', 'rng', 'loader_rng'} <= saved.keys()
        original_bytes = path.read_bytes()
        rejects(FileExistsError, lambda: wrapped.save_checkpoint(path, loader))
        assert path.read_bytes() == original_bytes
        # Prior property checkpoints use their original field names.
        saved['model'] = saved.pop('model_state_dict')
        saved['optimizer'] = saved.pop('optimizer_state_dict')
        saved['scheduler'] = saved.pop('lr_scheduler_state_dict', None)
        selected = saved.pop('selection_state')
        saved['best_score'], saved['best_weights'] = selected.get('score'), selected.get('weights')
        torch.save(saved, Path(root) / 'legacy.pt')
        wrapped.load_checkpoint(Path(root) / 'legacy.pt')
        assert wrapped.epoch == 2
        assert property_train_model(wrapped, loader, val_loader=loader, end_epoch=2) is wrapped.original_model


def classifier_resume_and_modes():
    loader = classification_loader()
    initial = nn.Sequential(nn.Flatten(), nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))
    for cls in (GLSModelWrapper, SetBasedModelWrapper):
        for unit in ('epoch', 'batch'):
            with tempfile.TemporaryDirectory() as root:
                def make(path=None):
                    extra = dict(shi_init=False, bound_method='IBP') if cls is GLSModelWrapper else {}
                    return cls(copy.deepcopy(initial), (1, 2, 2), .1, 2,
                        device='cpu', warm_up_epochs=0, ramp_up_epochs=1,
                        lr_decay_kwargs=dict(milestones=(1,), gamma=.9),
                        lr_decay_schedule_unit=unit, checkpoint_save_path=path,
                        checkpoint_save_interval=10, **extra)
                full = make()
                from CTRAIN.model_wrappers.configs import get_config_space
                config = dict(get_config_space(full, 2, .1).get_default_configuration())
                if cls is SetBasedModelWrapper:
                    assert full._recreate_for_hpo(config, 2).lr_decay_schedule_unit == unit
                else:
                    (Path(root) / 'nets').mkdir()
                    observed_units = []
                    def hpo_train(candidate, *args, **kwargs):
                        observed_units.append(candidate.options['lr_decay_schedule_unit'])
                    with patch.object(GLSModelWrapper, 'train_model', autospec=True, side_effect=hpo_train), \
                         patch.object(GLSModelWrapper, '_evaluate_hpo_model', return_value=(.5, .4, .3)):
                        full._hpo_runner(config, 1, 2, loader, loader, root)
                    assert observed_units == [unit]
                rng = torch.get_rng_state()
                full.train_model(loader)
                partial = make(root)
                torch.set_rng_state(rng)
                partial.train_model(loader, end_epoch=1)
                checkpoint = next(Path(root).glob('*.pt'))
                saved_rng = torch.get_rng_state()
                resumed = make()
                torch.set_rng_state(saved_rng)
                resumed.resume_from_checkpoint(checkpoint, loader)
                assert resumed.history == full.history
                assert resumed.lr_scheduler.state_dict() == full.lr_scheduler.state_dict()
                for key, value in full.state_dict().items():
                    torch.testing.assert_close(value, resumed.state_dict()[key], rtol=0, atol=0)
                before = copy.deepcopy(resumed.state_dict())
                resumed.train_model(loader, start_epoch=2, end_epoch=2)
                for key, value in before.items():
                    torch.testing.assert_close(value, resumed.state_dict()[key], rtol=0, atol=0)
                # Invalid shared options fail before any model update.
                rejects(ValueError, lambda: resumed.train_model(loader, start_epoch=True))
    gls = GLSModelWrapper(copy.deepcopy(initial), (1, 2, 2), .1, 1, device='cpu',
        warm_up_epochs=0, ramp_up_epochs=0, shi_init=False)
    gls.original_model.eval()
    gls.bounded_model.eval()
    models = (gls.original_model, gls.bounded_model)
    modes = {module: module.training for model in models for module in model.modules()}
    flags = {p: p.requires_grad for model in models for p in model.parameters()}
    with patch.object(gls.optimizer, 'step', side_effect=RuntimeError('injected')):
        rejects(RuntimeError, lambda: gls.train_model(loader))
    assert all(module.training == training for module, training in modes.items())
    assert all(p.requires_grad == flag for p, flag in flags.items())
    # Adding a progress bar must not consume an extra DataLoader seed before GLS noise.
    rng = torch.get_rng_state()
    iter(loader)
    expected_rng = torch.get_rng_state()
    torch.set_rng_state(rng)
    observed = []
    def record_rng(closure):
        observed.append(torch.get_rng_state())
        return 0.
    with patch.object(gls.optimizer, 'step', side_effect=record_rng), \
         patch.object(gls.lr_scheduler, 'step'):
        gls.train_model(loader)
    assert torch.equal(observed[0], expected_rng)
    with tempfile.TemporaryDirectory() as root:
        rejects(ValueError, lambda: save_checkpoint(initial, gls.optimizer, 0., 1, root,
                                                   extra_state={'epoch': 8}))
        assert not list(Path(root).iterdir())


if __name__ == '__main__':
    torch.set_num_threads(1)
    torch.manual_seed(23)
    shared_helpers()
    explicit_properties()
    classifier_resume_and_modes()
    print('Shared training lifecycle, explicit properties, compatibility, checkpoints and resume checks passed')
