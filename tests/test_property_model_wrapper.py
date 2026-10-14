"""Runnable CPU regression for the restored dev property-training API."""
import copy
import tempfile
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from CTRAIN import PropertyModelWrapper
from CTRAIN.model_wrappers import PropertyModelWrapper as ExportedWrapper
from CTRAIN.properties import And, Or, Outputs, Box, PropertySpec
from CTRAIN.eval.properties import partition_state, validate_partition
from CTRAIN.train.certified import property_train_model


def loader():
    inputs = torch.tensor([[-1.], [-.5], [.5], [1.]], dtype=torch.float64)
    return DataLoader(TensorDataset(inputs, torch.full_like(inputs, .4)),
                      batch_size=2, shuffle=True, generator=torch.Generator().manual_seed(9))


def wrapper(model=None, properties=None, **kwargs):
    if model is None:
        model = nn.Linear(1, 1).double()
        with torch.no_grad():
            model.weight.fill_(.2)
            model.bias.fill_(-.1)
    return PropertyModelWrapper(
        model, (1,), properties=properties or [PropertySpec('positive', Outputs()[0] >= 0,
                                                           [Box([-1.], [1.])])],
        task_loss=nn.MSELoss(), num_epochs=3, device='cpu', lr=.05,
        property_weight=5., property_batch_size=4, bound_method='IBP', **kwargs)


def main():
    torch.set_num_threads(1)
    assert PropertyModelWrapper is ExportedWrapper
    assert callable(property_train_model)
    model = nn.Linear(1, 1).double()
    with torch.no_grad():
        model.weight.fill_(.1)
        model.bias.fill_(.2)
    y = Outputs()
    spec = PropertySpec('logical', And(y[0] >= .05, Or(y[0] <= .5, y[0] >= .05)),
                        [Box([-1.], [1.])])
    bounded = wrapper(model, [spec])
    assert model.training
    for method in ('IBP', 'CROWN-IBP', 'CROWN'):
        report = bounded.certify_properties(method=method, refine=False)
        assert report['certified'] and report['properties'][0]['certified']
        assert len(report['cells']) == 1 and model.training
    lower = bounded.bounds(0, [spec.domains[0]], 'CROWN')
    torch.testing.assert_close(lower[:, 0], torch.tensor([.05], dtype=torch.float64))
    lower[:, 0].sum().backward()
    torch.testing.assert_close(model.weight.grad, torch.tensor([[-1.]], dtype=torch.float64))
    torch.testing.assert_close(model.bias.grad, torch.ones_like(model.bias))
    partition = partition_state(bounded.partition)
    outside = PropertySpec('outside', y[0] >= 1., [Box([-1.], [1.])])
    report = bounded.certify_properties(properties=[outside], refine=False)
    assert not report['certified'] and report['cells'][0]['outcome'] == 'unknown'
    assert partition_state(bounded.partition) == partition
    assert bounded.properties[0].name == 'logical'

    # Refinement must preserve the complete original box and its sampling weight.
    refinement = wrapper(refinement=True, max_cells=4, max_depth=2, split_count=1)
    refinement.train_model(loader(), end_epoch=1)
    validate_partition(refinement.partition, refinement.properties)
    assert len(refinement.partition) > 1
    assert sum(cell['probability'] for cell in refinement.partition) == 1.
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in refinement.original_model.parameters())

    initial = copy.deepcopy(wrapper().original_model)
    full = wrapper(copy.deepcopy(initial), weight_ramp_epochs=2,
                   lr_scheduler_func=torch.optim.lr_scheduler.StepLR,
                   lr_scheduler_kwargs={'step_size': 1, 'gamma': .9})
    torch.manual_seed(13)
    full.train_model(loader())
    interrupted = wrapper(copy.deepcopy(initial), weight_ramp_epochs=2,
                          lr_scheduler_func=torch.optim.lr_scheduler.StepLR,
                          lr_scheduler_kwargs={'step_size': 1, 'gamma': .9})
    torch.manual_seed(13)
    training = loader()
    interrupted.train_model(training, end_epoch=1)
    with tempfile.TemporaryDirectory(prefix='ctrain-property-restore-') as directory:
        checkpoint = Path(directory) / 'epoch.pt'
        interrupted.save_checkpoint(checkpoint, training)
        try:
            interrupted.save_checkpoint(checkpoint)
        except FileExistsError:
            pass
        else:
            raise AssertionError('Existing trajectory checkpoint was overwritten')
        resumed = wrapper(copy.deepcopy(initial), weight_ramp_epochs=2,
                          lr_scheduler_func=torch.optim.lr_scheduler.StepLR,
                          lr_scheduler_kwargs={'step_size': 1, 'gamma': .9})
        resumed.load_checkpoint(checkpoint)
        resumed.train_model(loader())
        assert resumed.epoch == full.epoch == 3
        assert resumed.history == full.history
        assert resumed.lr_scheduler.state_dict() == full.lr_scheduler.state_dict()
        for name, value in full.original_model.state_dict().items():
            torch.testing.assert_close(value, resumed.original_model.state_dict()[name], rtol=0, atol=0)
    print('Property wrapper imports, Boolean/affine bounds, gradients, refinement and checkpoint resumption passed')


if __name__ == '__main__':
    main()
