"""Swap class agreement, shared bound gradients and exact-point training."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.utils.data import TensorDataset

from CTRAIN.bound.properties import PropertyBounds, box_endpoints
from CTRAIN.model_definitions import MNISTAdditionClassifier, MNISTSwapGraph
from CTRAIN.properties.training import resolve_training_options
from CTRAIN.train.certified.losses.property_methods import get_property_training_loss
from case_studies.mnist_addition.data import MNISTPropertyDataset
from case_studies.mnist_addition.run import main, parse_args, specifications, swap_formula


def main_test():
    torch.set_num_threads(1)
    torch.manual_seed(3)
    formula = swap_formula()
    # Integer logits include ties, which must follow torch.argmax's first index.
    logits = torch.randint(0, 3, (128, 78)).float()
    original, swapped = logits.split(39, 1)
    predictions = lambda values: torch.stack([head.argmax(1) for head in values.split((10, 10, 19), 1)], 1)
    expected = (predictions(original) == predictions(swapped)[:, [1, 0, 2]]).all(1)
    assert torch.equal(formula.evaluate(logits).flatten(), expected)
    a, b, total = original.split((10, 10, 19), 1)
    agreeing = torch.cat((original, b, a, total), 1)
    assert formula.evaluate(agreeing).all()
    assert formula.evaluate(torch.zeros(1, 78)).all()
    assert formula.margin(torch.zeros(1, 78)).max() == 0
    unique = torch.zeros(1, 78)
    unique[:, [0, 10, 20, 39, 49, 59]] = 2
    assert formula.evaluate(unique).all()
    for start in (39, 49, 59):
        mismatch = unique.clone()
        mismatch[:, start + 1] = 3
        assert not formula.evaluate(mismatch).any()

    inputs = torch.randn(2, 2, 28, 28)
    pool = MNISTPropertyDataset(TensorDataset(inputs, torch.zeros(2, 3).long()), 0)
    boxes = [pool[i] for i in range(len(pool))]
    lower, upper = box_endpoints(boxes, inputs)
    assert torch.equal(lower, inputs) and torch.equal(upper, inputs)
    assert pool.to_dict()['kind'] == 'mnist_pair_points'
    model = MNISTAdditionClassifier(8)
    graph = MNISTSwapGraph(model)
    assert {id(p) for p in graph.parameters()} == {id(p) for p in model.parameters()}
    torch.testing.assert_close(graph(inputs), torch.cat((model(inputs), model(inputs.flip(1))), 1))
    spec = specifications(inputs, 0, property_kind='swap')
    bounds = PropertyBounds(graph, (2, 28, 28), [spec], {'property_bound': 'atomic'})
    for method in ('IBP', 'CROWN-IBP', 'CROWN'):
        torch.testing.assert_close(bounds(0, boxes, method).amin(1, keepdim=True), formula.margin(graph(inputs)),
                                   atol=2e-5, rtol=2e-5)
    # Constant operand heads disagree: the bound loss must update shared weights.
    with torch.no_grad():
        model[-1].weight.zero_()
        model[-1].bias.zero_()
        model[-1].bias[0] = 2
        model[-1].bias[11] = 2
        model[-1].bias[20] = 2
    loss = (.001 - bounds(0, boxes, 'CROWN')).relu().mean()
    assert loss > 0
    loss.backward()
    assert model[-1].bias.grad.abs().sum() > 0
    for method in ('MTL-IBP', 'SABR'):
        wrapper = SimpleNamespace(training_method=method, margin=.001, reference=inputs,
                                  verification_model=graph, bounds=bounds,
                                  training_options=resolve_training_options(
                                      method, {'pgd_steps': 1, 'pgd_restarts': 1})[1])
        model.zero_grad()
        method_loss = get_property_training_loss(wrapper, 0, boxes)
        assert torch.isfinite(method_loss) and method_loss > 0
        method_loss.backward()
        assert model[-1].bias.grad.abs().sum() > 0
    with torch.no_grad():
        model[-1].bias[11] = 0
        model[-1].bias[10] = 2
    assert (bounds(0, boxes, 'CROWN') > 0).all()

    assert parse_args(['--property', 'swap']).epsilon == 0
    for argv in (['--property', 'swap', '--task', 'regression'], ['--epsilon', '0']):
        with contextlib.redirect_stderr(io.StringIO()):
            try:
                parse_args(argv)
            except SystemExit as error:
                assert error.code == 2
            else:
                raise AssertionError(f'Invalid settings accepted: {argv}')
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / 'run'
        with contextlib.redirect_stdout(io.StringIO()), patch(
                'CTRAIN.train.certified.properties.get_property_training_loss',
                wraps=get_property_training_loss) as penalty:
            main(['--smoke', '--property', 'swap', '--skip-ce-baseline', '--property-boxes', '7',
                  '--batch-size', '9', '--property-batch-size', '3',
                  '--warm-up-epochs', '1', '--weight-ramp-epochs', '1',
                  '--output', str(output)])
        assert [len(call.args[2]) for call in penalty.call_args_list] == [3, 3, 3, 3]
        config = json.loads((output / 'config.json').read_text())
        result = json.loads((output / 'certified/results.json').read_text())
        assert config['epsilon'] == 0 and config['case'] == 'experimental_mnist_swap_equivariance'
        assert config['batch_size'] == 9 and config['property_batch_size'] == 3
        assert result['verification']['train']['total_boxes'] == 7
        assert result['verification']['test']['total_boxes'] == 16
        assert result['history'][0]['phase'] == 'warm_up'
        assert result['history'][0]['certified_fraction'] is None
        assert 0 <= result['history'][1]['certified_fraction'] <= 1
        history = json.loads((output / 'certified/checkpoints/history.json').read_text())
        assert history == result['history']
        assert len(history[1]['metrics']['test_accuracy_per_output']) == 3
        assert 0 <= history[1]['metrics']['test_empirical_swap_equivariant_fraction'] <= 1
        assert len(result['metrics']['test']['swap_agreement_per_output']) == 3
        assert (result['verification']['test']['certified_fraction'] <=
                result['metrics']['test']['empirical_swap_equivariant_fraction'])
    print('MNIST swap formula, shared bounds/gradients and point training checks passed.')


if __name__ == '__main__':
    main_test()
