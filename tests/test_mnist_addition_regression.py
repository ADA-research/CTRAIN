"""Check residual direction, lazy box limits and the regression training path."""
import contextlib
import io
import json
from pathlib import Path
import tempfile

import torch
from torch import nn

from CTRAIN.bound.properties import PropertyBounds
from CTRAIN.model_definitions import MNISTAdditionRegressor
from case_studies.mnist_addition.run import main, parse_args, specifications


def main_test():
    torch.set_num_threads(1)
    spec = specifications(torch.zeros(1, 2, 28, 28), .02, 'regression', .25)
    predictions = torch.tensor([[2., 3., 5.25], [2., 3., 5.5], [2., 3., -100.]])
    assert spec.formula.evaluate(predictions).flatten().tolist() == [True, False, True]
    model = MNISTAdditionRegressor(8)
    with torch.no_grad():
        model[-1].weight.zero_()
        model[-1].bias.copy_(torch.tensor([2., 3., 5.]))
    bounds = PropertyBounds(model, (2, 28, 28), [spec], {'property_bound': 'atomic'})
    torch.testing.assert_close(bounds(0, spec.domains, 'CROWN'), torch.tensor([[.25]]))
    nn.MSELoss()(model(torch.zeros(1, 2, 28, 28)), torch.tensor([[1., 4., 6.]])).backward()
    assert (model[-1].bias.grad != 0).all()

    for argv in (['--tau', '.25'], ['--task', 'regression', '--tau', 'nan'],
                 ['--task', 'regression', '--tau', '-1'], ['--property-boxes', '0'],
                 ['--property-boxes', '60001'], ['--property-batch-size', '0'],
                 ['--property-batch-size', '-1']):
        with contextlib.redirect_stderr(io.StringIO()):
            try:
                parse_args(argv)
            except SystemExit as error:
                assert error.code == 2
            else:
                raise AssertionError(f'Invalid settings accepted: {argv}')

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        with contextlib.redirect_stdout(io.StringIO()):
            main(['--smoke', '--task', 'regression', '--tau', '.25', '--property-boxes', '7',
                  '--batch-size', '8', '--warm-up-epochs', '1', '--weight-ramp-epochs', '1',
                  '--output', str(root / 'run')])
        config = json.loads((root / 'run/config.json').read_text())
        assert config['tau'] == .25 and config['outputs'] == ['a', 'b', 'sum']
        assert len(config['data']['provenance']['train']['pair_indices']) == 32
        result = json.loads((root / 'run/results.json').read_text())
        assert set(result['results']) == {'task_only', 'certified'}
        for arm in result['results']:
            saved = json.loads((root / f'run/{arm}/results.json').read_text())
            assert saved['verification']['train']['total_boxes'] == 7
            assert saved['verification']['test']['total_boxes'] == 16
            assert len(saved['metrics']['test']['mse_per_output']) == 3
            assert 'argmax_ties' not in saved['verification']['test']
            assert all(epoch['cells'] == 7 for epoch in saved['history'])
    print('MNIST regression property, training and box-count checks passed.')


if __name__ == '__main__':
    main_test()
