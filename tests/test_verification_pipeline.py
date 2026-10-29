"""Shared screening, adaptive indexing, abCROWN config and failure regressions."""
import json
import io
from contextlib import redirect_stdout
import pickle
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, TensorDataset
import yaml
from CTRAIN.eval.eval import eval_adaptive, eval_complete
from CTRAIN.complete_verification.abCROWN.verify import abcrown_eval, limited_abcrown_eval
from CTRAIN.complete_verification.abCROWN.util import get_abcrown_standard_conf


def loader_for(values):
    loader = DataLoader(TensorDataset(torch.tensor(values).reshape(-1, 1),
                                     torch.zeros(len(values), dtype=torch.long)), batch_size=2)
    loader.min, loader.max, loader.std = torch.tensor(-2.), torch.tensor(5.), torch.tensor(1.)
    return loader


def main():
    loader = loader_for([0., 1., 2., 3.])
    seen = []
    def bound(method, winning):
        def compute(**kwargs):
            values = kwargs['data'].flatten().tolist()
            seen.append((method, values))
            lower = torch.tensor([[1. if value == winning else -1.] for value in values])
            return lower, lower
        return compute
    with patch('CTRAIN.eval.eval.bound_ibp', side_effect=bound('IBP', 0.)), \
         patch('CTRAIN.eval.eval.bound_crown_ibp', side_effect=bound('CROWN-IBP', 1.)), \
         patch('CTRAIN.eval.eval.bound_crown', side_effect=bound('CROWN', 2.)):
        count, total, certified, records = eval_adaptive(torch.nn.Identity(), 0.1, loader, test_samples=3, device='cpu')
    assert (count.item(), total) == (3, 3)
    assert certified.tolist() == [True, True, True]
    assert list(records) == [0, 1, 2]
    assert seen == [('IBP', [0., 1.]), ('CROWN-IBP', [1.]), ('IBP', [2.]),
                    ('CROWN-IBP', [2.]), ('CROWN', [2.])], seen

    class Model(torch.nn.Module):
        def forward(self, x):
            return torch.cat((x, -x), dim=1)
    def incomplete(**kwargs):
        assert kwargs['data_loader'].dataset.indices == [1, 2, 3]
        return 1, 3, torch.tensor([True, False, False]), {
            i: {'method': 'IBP', 'running_time': .1} for i in range(3)}
    def attack(**kwargs):
        assert kwargs['data_loader'].dataset.indices == [2, 3]
        return .5, torch.tensor([True, False]), {
            i: {'method': 'PGD', 'running_time': .2} for i in range(2)}
    with tempfile.TemporaryDirectory() as directory, \
         patch('CTRAIN.eval.eval.eval_adaptive', side_effect=incomplete), \
         patch('CTRAIN.eval.eval.eval_adversarial', side_effect=attack), \
         patch('CTRAIN.eval.eval.export_onnx', side_effect=lambda **kw: Path(kw['file_name']).write_bytes(b'toy-onnx')), \
         patch('CTRAIN.eval.eval.abcrown_eval', return_value=(.3, 'unsat')) as solve:
        result = eval_complete(Model(), torch.tensor(.1), loader_for([-1., 1., 1., 1., 1.]),
                               n_classes=2, device='cpu', test_samples=4, results_path=directory,
                               abcrown_config_dict={'bab': {'hugetensor_allocator': True}})
        assert result == (.5, .5)
        assert solve.call_count == 1
        assert solve.call_args.kwargs['config']['bab']['hugetensor_allocator'] is False
        records = json.loads((Path(directory) / 'results.json').read_text())
        assert [records[str(i)]['method'] for i in range(4)] == ['clean_classification', 'IBP', 'PGD', 'abCROWN']
        assert abs(records['3']['running_time'] - .6) < 1e-6

    for elapsed, failure, expected in [(11., False, 'timeout'), (1., False, 'unknown'), (1., True, 'unknown')]:
        class FakeCrown:
            def __init__(self, argv):
                self.config = yaml.safe_load(Path(argv[1]).read_text())
                assert self.config['bab']['hugetensor_allocator'] is False
                self.logger = SimpleNamespace(timeout_threshold=10.)
            def main(self):
                if failure:
                    raise RuntimeError('CUDA out of memory')
                with open(self.config['general']['output_file'], 'wb') as file:
                    pickle.dump({'results': 'timeout', 'time': elapsed}, file)
        config = get_abcrown_standard_conf()
        config['bab']['hugetensor_allocator'] = True
        with patch('CTRAIN.complete_verification.abCROWN.verify.ABCROWN', FakeCrown), \
             patch('CTRAIN.complete_verification.abCROWN.verify.read_vnnlib'), redirect_stdout(io.StringIO()):
            _, status = abcrown_eval(config, 1, 'fake.vnnlib', timeout=10.)
        assert status == expected, (status, expected)
        assert config['bab']['hugetensor_allocator'] is True

    with tempfile.TemporaryDirectory() as directory:
        runner = Path(directory) / 'runner.py'
        runner.write_text("import pickle, sys\nwith open(sys.argv[2], 'wb') as f: pickle.dump((.1, 'unsat'), f)\n")
        assert limited_abcrown_eval(runner_path=str(runner), timeout=1.) == (.1, 'unsat')
        runner.write_text('raise SystemExit(1)\n')
        assert limited_abcrown_eval(runner_path=str(runner), timeout=1.)[1] == 'unknown'
        runner.write_text('import time\ntime.sleep(10)\n')
        assert limited_abcrown_eval(runner_path=str(runner), timeout=.1)[1] == 'timeout'
    print('Verification pipeline regressions passed')


if __name__ == '__main__':
    main()
