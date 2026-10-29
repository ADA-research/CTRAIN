"""Runnable NeuralSAT adapter and shared evaluation regression (no solver install)."""
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader, TensorDataset
from CTRAIN.complete_verification.neuralsat import neuralsat_eval, validate_neuralsat_command
from CTRAIN.eval.eval import eval_complete, eval_complete_abcrown


def main():
    bundled = validate_neuralsat_command(None)
    assert bundled[0] == sys.executable
    assert Path(bundled[1]).is_file()
    from contextlib import ExitStack
    with ExitStack() as stack:
        def incomplete(**kwargs):
            count = kwargs['test_samples']
            return 0, count, torch.zeros(count, dtype=torch.bool), {
                i: {'result': None, 'method': 'CROWN', 'running_time': 0.} for i in range(count)}
        def no_attack(**kwargs):
            count = kwargs['test_samples']
            return 1., torch.zeros(count, dtype=torch.bool), {
                i: {'result': None, 'method': 'PGD', 'running_time': 0.} for i in range(count)}
        stack.enter_context(patch('CTRAIN.eval.eval.eval_adaptive', side_effect=incomplete))
        stack.enter_context(patch('CTRAIN.eval.eval.eval_adversarial', side_effect=no_attack))
        main_checks()


def main_checks():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        runner = root / 'solver.py'
        # Instrumented Python startup must not consume normal result-case budgets.
        for body, expected, timeout in [
            ("p.write_text('unsat\\n')", 'unsat', 5.),
            ("p.write_text('sat,0.1\\n(X_0 0.5)')", 'sat', 5.),
            ("p.write_text('timeout\\n')", 'timeout', 5.),
            ("p.write_text('invalid_counterexample\\n')", 'unknown', 5.),
            ("print('unsat')", 'unknown', 5.),
            ("p.write_text('unsat\\n'); raise SystemExit(1)", 'unknown', 5.),
            ("import time; time.sleep(5)", 'timeout', .2),
        ]:
            runner.write_text("import sys\nfrom pathlib import Path\np = Path(sys.argv[sys.argv.index('--result_file') + 1])\n" + body)
            runtime, result = neuralsat_eval(root / 'model.onnx', root / 'spec.vnnlib',
                                            [sys.executable, str(runner)], timeout=timeout)
            assert result == expected, (result, expected)
            assert runtime >= 0

        loader = DataLoader(TensorDataset(torch.tensor([[[[0.05]]], [[[0.5]]], [[[0.95]]]]),
                                         torch.zeros(3, dtype=torch.long)), batch_size=2)
        loader.normalised = False
        loader.std, loader.min, loader.max = torch.tensor(1.), torch.tensor(0.), torch.tensor(1.)
        seen = []
        def verify(model_path, spec, command, **kwargs):
            text = Path(spec).read_text()
            assert '(and (<= Y_0 Y_1))' in text
            seen.append(text)
            return 0.1, ['unsat', 'sat'][len(seen) - 1]
        model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(1, 2)).eval()
        with torch.no_grad():
            model[1].weight.zero_()
            model[1].bias.copy_(torch.tensor([1., 0.]))
        with patch('CTRAIN.eval.eval.neuralsat_eval', side_effect=verify):
            result = eval_complete(model, torch.tensor(0.1), loader,
                                   n_classes=2, input_shape=[1, 1, 1], device='cpu',
                                   engine='neuralsat', neuralsat_command=[sys.executable, str(runner)],
                                   results_path=directory, start_idx=1)
        assert result == (0.5, 0.5), result
        assert len(seen) == 2
        saved = json.loads((root / 'results.json').read_text())
        assert saved['1']['method'] == 'neuralsat'
        try:
            eval_complete(torch.nn.Identity(), torch.tensor(0.1), loader, device='cpu',
                          results_path=directory, warm_start=True, start_idx=1)
        except ValueError as exc:
            assert 'different verification engine' in str(exc)
        else:
            raise AssertionError('Cross-engine warm start accepted')
        from CTRAIN.model_wrappers.model_wrapper import CTRAINWrapper
        wrapper = SimpleNamespace(eps=0.1, device='cpu', bounded_model=model,
                                  n_classes=2, input_shape=[1, 1, 1])
        with patch('CTRAIN.model_wrappers.model_wrapper.eval_acc', return_value=1.), \
             patch('CTRAIN.model_wrappers.model_wrapper.eval_complete', return_value=(0.5, 0.5)) as evaluate:
            assert CTRAINWrapper.evaluate_complete(
                wrapper, loader, engine='neuralsat',
                neuralsat_command=[sys.executable, str(runner)], neuralsat_batch_size=23,
            ) == (1., 0.5, 0.5)
            assert evaluate.call_args.kwargs['engine'] == 'neuralsat'
            assert evaluate.call_args.kwargs['neuralsat_batch_size'] == 23
        with patch('CTRAIN.eval.eval.eval_complete', return_value=(0.2, 0.3)) as shared:
            assert eval_complete_abcrown('model') == (0.2, 0.3)
            shared.assert_called_once_with('model')
    print('NeuralSAT regressions passed')


if __name__ == '__main__':
    main()
