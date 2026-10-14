"""General ONNX/VNNLIB import: polarity, boundaries, dimensions and training."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from CTRAIN import PropertyModelWrapper
from CTRAIN.properties import load_vnnlib_property
from CTRAIN.properties.vnnlib import _input_clauses
from CTRAIN.data_loaders import load_vnnlib_problem
from CTRAIN.util import export_onnx


def rejects(function):
    try:
        function()
    except ValueError:
        return
    raise AssertionError('Expected ValueError')


def specification(outputs=2, predicate='(<= Y_0 Y_1)'):
    return '\n'.join([
        '(set-logic QF_LRA)',
        *[f'(declare-const X_{i} Real)' for i in range(2)],
        *[f'(declare-const Y_{i} Real)' for i in range(outputs)],
        '(assert (and (<= 0 X_0) (<= (* 2 X_0) 2)))',
        '(assert (>= X_1 -1))', '(assert (<= X_1 1))',
        f'(assert {predicate})',
    ])


def main():
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        path = root / 'my_property.vnnlib'
        source = specification(predicate='(or (<= (+ Y_0 (* 2 Y_1)) (/ 1 2)) (not (< Y_1 0)))')
        path.write_text(source)
        unsafe = load_vnnlib_property(path)
        safe = load_vnnlib_property(path, output_semantics='safe')
        values = torch.tensor([[-1., -1.], [0., 0.], [2.5, -1.], [3., -1.], [1., 1.]])
        actual_unsafe = (values[:, 0] + 2 * values[:, 1] <= .5) | ~(values[:, 1] < 0)
        assert torch.equal(unsafe.formula.evaluate(values).flatten(), ~actual_unsafe)
        assert torch.equal(safe.formula.evaluate(values).flatten(), actual_unsafe)
        assert unsafe.domains[0].lower.tolist() == [0., -1.]
        assert unsafe.domains[0].upper.tolist() == [1., 1.]
        path.write_text(source + '\n(assert (or (<= X_0 .25) (>= X_0 .75)))')
        union = load_vnnlib_property(path)
        assert len(union.domains) == 2
        assert [b.lower.flatten()[0].item() for b in union.domains] == [0., .75]
        assert [b.upper.flatten()[0].item() for b in union.domains] == [.25, 1.]
        assert not any((b.lower <= torch.tensor([.5, 0.])).all() and
                       (torch.tensor([.5, 0.]) <= b.upper).all() for b in union.domains)
        rejects(lambda: _input_clauses(['and', *[
            ['or', ['<=', 'X_0', '0'], ['>=', 'X_0', '1']] for _ in range(11)]]))
        path.write_text(source)
        reshaped = load_vnnlib_property(path, input_shape=(1, 2))
        assert reshaped.domains[0].lower.shape == (1, 2)
        # Negate the conjunction of separate assertions, preserving a strict boundary.
        path.write_text(specification() + '\n(assert (< Y_1 2))')
        multiple = load_vnnlib_property(path)
        values = torch.tensor([[0., 0.], [2., 2.], [3., 1.]])
        assert multiple.formula.evaluate(values).flatten().tolist() == [False, True, True]
        for invalid in (
                specification(predicate='(<= (* Y_0 Y_1) 0)'),
                specification(predicate='(= Y_0 Y_1)'),
                specification(predicate='(<= Y_3 0)'),
                specification(predicate='(or (<= X_0 .5) (>= Y_0 .7))'),
                specification(predicate='(<= X_0 Y_0)'),
                specification() + '\n(assert (<= (+ X_0 X_1) 0))',
                specification().replace('(assert (<= X_1 1))', ''),
                specification().replace('(assert (<= X_1 1))', '(assert (<= X_1 -2))'),
                specification().replace('(declare-const X_0 Real)', '(declare-const X_2 Real)'),
                specification() + '\n(assert (<= (* 0 X_0) 1))',
                '(assert (<= Y_0 0)',
        ):
            path.write_text(invalid)
            rejects(lambda: load_vnnlib_property(path))
        path.write_text(source)
        rejects(lambda: load_vnnlib_property(path, input_shape=(3,)))
        rejects(lambda: load_vnnlib_property(path, output_semantics='unsure'))
        assert not list(root.glob('*.compiled'))
        # Arbitrary dimensions and filenames, no preset manifest or network registry.
        path.write_text(specification(outputs=3, predicate='(or (<= Y_0 -1) (<= Y_2 Y_1))'))
        original = nn.Linear(2, 3).double()
        with torch.no_grad():
            original.weight.zero_()
            original.bias.copy_(torch.tensor([1., 1., 2.], dtype=torch.double))
        model_path = root / 'my_network.onnx'
        export_onnx(original, str(model_path), 1, (2,))
        model, spec, provenance = load_vnnlib_problem(model_path, path)
        assert spec.name == 'my_property' and next(model.parameters()).dtype == torch.float64
        assert provenance['input_shape'] == [2] and provenance['output_semantics'] == 'unsafe'
        shaped_path = root / 'shaped.onnx'
        shaped = nn.Sequential(nn.Flatten(), original)
        export_onnx(shaped, str(shaped_path), 1, (1, 2))
        shaped_model, shaped_spec, _ = load_vnnlib_problem(shaped_path, path)
        assert shaped_spec.domains[0].lower.shape == (1, 2)
        torch.testing.assert_close(shaped_model(torch.zeros(3, 1, 2, dtype=torch.double)),
                                   shaped(torch.zeros(3, 1, 2, dtype=torch.double)))
        rejects(lambda: load_vnnlib_problem(model_path, path, expected_sha256={}))
        checked, _, _ = load_vnnlib_problem(model_path, path, expected_sha256=provenance['sha256'])
        torch.testing.assert_close(checked(torch.zeros(3, 2, dtype=torch.double)),
                                   original(torch.zeros(3, 2, dtype=torch.double)))
        wrapper = PropertyModelWrapper(model, (2,), properties=[spec], task_loss=nn.MSELoss(),
                                       num_epochs=1, property_batch_size=2)
        data = torch.tensor([[0., 0.], [.5, .5]], dtype=torch.double)
        wrapper.train_model(DataLoader(TensorDataset(data, original(data).detach()), batch_size=2))
        assert wrapper.certify_properties(method='CROWN-IBP')['certified']
        path.write_text(specification(outputs=2))
        rejects(lambda: load_vnnlib_problem(model_path, path))
        # CLI uses the selected pair and inferred input shape, preserving float64.
        path.write_text(specification(outputs=3, predicate='(or (<= Y_0 -1) (<= Y_2 Y_1))'))
        extra = root / 'extra_property.vnnlib'
        extra.write_text(specification(outputs=3, predicate='(<= Y_0 0)').replace(
            '(assert (and (<= 0 X_0) (<= (* 2 X_0) 2)))', '(assert (and (>= X_0 2) (<= X_0 3)))'))
        third = root / 'third_property.vnnlib'
        third.write_text(specification(outputs=3, predicate='(<= Y_1 0)').replace(
            '(assert (and (<= 0 X_0) (<= (* 2 X_0) 2)))', '(assert (and (>= X_0 -3) (<= X_0 -2)))'))
        output = root / 'run'
        subprocess.run([sys.executable, '-m', 'case_studies.acasxu.run', '--model', str(model_path),
                        '--specification', str(path), '--additional-specifications', str(extra), str(third),
                        '--epochs', '1', '--output', str(output)],
                       check=True, stdout=subprocess.DEVNULL)
        results = json.loads((output / 'results.json').read_text())
        config = json.loads((output / 'config.json').read_text())
        assert results['after']['certified'] and config['case'] == 'user_selected'
        assert config['artifacts']['sha256'] == provenance['sha256']
        names = {path.stem, extra.stem, third.stem}
        assert {prop['name'] for prop in config['training']['properties']} == names
        assert {prop['name'] for prop in results['after']['properties'] if prop['certified']} == names
        assert sum(value['train_samples'] for value in results['property_metrics'].values()) == 4096
        assert sum(value['validation_samples'] for value in results['property_metrics'].values()) == 2048
        assert all(value['train_samples'] > 0 and value['validation_samples'] > 0
                   for value in results['property_metrics'].values())
        assert len(config['additional_specification_sha256']) == 2
        # The comparison must detect a deliberately introduced property regression.
        benchmark = root / 'benchmark'
        (benchmark / 'vnnlib').mkdir(parents=True)
        (benchmark / 'vnnlib/prop_1.vnnlib').write_text(path.read_text())
        (benchmark / 'instances.csv').write_text('onnx/my_network.onnx,vnnlib/prop_1.vnnlib,1\n')
        weights = torch.load(output / 'model.pt', weights_only=True)
        bias_name = next(key for key, value in model.named_parameters()
                         if key.endswith('bias') and value.numel() == 3)
        weights[bias_name][2] = 0
        torch.save(weights, output / 'model.pt')
        comparison = root / 'comparison'
        subprocess.run([sys.executable, '-m', 'case_studies.acasxu.evaluate', '--run', str(output),
                        '--benchmark', str(benchmark), '--samples', '16', '--max-cells', '2',
                        '--output', str(comparison)], check=True, stdout=subprocess.DEVNULL)
        result = json.loads((comparison / 'results.json').read_text())['properties'][0]
        assert result['benchmark_applicable']
        assert result['before']['outcome'] == 'certified' and result['after']['outcome'] == 'violated'
        assert result['newly_violating_inputs'] == 16 and result['fixed_inputs'] == 0
        alpha_output = root / 'alpha_comparison'
        subprocess.run([sys.executable, '-m', 'case_studies.acasxu.evaluate', '--run', str(output),
                        '--benchmark', str(benchmark), '--samples', '16', '--method', 'ALPHA-CROWN',
                        '--alpha-iterations', '4', '--assigned-only', '--cover-from',
                        str(comparison / 'results.json'), '--output', str(alpha_output)],
                       check=True, stdout=subprocess.DEVNULL)
        alpha_report = json.loads((alpha_output / 'results.json').read_text())
        assert alpha_report['cover_sha256'] and alpha_report['alpha_iterations'] == 4
        alpha_result = alpha_report['properties'][0]
        assert alpha_result['before']['outcome'] == 'certified'
        assert alpha_result['after']['outcome'] == 'violated'
        assert alpha_result['before']['certificate']['method'] == 'ALPHA-CROWN'
        extra.write_text(extra.read_text() + '\n; changed source\n')
        rejected = subprocess.run([sys.executable, '-m', 'case_studies.acasxu.evaluate', '--run', str(output),
            '--benchmark', str(benchmark), '--output', str(root / 'changed_source')], capture_output=True)
        assert rejected.returncode != 0 and b'Training specification hash mismatch' in rejected.stderr
    print('VNNLIB polarity, ties, affine logic, input bounds, arbitrary models and CLI checks passed')


if __name__ == '__main__':
    main()
