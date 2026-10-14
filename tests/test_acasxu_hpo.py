"""Numerical execution, complete benchmark enumeration and honest repair outcomes."""
import argparse
import json
import shutil
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import torch
from torch import nn

from case_studies.acasxu import hpo
from CTRAIN.eval.properties import export_property_graph, validate_partition
from CTRAIN.properties import Box, Outputs, PropertySpec


def main():
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        benchmark = root / 'benchmark'
        (benchmark / 'onnx').mkdir(parents=True)
        (benchmark / 'vnnlib').mkdir()
        model = nn.Linear(5, 5)
        with torch.no_grad():
            model.weight.zero_()
            model.bias.copy_(torch.arange(5.))
        model_path = benchmark / 'onnx' / 'ACASXU_run2a_1_1_batch_2000.onnx'
        export_property_graph(model, model_path, (5,), torch.zeros(2, 5))
        header = '\n'.join([f'(declare-const {v}_{i} Real)' for v in ('X', 'Y') for i in range(5)]
            + [f'(assert (>= X_{i} -1))\n(assert (<= X_{i} 1))' for i in range(5)])
        for prop, predicate in ((1, '(>= Y_0 4)'), (3, '(<= Y_0 Y_1)')):
            (benchmark / 'vnnlib' / f'prop_{prop}.vnnlib').write_text(header + f'\n(assert {predicate})\n')
        instances = benchmark / 'instances.csv'
        instances.write_text(f'onnx/{model_path.name},vnnlib/prop_1.vnnlib,1\n'
                             f'onnx/{model_path.name},vnnlib/prop_3.vnnlib,1\n')
        cases = hpo.benchmark_cases(benchmark)
        assert len(cases) == 1 and len(cases[0]['specifications']) == 2
        portable = root / 'portable'
        shutil.copytree(benchmark, portable)
        report = hpo.collect(portable)
        assert report['total_obligations'] == 2 and report['missing_models'] == ['1_1']
        specs = [PropertySpec('p', Outputs()[0] > 0, [Box([-1] * 5, [1] * 5), Box([0] * 5, [1] * 5)]),
                 PropertySpec('q', Outputs()[1] > 0, [Box([0] * 5, [1] * 5)])]
        cover = hpo.fixed_cover(specs, 8)
        validate_partition(cover, specs)
        assert len(cover) == 24
        safe_model = nn.Linear(5, 5)
        with torch.no_grad():
            safe_model.weight.zero_(); safe_model.bias.fill_(1)
        report = hpo.cover_metrics(hpo.PropertyBounds(safe_model, (5,), specs), cover, specs)
        assert report['certified'] and all(r['certified_fraction'] == 1 for r in report['properties'])
        inputs, _ = hpo.fidelity_inputs(specs, model.weight, 64, 1)
        assert inputs.shape == (64, 5) and (inputs >= -1).all() and (inputs <= 1).all()
        args = argparse.Namespace(benchmark=benchmark, network='1_1', output=root / 'repair',
            seed=1, epochs=1, trials=1, samples=16, train_samples=32, selection_cells=2,
            training_cells=2, solver_timeout=1., device='cpu')
        hpo.run(args)
        result = json.loads((args.output / 'results.json').read_text())
        assert result['completed_trials'] == 1 and result['selected_trial'] == 0
        assert result['status'] == 'repair_incomplete' and not result['verification']['certified']
        assert len(result['test']['properties']) == 2
        assert (args.output / 'optuna_study.db').is_file()
        assert (args.output / 'final' / 'model.onnx').is_file()
        assert (args.output / 'final' / 'config.json').is_file()
        assert hpo.objective_values(result['test'])[0] == 0
        try:
            hpo.run(args)
        except FileExistsError:
            pass
        else:
            raise AssertionError('Existing runs must not be overwritten')
        hpo.write_json(root / 'submission.json', dict(cases=cases))
        (args.output / 'results.json').rename(root / 'repair_result.json')
        assert hpo.collect(root)['missing_models'] == ['1_1']
        (root / '1_1').mkdir()
        (root / 'repair_result.json').rename(root / '1_1' / 'results.json')
        assert hpo.collect(root)['incomplete_repairs'] == ['1_1']
        # Sampled hybrid options must reach the actual trainer and selected artifacts.
        constructor = hpo.PropertyModelWrapper
        for method, name, limits in (('MTL-IBP', 'alpha', [0., 1.]),
                                     ('SABR', 'subselection_ratio', [.01, 1.])):
            wrappers = []
            def capture_wrapper(*positional, **keywords):
                wrapper = constructor(*positional, **keywords)
                wrappers.append(wrapper)
                return wrapper
            args.output = root / method
            args.training_method = method
            args.training_options = {'pgd_steps': 2, name: .25}
            with patch.object(hpo, 'PropertyModelWrapper', side_effect=capture_wrapper):
                hpo.run(args)
            config = json.loads((args.output / 'config.json').read_text())
            assert config['search_space'][name] == limits
            assert set(config['search_space']) & {'alpha', 'subselection_ratio'} == {name}
            study = hpo.optuna.load_study(study_name='acasxu_1_1',
                storage=f'sqlite:///{args.output}/optuna_study.db')
            trial = study.trials[0]
            parameters = json.loads((args.output / 'trials/000/config.json').read_text())
            assert parameters['training_options'][name] == trial.params[name]
            assert wrappers[0].training_options[name] == trial.params[name]
            assert wrappers[0].training_options['pgd_steps'] == 2
            assert args.training_options == {'pgd_steps': 2, name: .25}
            selected = json.loads((args.output / 'final/config.json').read_text())
            assert selected['training']['training_options'] == parameters['training_options']
            assert trial.distributions[name].log == (method == 'SABR')
        # Solver SAT alone is never reported as a validated violation or a proof.
        uncertain_spec = PropertySpec('p', Outputs()[0] > 0, [Box([-1] * 5, [1] * 5)])
        with patch.object(hpo, 'evaluate_cells', return_value=([dict(property='p', outcome='unknown')], 0)), \
                patch.object(hpo, 'find_counterexample', return_value=None), \
                patch('CTRAIN.complete_verification.abCROWN.verify.limited_abcrown_eval', return_value=(0., 'sat')):
            report = hpo.verify(safe_model, model_path, dict(specifications=[dict(path='unused', timeout=1)]),
                [uncertain_spec], dict(properties=[dict(counterexample=None)]), root / 'sat', 1)
        assert not report['certified'] and report['properties'][0]['outcome'] == 'unknown'
        # A model with all assigned properties certified is not retrained.
        instances.write_text(f'onnx/{model_path.name},vnnlib/prop_1.vnnlib,1\n')
        args.output = root / 'safe'
        hpo.run(args)
        result = json.loads((args.output / 'results.json').read_text())
        assert result['status'] == 'original_certified' and result['completed_trials'] == 0
    print('ACAS multiobjective repair workflow checks passed')


if __name__ == '__main__':
    main()
