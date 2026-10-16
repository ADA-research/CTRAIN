"""Small execution and selection regressions for the arithmetic HPO experiment."""
import contextlib
import io
import json
import math
import shutil
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
from types import SimpleNamespace

import optuna
import torch

from case_studies.mnist_addition.run import make_wrapper, parse_args
from case_studies.mnist_addition.hpo import comparison, load_data, main


def main_test():
    torch.set_num_threads(1)
    args = parse_args(['--smoke'])
    pools, _, provenance = load_data(args, 'holdout', 8)
    train = set(provenance['train']['source_indices'])
    holdout = set(provenance['selection']['source_indices'])
    assert not train & holdout and train | holdout == set(range(32))
    for pair in provenance['train']['pair_indices']:
        assert set(pair) <= train
    for pair in provenance['selection']['pair_indices']:
        assert set(pair) <= holdout
    assert provenance['train']['unique_images_per_position'] == [24, 24]
    assert len(pools['train']) == 32
    args.seed += 1
    _, _, again = load_data(args, 'holdout', 8)
    assert again['selection'] == provenance['selection']
    assert again['test'] == provenance['test']
    _, _, test_protocol = load_data(args, 'test', 8)
    assert test_protocol['selection'] == test_protocol['test']
    assert test_protocol['train']['unique_images_per_position'] == [32, 32]

    def row(number, selection, test, arm='hpo'):
        def score(pair):
            return dict(metrics=dict(triple_accuracy=pair[0]), certified_fraction=pair[1])
        return dict(arm=arm, seed=0, trial=number, selection=score(selection), test=score(test))

    rows = [row(0, (.9, .3), (.1, .1)), row(1, (.5, .8), (.4, .4)),
            row(2, (.4, .2), (1., 1.)), row(None, (.95, .9), (.9, .9), 'fixed_certified')]
    report = comparison(rows)
    assert {row['trial'] for row in report['pareto_front']} == {0, 1}
    assert report['tuned']['clean']['trial'] == 0
    assert report['tuned']['consistency']['trial'] == report['tuned']['balanced']['trial'] == 1
    assert abs(report['selection_hypervolume']['hpo'] - .52) < 1e-12
    assert abs(report['test_hypervolume_of_selection_front'] - .16) < 1e-12

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        config = root / 'config.json'
        config.write_text(json.dumps(dict(epochs=2, train_pairs=60000, test_pairs=8,
            batch_size=16, property_batch_size=3, width=8,
            weight_ramp_epochs=1, property_weight=10., method='CROWN-IBP', eval_method='CROWN')))
        for method in ('IBP', 'CROWN-IBP', 'CROWN', 'MTL-IBP', 'SABR', 'PGD'):
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                main(['--config', str(config), '--output', str(root / 'dry'),
                      '--method', method, '--smoke', '--device', 'cpu', '--dry-run'])
            space = json.loads(buffer.getvalue())['search_space']
            assert space['bound_method'] == [method]
            expected = {'MTL-IBP': {'alpha'}, 'SABR': {'subselection_ratio'}}.get(method, set())
            assert set(space) & {'alpha', 'subselection_ratio'} == expected
        for method, name in (('MTL-IBP', 'alpha'), ('SABR', 'subselection_ratio')):
            output = root / method
            fixed = {'pgd_steps': 2, name: .25}
            with patch('case_studies.mnist_addition.hpo.make_wrapper', wraps=make_wrapper) as recreate, \
                    contextlib.redirect_stdout(io.StringIO()):
                main(['--config', str(config), '--output', str(output), '--method', method,
                      '--training-options', json.dumps(fixed), '--smoke', '--device', 'cpu',
                      '--trials', '1', '--sampler', 'nsgaii'])
            result = json.loads((output / 'results.json').read_text())
            study = optuna.load_study(study_name='mnist_addition',
                storage=f'sqlite:///{output}/optuna_study.db')
            trial = study.trials[0]
            for record in result['records']:
                checkpoint = torch.load(Path(record['directory']) / 'checkpoints/2_checkpoint.pt',
                                        weights_only=False)
                options = checkpoint['configuration']['training_options']
                assert options['pgd_steps'] == 2
                expected = trial.params[name] if record['arm'] == 'hpo' else .25
                assert options[name] == record['parameters']['training_options'][name] == expected
            assert [call.args[2]['training_options'][name] for call in recreate.call_args_list] == [
                record['parameters']['training_options'][name]
                for record in result['records'] if 'test' in record]
            assert trial.distributions[name].log == (method == 'SABR')
            if method == 'SABR':
                # The diagnostic must work for new studies, not just historical trial 98.
                from case_studies.mnist_addition import local_robustness
                from case_studies.mnist_addition.run import mnist_source
                study_root = root / 'diagnostic_study'
                shutil.copytree(output, study_root / 'seed_0')
                images, labels = mnist_source(parse_args(['--smoke']), False, 43, 16)
                diagnostic = root / 'diagnostic'
                with patch('torchvision.datasets.MNIST', return_value=SimpleNamespace(
                        data=images, targets=labels)) as dataset, \
                        patch.object(sys, 'argv', ['local_robustness', '--study-root', str(study_root),
                            '--trial', '0', '--data-root', str(root / 'cache'), '--output', str(diagnostic),
                            '--steps', '1', '--restarts', '1', '--limit', '1']), \
                        contextlib.redirect_stdout(io.StringIO()):
                    local_robustness.main()
                assert dataset.call_args.args[0] == root / 'cache'
                assert json.loads((diagnostic / 'results.json').read_text())['summary']['total_pairs'] == 1
        trained_methods = set()
        for seed in (0, 1, 2):
            output = root / f'seed_{seed}'
            with contextlib.redirect_stdout(io.StringIO()):
                main(['--config', str(config), '--output', str(output), '--seed', str(seed),
                      '--smoke', '--device', 'cpu', '--trials', '2', '--sampler', 'nsgaii'])
            result = json.loads((output / 'results.json').read_text())
            assert result['smoke'] and result['completed_trials'] == 2
            assert len(result['records']) == 4
            trained_methods.update(row['parameters']['bound_method'] for row in result['records'])
            assert len(result['pareto_front']) >= 1
            study = optuna.load_study(study_name='mnist_addition', storage=f'sqlite:///{output}/optuna_study.db')
            for trial in study.trials:
                record = next(row for row in result['records'] if row['trial'] == trial.number)
                assert tuple(trial.values) == (record['selection']['metrics']['triple_accuracy'],
                                               record['selection']['certified_fraction'])
                checkpoint = torch.load(Path(record['directory']) / 'checkpoints/2_checkpoint.pt',
                                        weights_only=False)
                for key in ('lr', 'property_weight', 'warm_up_epochs', 'weight_ramp_epochs'):
                    assert checkpoint['configuration'][key] == trial.params[key]
                assert 'batch_size' not in trial.params and 'log2_batch_size' not in trial.params
                assert checkpoint['configuration']['bound_method'] == trial.params['bound_method']
                assert not {'log2_property_batch_size', 'log2_property_boxes'} & trial.params.keys()
                assert record['parameters']['batch_size'] == 16
                assert record['parameters']['property_batch_size'] == 3
                assert checkpoint['configuration']['property_batch_size'] == 3
                assert checkpoint['configuration']['match_property_batch_size'] is False
                assert checkpoint['configuration']['property_dataset']['count'] == 32
                assert len(checkpoint['configuration']['properties'][0]['domains']) == 1
                assert checkpoint['partition'] == []
                assert all(entry['cells'] == 32 for entry in checkpoint['history'])
                for entry in checkpoint['history']:
                    if entry['phase'] == 'warm_up':
                        assert entry['property_loss'] == entry['effective_property_weight'] == 0
                        assert entry['certified'] is None and not entry['property_evaluated']
                expected_steps = 2 * math.ceil(32 / 16)
                assert all(float(state['step']) == expected_steps
                           for state in checkpoint['optimizer']['state'].values())
                assert (Path(record['directory']) / 'model.pt').exists()
            front_ids = {row['trial'] for row in result['pareto_front']}
            for row in result['records']:
                assert ('test' in row) == (row['arm'] != 'hpo' or row['trial'] in front_ids)
                assert len(json.loads((Path(row['directory']) / 'history.json').read_text())) == 2
                assert row['selection']['total_boxes'] == 8
        assert trained_methods == {'IBP', 'CROWN-IBP', 'CROWN'}
        with contextlib.redirect_stdout(io.StringIO()):
            main(['--summarize', str(root)])
        report = json.loads((root / 'comparison/results.json').read_text())
        assert len(report['records']) == 12
        assert all('test' in row for row in report['pareto_front'])
        for name in ('all_trials.csv', 'pareto_front.csv', 'pareto_front.png', 'pareto_front.pdf',
                     'trajectory.png', 'trajectory.pdf', 'summary.md'):
            assert (root / 'comparison' / name).stat().st_size > 0
        ce_root = root / 'ce_studies'
        for seed in (0, 1, 2):
            output = ce_root / f'seed_{seed}'
            with contextlib.redirect_stdout(io.StringIO()):
                main(['--config', str(config), '--output', str(output), '--seed', str(seed),
                      '--smoke', '--device', 'cpu', '--trials', '2', '--sampler', 'nsgaii', '--ce-only'])
            result = json.loads((output / 'results.json').read_text())
            assert len(result['records']) == 3
            assert {row['arm'] for row in result['records']} == {'ce_only', 'ce_hpo'}
            study = optuna.load_study(study_name='mnist_addition', storage=f'sqlite:///{output}/optuna_study.db')
            assert all(set(trial.params) == {'lr'} for trial in study.trials)
            for row in result['records']:
                checkpoint = torch.load(Path(row['directory']) / 'checkpoints/2_checkpoint.pt', weights_only=False)
                assert checkpoint['configuration']['property_weight'] == 0.
                assert all(entry['property_loss'] == 0. for entry in checkpoint['history'])
                assert all(float(state['step']) == 4 for state in checkpoint['optimizer']['state'].values())
            # CE HPO adds no new baseline training semantics.
            original = json.loads((root / f'seed_{seed}/ce_only/results.json').read_text())
            repeated = json.loads((output / 'ce_only/results.json').read_text())
            assert repeated['selection'] == original['selection']
        from matplotlib.figure import Figure
        savefig, plots = Figure.savefig, {}

        def capture_plot(figure, path, *args, **kwargs):
            if Path(path).suffix == '.png':
                plots[Path(path).stem] = [(axis.get_title(), [(collection.get_label(), len(collection.get_offsets()))
                                                             for collection in axis.collections]) for axis in figure.axes]
            return savefig(figure, path, *args, **kwargs)

        with patch.object(Figure, 'savefig', capture_plot), contextlib.redirect_stdout(io.StringIO()):
            main(['--summarize', str(root), '--ce-study', str(ce_root)])
        paired = json.loads((ce_root / 'comparison/results.json').read_text())
        assert len(paired['records']) == 18
        assert paired['ce_pareto_front'] and all(row['arm'] == 'ce_hpo' for row in paired['ce_pareto_front'])
        assert (ce_root / 'comparison/ce_pareto_front.csv').stat().st_size > 0
        assert set(paired['ce_tuned']) == {'clean', 'consistency', 'balanced'}
        assert not paired['partial']
        assert plots.keys() == {'pareto_front', 'trajectory'}
        for panels in plots.values():
            assert panels[1][0] == 'Test performance of validation-selected models'
        assert all(label.endswith('front') for _, series in plots['pareto_front'] for label, _ in series)
        front_count = sum(len(paired[key]) for key in ('pareto_front', 'ce_pareto_front', 'combined_pareto_front'))
        assert sum(count for _, count in plots['pareto_front'][0][1]) == front_count
        assert sum(count for _, count in plots['trajectory'][0][1]) == len(paired['records']) + front_count
        assert paired['combined_selection_hypervolume'] >= max(paired['selection_hypervolume'][arm]
                                                               for arm in ('hpo', 'ce_hpo'))
        # Reuse trained fixtures, including a seed with no finished trials and absent test scores.
        running = root / 'running_certified'
        for seed, count in enumerate((1, 0, 2)):
            directory = running / f'seed_{seed}'
            directory.mkdir(parents=True)
            for name in ('config.json', 'initial_model.pt'):
                (directory / name).write_bytes((root / f'seed_{seed}' / name).read_bytes())
            rows = json.loads((root / f'seed_{seed}/all_trials.json').read_text())
            rows = [row for row in rows if row['arm'] != 'hpo' or row['trial'] < count]
            for row in rows:
                row.pop('test', None)
            if seed == 0:
                trial = next(row for row in rows if row['arm'] == 'hpo')
                trial['selection']['metrics']['triple_accuracy'] = .99
                trial['selection']['certified_fraction'] = 1.
            (directory / 'all_trials.json').write_text(json.dumps(rows))
        for extra in ([], ['--allow-partial']):
            try:
                main(['--summarize', str(running), '--ce-study', str(ce_root), *extra])
            except ValueError:
                pass
            else:
                raise AssertionError('Incomplete studies or a missing snapshot destination were accepted')
        snapshot = root / 'interim'
        command = ['--summarize', str(running), '--ce-study', str(ce_root),
                   '--allow-partial', '--output', str(snapshot)]
        with contextlib.redirect_stdout(io.StringIO()):
            main(command)
        interim = json.loads((snapshot / 'results.json').read_text())
        assert interim['partial']
        assert [row['completed_trials'] for row in interim['study_progress']['hpo']] == [1, 0, 2]
        assert all(not row['finished'] for row in interim['study_progress']['hpo'])
        assert all(row['finished'] for row in interim['study_progress']['ce_hpo'])
        assert interim['test_hypervolume_of_selection_front'] is None
        assert interim['combined_test_hypervolume_of_selection_front'] is None
        assert all('test' in row and str(ce_root) in row['directory']
                   for row in interim['records'] if row['arm'] == 'ce_only')
        assert any(row['arm'] == 'hpo' and row['seed'] == 0 for row in interim['combined_pareto_front'])
        assert all(row['arm'] in ('hpo', 'ce_hpo') for row in interim['combined_pareto_front'])
        for name in ('combined_pareto_front.csv', 'pareto_front.png', 'pareto_front.pdf',
                     'trajectory.png', 'trajectory.pdf', 'summary.md'):
            assert (snapshot / name).stat().st_size > 0
        assert 'Interim snapshot' in (snapshot / 'summary.md').read_text()
        assert '— | —' in (snapshot / 'summary.md').read_text()
        try:
            main(command)
        except FileExistsError:
            pass
        else:
            raise AssertionError('An existing snapshot was overwritten')
        # Comparison must reject an unpaired initialization, even if metrics look plausible.
        initial_path = ce_root / 'seed_0/initial_model.pt'
        initial = torch.load(initial_path, weights_only=True)
        next(iter(initial.values())).add_(1.)
        torch.save(initial, initial_path)
        try:
            main(['--summarize', str(root), '--ce-study', str(ce_root)])
        except ValueError as error:
            assert 'initial weights' in str(error)
        else:
            raise AssertionError('Different CE initializations were pooled')
        try:
            main(['--summarize', str(root)])
        except FileExistsError:
            pass
        else:
            raise AssertionError('An existing comparison was overwritten')
        try:
            main(['--config', str(config), '--output', str(root / 'seed_0'), '--smoke', '--device', 'cpu'])
        except FileExistsError:
            pass
        else:
            raise AssertionError('An existing study was overwritten')
        changed = json.loads((root / 'seed_1/config.json').read_text())
        changed['parameters']['epsilon'] = .03
        (root / 'seed_1/config.json').write_text(json.dumps(changed))
        try:
            main(['--summarize', str(root)])
        except ValueError as error:
            assert 'epsilon' in str(error)
        else:
            raise AssertionError('Incompatible experiments were pooled')

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        config = root / 'config.json'
        config.write_text(json.dumps(dict(epochs=2, train_pairs=60000, batch_size=9,
                                         property_batch_size=3, property_boxes=7)))
        with contextlib.redirect_stdout(io.StringIO()):
            main(['--config', str(config), '--output', str(root / 'swap'), '--smoke',
                  '--property', 'swap', '--method', 'IBP', '--sampler', 'nsgaii', '--trials', '1'])
        saved = json.loads((root / 'swap/config.json').read_text())
        assert saved['parameters']['epsilon'] == 0
        assert saved['data']['train']['property_pair_count'] == 7
        result = json.loads((root / 'swap/results.json').read_text())
        for record in result['records']:
            assert record['parameters']['bound_method'] == 'IBP'
            assert 'empirical_swap_equivariant_fraction' in record['selection']['metrics']
            training = json.loads((Path(record['directory']) / 'training_config.json').read_text())
            assert training['property_dataset']['kind'] == 'mnist_pair_points'
            assert training['property_dataset']['count'] == 7

    # Cross the actual ten-trial startup boundary without another training campaign.
    from optuna_integration import BoTorchSampler
    sampler = BoTorchSampler(seed=0, n_startup_trials=10, device=torch.device('cpu'),
                             constraints_func=lambda trial: trial.user_attrs['constraints'])
    study = optuna.create_study(directions=['maximize', 'maximize'], sampler=sampler)

    def synthetic_objective(trial):
        x, y = trial.suggest_float('x', 0., 1.), trial.suggest_float('y', 0., 1.)
        trial.suggest_int('warm_up_epochs', 0, 1)
        trial.suggest_int('weight_ramp_epochs', 0, 1)
        trial.suggest_categorical('bound_method', ['IBP', 'CROWN-IBP', 'CROWN'])
        values = (x, 1. - y)
        trial.set_user_attr('constraints', [-value for value in values])
        return values

    study.optimize(synthetic_objective, n_trials=11)
    assert sampler._candidates_func.__name__ == 'qehvi_candidates_func'
    assert all(trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials)
    print('MNIST HPO data isolation, option propagation, selection and pooled comparison checks passed.')


if __name__ == '__main__':
    main_test()
