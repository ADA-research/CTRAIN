"""Adapt the ICML Pareto-search protocol to MNIST arithmetic or swap properties."""
import argparse
import csv
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

import auto_LiRPA
import optuna
import torch
import torchvision
from CTRAIN.model_definitions import MNISTAdditionClassifier
from papers.rethinking_evaluation_paradigms.eval.util import hypervolume_2d, pareto_front
from case_studies.common import PROPERTY_METHOD_SEARCH_SPACE, sample_training_options, source_provenance
from case_studies.mnist_addition.run import (certify, make_wrapper, mnist_source, parse_args,
                            specifications, task_metrics, train_trial)
from case_studies.mnist_addition.data import paired_mnist_dataset


DATA_SEED = 42  # All study seeds see the same holdout and evaluation pairs.
PAPER = 'https://proceedings.mlr.press/v306/kaulen26a.html'


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def objectives(record, split='selection'):
    result = record[split]
    return result['metrics']['triple_accuracy'], result['certified_fraction']


def comparison(records, hpo_arm='hpo'):
    """Select using HPO metrics only, including when test rankings differ."""
    trials = [row for row in records if row['arm'] == hpo_arm]
    front = pareto_front(trials, objectives)
    selected = {}
    if front:
        for name, score in (
                ('clean', lambda row: objectives(row)),
                ('consistency', lambda row: tuple(reversed(objectives(row)))),
                ('balanced', lambda row: (min(objectives(row)), sum(objectives(row))))):
            row = max(front, key=score)
            selected[name] = dict(seed=row['seed'], trial=row['trial'])
    groups = {arm: [row for row in records if row['arm'] == arm]
              for arm in ('ce_only', 'fixed_certified', 'hpo', 'ce_hpo')}
    return dict(pareto_front=front, tuned=selected, reference_point=[0., 0.],
                selection_hypervolume={arm: hypervolume_2d(rows, objectives)
                                       for arm, rows in groups.items()},
                test_hypervolume_of_selection_front=hypervolume_2d(
                    front, lambda row: objectives(row, 'test'))
                if front and all('test' in row for row in front) else None,
                selection_rule='Pareto filtering and tuned representatives use selection metrics only; '
                               'test metrics never change membership or representative choices')


def load_data(args, selection_split, eval_pairs):
    """Split source images before pairing, so neither operand leaks into holdout."""
    pools, raw, provenance = {}, {}, {}
    for original_split in ('train', 'test'):
        images, labels = mnist_source(args, original_split == 'train',
                                     DATA_SEED + (original_split == 'test'),
                                     32 if original_split == 'train' else 16)
        source_hashes = dict(images_sha256=hashlib.sha256(images.numpy().tobytes()).hexdigest(),
                             labels_sha256=hashlib.sha256(labels.numpy().tobytes()).hexdigest())
        source_indices = torch.arange(len(labels))
        groups = [(original_split, source_indices)]
        if original_split == 'train' and selection_split == 'holdout':
            shuffled = torch.randperm(len(labels), generator=torch.Generator().manual_seed(DATA_SEED))
            holdout_size = 8 if args.smoke else 6000
            groups = [('train', shuffled[holdout_size:]), ('selection', shuffled[:holdout_size])]
        for split, subset in groups:
            count = args.train_pairs if split == 'train' else eval_pairs
            pools[split], indices = paired_mnist_dataset(
                images[subset], labels[subset], count,
                args.seed if split == 'train' else DATA_SEED + 1,
                cover_all=split == 'train')
            domain_count = 1 if split == 'train' else count
            raw[split] = (pools[split].tensors[0][:domain_count] if args.property == 'swap'
                          else images[subset[indices[:domain_count]]].float() / 255)
            provenance[split] = dict(original_split=original_split, source_indices=subset.tolist(),
                pair_indices=subset[indices].tolist(), **source_hashes,
                unique_images_per_position=[int(indices[:, side].unique().numel()) for side in range(2)])
    if selection_split == 'test':
        pools['selection'], raw['selection'], provenance['selection'] = (
            pools['test'], raw['test'], provenance['test'])
    return pools, raw, provenance


def evaluate(wrapper, dataset, spec, args, directory, split):
    wrapper.original_model.eval()
    metrics = task_metrics(wrapper.original_model, dataset, args.device, args.batch_size,
                           swap=args.property == 'swap')
    report = certify(wrapper, spec, args.eval_method)
    write_json(directory / f'{split}_verification.json', report)
    return dict(metrics=metrics, certified_fraction=report['certified_fraction'],
                certified_count=report['certified_count'], total_boxes=report['total_boxes'])


def run_study(args, options, search_space):
    if args.output.exists():
        raise FileExistsError('output must be a fresh directory')
    torch.set_num_threads(1)
    hpo_arm = 'ce_hpo' if options.ce_only else 'hpo'
    pools, raw, provenance = load_data(args, options.selection_split, options.eval_pairs)
    args.output.mkdir(parents=True, exist_ok=False)
    specs = {}
    for split in ('train', 'selection', 'test'):
        if split == 'test' and options.selection_split == 'test':
            specs[split] = specs['selection']
            continue
        specs[split] = specifications(raw[split], args.epsilon, property_kind=args.property)
        provenance[split]['property_pair_count'] = (args.property_boxes or len(pools[split])) if split == 'train' else len(pools[split])
        provenance[split]['specification_role'] = ('formula template; selected training domains generated on demand'
                                                  if split == 'train' else 'evaluation boxes in memory')
    torch.manual_seed(args.seed)
    initial = MNISTAdditionClassifier(args.width).state_dict()
    torch.save(initial, args.output / 'initial_model.pt')
    source = source_provenance()
    root = Path(__file__).resolve().parents[2]
    for relative in ('case_studies/mnist_addition/hpo.py',
                     'papers/rethinking_evaluation_paradigms/eval/util.py'):
        source['sha256'][relative] = hashlib.sha256((root / relative).read_bytes()).hexdigest()
    config = dict(parameters={key: str(value) if isinstance(value, Path) else value
                              for key, value in vars(args).items()},
                  selection_split=options.selection_split, eval_pairs=options.eval_pairs,
                  trials=options.trials, sampler=options.sampler, search_space=search_space,
                  training_mode='ce_only' if options.ce_only else 'certified',
                  data_seed=DATA_SEED, data=provenance, source=source,
                  paper=PAPER, objectives=['clean triple accuracy', 'certified swap-equivariance fraction'
                                          if args.property == 'swap' else 'certified arithmetic-consistency fraction'],
                  scope='floating-point affine LiRPA bounds; swap certificates cover exact evaluated pairs'
                        if args.property == 'swap' else
                        'floating-point affine LiRPA bounds; consistency is not recognition robustness',
                  epoch_selection='final epoch; no validation-selected weights',
                  training_properties=f'lazy domains for {provenance["train"]["property_pair_count"]} training pairs; uniform sampling with replacement',
                  batch_sampling='separate task/property batch sizes; random property boxes with replacement per task update',
                  adaptation=f'{len(search_space)}-parameter search; installed BoTorch GP defaults; '
                             'no complete verification or paper score reproduction',
                  environment=dict(python=platform.python_version(), torch=torch.__version__,
                      torchvision=torchvision.__version__, auto_lirpa=auto_LiRPA.__version__, optuna=optuna.__version__))
    if options.sampler == 'botorch':
        import botorch
        import optuna_integration
        config['environment'].update(botorch=botorch.__version__,
                                     optuna_integration=optuna_integration.__version__)
    write_json(args.output / 'config.json', config)
    records = []

    def train(parameters, arm, number=None):
        directory = args.output / (arm if number is None else f'trials/{number:03d}')
        directory.mkdir(parents=True, exist_ok=False)
        wrapper, training_seconds = train_trial(args, pools['train'], specs['train'],
                                               parameters, initial, directory,
                                               metric_dataset=pools['selection'], metric_split='selection')
        selection = evaluate(wrapper, pools['selection'], specs['selection'], args, directory, 'selection')
        torch.save({key: value.detach().cpu() for key, value in wrapper.original_model.state_dict().items()},
                   directory / 'model.pt')
        write_json(directory / 'history.json', wrapper.history)
        write_json(directory / 'training_config.json', wrapper.configuration)
        record = dict(arm=arm, seed=args.seed, trial=number, parameters=parameters,
                      directory=str(directory.resolve()), training_seconds=training_seconds, selection=selection)
        write_json(directory / 'results.json', record)
        records.append(record)
        write_json(args.output / 'all_trials.json', records)
        return record

    defaults = {name: getattr(args, name) for name in
                ('lr', 'property_weight', 'warm_up_epochs', 'weight_ramp_epochs', 'batch_size', 'property_batch_size')}
    defaults['bound_method'] = args.method
    defaults['training_options'] = copy.deepcopy(args.training_options)
    train(dict(defaults, property_weight=0.), 'ce_only')
    if not options.ce_only:
        train(defaults, 'fixed_certified')

    def objective(trial):
        parameters = dict(defaults, lr=trial.suggest_float('lr', *search_space['lr'], log=True))
        if options.ce_only:
            parameters['property_weight'] = 0.
        else:
            parameters.update(
                property_weight=trial.suggest_float('property_weight', *search_space['property_weight'], log=True),
                warm_up_epochs=trial.suggest_int('warm_up_epochs', *search_space['warm_up_epochs']),
                weight_ramp_epochs=trial.suggest_int('weight_ramp_epochs', *search_space['weight_ramp_epochs']),
                bound_method=trial.suggest_categorical('bound_method', search_space['bound_method']))
            parameters['training_options'] = sample_training_options(
                trial, parameters['bound_method'], args.training_options)
        record = train(parameters, hpo_arm, trial.number)
        values = objectives(record)
        trial.set_user_attr('constraints', [-value for value in values])
        trial.set_user_attr('checkpoint_path', str(Path(record['directory']) / 'model.pt'))
        trial.set_user_attr('metrics', record['selection'])
        return values

    def constraints(trial):
        return trial.user_attrs['constraints']

    if options.sampler == 'botorch':
        from optuna_integration import BoTorchSampler
        sampler = BoTorchSampler(seed=args.seed, n_startup_trials=10, constraints_func=constraints,
                                 device=torch.device('cpu'))
    else:
        sampler = optuna.samplers.NSGAIISampler(seed=args.seed, constraints_func=constraints)
    study = optuna.create_study(study_name='mnist_addition', directions=['maximize', 'maximize'],
                               sampler=sampler, storage=f'sqlite:///{args.output.resolve()}/optuna_study.db')
    # Baselines are separate: the paper's search starts without known configurations.
    study.optimize(objective, n_trials=options.trials, gc_after_trial=True)
    front = comparison(records, hpo_arm)['pareto_front']
    for record in [row for row in records if row['arm'] != hpo_arm] + front:
        directory = Path(record['directory'])
        if options.selection_split == 'test':
            record['test'] = record['selection']
        else:
            state = torch.load(directory / 'model.pt', map_location='cpu', weights_only=True)
            wrapper = make_wrapper(args, specs['train'], record['parameters'], state)
            record['test'] = evaluate(wrapper, pools['test'], specs['test'], args, directory, 'test')
        write_json(directory / 'results.json', record)
    summary = dict(seed=args.seed, smoke=args.smoke, selection_split=options.selection_split,
                   completed_trials=len(study.trials), records=records, **comparison(records, hpo_arm))
    write_json(args.output / 'results.json', summary)
    write_json(args.output / 'all_trials.json', records)
    print(json.dumps({key: value for key, value in summary.items() if key not in ('records', 'pareto_front')}, indent=2))


def load_studies(root, allow_partial=False):
    records, configurations, progress = [], [], []
    for seed in (0, 1, 2):
        directory = root / f'seed_{seed}'
        config = json.loads((directory / 'config.json').read_text())
        summary_path = directory / 'results.json'
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else None
        if summary is None and not allow_partial:
            raise ValueError('Expected three complete studies with seeds 0, 1 and 2')
        if summary is not None:
            if summary['seed'] != seed or summary['completed_trials'] != config['trials']:
                raise ValueError('Expected three complete studies with seeds 0, 1 and 2')
            rows = summary['records']
        else:
            path = directory / 'all_trials.json'
            rows = json.loads(path.read_text()) if path.exists() else []
        arm = 'ce_hpo' if config.get('training_mode') == 'ce_only' else 'hpo'
        trials = [row['trial'] for row in rows if row['arm'] == arm]
        if (any(row['seed'] != seed for row in rows) or len(set(trials)) != len(trials)
                or any(number not in range(config['trials']) for number in trials)
                or (summary is not None and len(trials) != config['trials'])):
            raise ValueError(f'Invalid trial records for seed {seed}')
        progress.append(dict(seed=seed, completed_trials=len(trials), trial_budget=config['trials'],
                             finished=summary is not None))
        configurations.append(config)
        records.extend(rows)
    # Different seeds must share the evaluation experiment before pooling scores.
    for config in configurations[1:]:
        for key in ('selection_split', 'eval_pairs', 'trials', 'sampler', 'search_space', 'source'):
            if config[key] != configurations[0][key]:
                raise ValueError(f'Studies disagree on {key}')
        for split in ('selection', 'test'):
            if config['data'][split] != configurations[0]['data'][split]:
                raise ValueError(f'Studies use different {split} data')
        for key, value in config['parameters'].items():
            if key not in ('seed', 'output') and value != configurations[0]['parameters'][key]:
                raise ValueError(f'Studies disagree on parameter {key}')
    return records, configurations, progress


def export_comparison(root, ce_study=None, *, allow_partial=False, output=None):
    """Pool selection scores, optionally taking a snapshot of unfinished studies."""
    if allow_partial and output is None:
        raise ValueError('Partial comparisons require a fresh --output directory')
    records, configurations, progress = load_studies(root, allow_partial)
    hpo_arm = 'ce_hpo' if configurations[0].get('training_mode') == 'ce_only' else 'hpo'
    study_progress = {hpo_arm: progress}
    study_roots = {hpo_arm: str(root.resolve())}
    if ce_study is not None:
        ce_records, ce_configs, ce_progress = load_studies(ce_study, allow_partial)
        study_progress['ce_hpo'] = ce_progress
        study_roots['ce_hpo'] = str(ce_study.resolve())
        for seed, (config, ce_config) in enumerate(zip(configurations, ce_configs)):
            if ce_config['training_mode'] != 'ce_only':
                raise ValueError('Expected a CE-only comparison study')
            for key in ('selection_split', 'eval_pairs', 'trials', 'sampler', 'data_seed', 'environment'):
                if config[key] != ce_config[key]:
                    raise ValueError(f'CE study differs on {key}')
            for key in config['parameters']:
                if key not in ('config', 'output') and config['parameters'][key] != ce_config['parameters'][key]:
                    raise ValueError(f'CE study differs on parameter {key}')
            if config['search_space']['lr'] != ce_config['search_space']['lr']:
                raise ValueError('CE study uses a different learning-rate range')
            for split in ('train', 'selection', 'test'):
                left = {k: v for k, v in config['data'][split].items() if k != 'vnnlib_sha256'}
                right = {k: v for k, v in ce_config['data'][split].items() if k != 'vnnlib_sha256'}
                if left != right:
                    raise ValueError(f'CE study uses different {split} data')
            ignored = 'case_studies/mnist_addition/hpo.py'
            if ({k: v for k, v in config['source']['sha256'].items() if k != ignored}
                    != {k: v for k, v in ce_config['source']['sha256'].items() if k != ignored}):
                raise ValueError('CE study uses different training or evaluation source')
            initial = torch.load(root / f'seed_{seed}/initial_model.pt', map_location='cpu', weights_only=True)
            ce_initial = torch.load(ce_study / f'seed_{seed}/initial_model.pt', map_location='cpu', weights_only=True)
            if initial.keys() != ce_initial.keys() or any(not torch.equal(value, ce_initial[key]) for key, value in initial.items()):
                raise ValueError('CE study uses different initial weights')
        # Prefer the CE study's baseline record when it already includes test evaluation.
        ce_baselines = {row['seed']: row for row in ce_records if row['arm'] == 'ce_only' and 'test' in row}
        records = [ce_baselines.get(row['seed'], row) if row['arm'] == 'ce_only' else row for row in records]
        records.extend(row for row in ce_records if row['arm'] == 'ce_hpo')
    result = dict(seeds=[0, 1, 2], records=records, smoke=configurations[0]['parameters']['smoke'],
                  created_at=datetime.now(timezone.utc).isoformat(), study_roots=study_roots, study_progress=study_progress,
                  partial=any(not seed['finished'] for group in study_progress.values() for seed in group),
                  selection_split=configurations[0]['selection_split'], **comparison(records, hpo_arm))
    if ce_study is not None:
        ce_report = comparison(records, 'ce_hpo')
        result.update(ce_pareto_front=ce_report['pareto_front'], ce_tuned=ce_report['tuned'],
                      ce_test_hypervolume_of_selection_front=ce_report['test_hypervolume_of_selection_front'])
    merged = pareto_front([row for row in records if row['arm'] in ('hpo', 'ce_hpo')], objectives)
    result.update(combined_pareto_front=merged, combined_selection_hypervolume=hypervolume_2d(merged, objectives),
                  combined_test_hypervolume_of_selection_front=hypervolume_2d(merged, lambda row: objectives(row, 'test'))
                  if merged and all('test' in row for row in merged) else None)
    destination = output if output is not None else (ce_study or root) / 'comparison'
    destination.mkdir(exist_ok=False)
    write_json(destination / 'results.json', result)
    fields = ['arm', 'seed', 'trial', 'lr', 'property_weight', 'warm_up_epochs', 'weight_ramp_epochs',
              'bound_method', 'training_options', 'batch_size', 'property_batch_size',
              'selection_triple_accuracy', 'selection_certified_fraction',
              'test_triple_accuracy', 'test_certified_fraction', 'training_seconds', 'checkpoint']
    exports = [('all_trials.csv', records), ('pareto_front.csv', result['pareto_front']),
               ('combined_pareto_front.csv', merged)]
    if ce_study is not None:
        exports.append(('ce_pareto_front.csv', result['ce_pareto_front']))
    for filename, rows in exports:
        with (destination / filename).open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                parameters = dict(row['parameters'], training_options=json.dumps(row['parameters'].get('training_options', {})))
                writer.writerow(dict(arm=row['arm'], seed=row['seed'], trial=row['trial'], **parameters,
                    selection_triple_accuracy=row['selection']['metrics']['triple_accuracy'], selection_certified_fraction=objectives(row)[1],
                    test_triple_accuracy=row['test']['metrics']['triple_accuracy'] if 'test' in row else '',
                    test_certified_fraction=objectives(row, 'test')[1] if 'test' in row else '',
                    training_seconds=row['training_seconds'], checkpoint=str(Path(row['directory']) / 'model.pt')))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels = {'hpo': 'Certified HPO', 'ce_hpo': 'CE-only HPO'}
    counts = [f"{labels[arm]}: {sum(seed['completed_trials'] for seed in group)}/"
              f"{sum(seed['trial_budget'] for seed in group)} trials" for arm, group in study_progress.items()]
    for filename, show_trials in (('pareto_front', False), ('trajectory', True)):
        figure, axes = plt.subplots(1, 2, figsize=(13, 5.5))
        figure.suptitle(('Interim comparison' if result['partial'] else 'Completed comparison') + '\n' + '; '.join(counts))
        for axis, split in zip(axes, ('selection', 'test')):
            if show_trials:
                for arm, label, marker, color in (('hpo', 'Certified HPO trials', '.', 'tab:orange'),
                                                  ('ce_only', 'CE baseline', 's', 'tab:green'),
                                                  ('ce_hpo', 'CE-only HPO trials', 'x', 'tab:blue'),
                                                  ('fixed_certified', 'Fixed certified baseline', '^', 'tab:purple')):
                    rows = [row for row in records if row['arm'] == arm and split in row]
                    if not rows:
                        continue
                    axis.scatter([objectives(row, split)[0] for row in rows],
                                 [objectives(row, split)[1] for row in rows], label=label, marker=marker, color=color, alpha=.65)
            for key, color, label in (('pareto_front', 'tab:orange' if hpo_arm == 'hpo' else 'tab:blue',
                                      f'{labels[hpo_arm]} selection front'),
                                      ('ce_pareto_front', 'tab:blue', 'CE-only selection front'),
                                      ('combined_pareto_front', 'black', 'Merged selection front')):
                front = sorted([row for row in result.get(key, []) if split in row], key=lambda row: objectives(row, split))
                if front:
                    axis.scatter([objectives(row, split)[0] for row in front],
                                 [objectives(row, split)[1] for row in front], facecolors='none', edgecolors=color, label=label)
                    if split == 'selection':
                        axis.plot([objectives(row)[0] for row in front], [objectives(row)[1] for row in front], color=color, alpha=.5)
            axis.set(xlabel='Clean exp(-task loss)', ylabel='Certified arithmetic-consistency fraction',
                     title='HPO selection' if split == 'selection' else 'Test performance of validation-selected models',
                     xlim=(0, 1), ylim=(0, 1))
            axis.legend(fontsize=8)
            axis.grid(alpha=.2)
            if split == 'test':
                missing = sum('test' not in row for row in merged)
                if missing:
                    axis.text(.02, .03, f'{missing}/{len(merged)} merged-front models await test evaluation',
                              transform=axis.transAxes, fontsize=8)
        figure.tight_layout()
        figure.savefig(destination / f'{filename}.png', dpi=180)
        figure.savefig(destination / f'{filename}.pdf')
        plt.close(figure)
    lines = ['# MNIST addition: baselines and multi-objective tuning', '',
             f"Selection protocol: {result['selection_split']}. Synthetic smoke: {result['smoke']}.", '',
             ('Interim snapshot: unequal completed search budgets; fronts may change.' if result['partial']
              else 'All studies completed.'), '', *counts, '',
             'Pareto membership and tuned representatives use selection scores only. '
             'Certification proves arithmetic agreement, not correct recognition. No complete verifier was run.', '',
             '| Model | Seed | Trial | Selection accuracy | Selection consistency | Test accuracy | Test consistency |',
             '|---|---:|---:|---:|---:|---:|---:|']
    chosen = [row for row in records if row['arm'] not in ('hpo', 'ce_hpo')]
    for tuned_key, front_key, prefix in (('tuned', 'pareto_front', 'tuned'), ('ce_tuned', 'ce_pareto_front', 'ce_tuned')):
        for name, identity in result.get(tuned_key, {}).items():
            row = next(row for row in result[front_key] if
                       row['seed'] == identity['seed'] and row['trial'] == identity['trial'])
            chosen.append(dict(row, arm=f'{prefix}_{name}'))
    for row in chosen:
        lines.append(f"| {row['arm']} | {row['seed']} | {row['trial']} | "
                     + ' | '.join(f'{value:.4f}' for value in objectives(row)) + ' | '
                     + (' | '.join(f'{value:.4f}' for value in objectives(row, 'test')) if 'test' in row else '— | —') + ' |')
    lines += ['', 'Selection hypervolume (reference [0, 0]):', '']
    lines += [f'- {arm}: {value:.6f}' for arm, value in result['selection_hypervolume'].items()]
    lines += [f"- Merged tuned front: {result['combined_selection_hypervolume']:.6f}", '',
              f"Merged selection Pareto points: {len(merged)}.", '',
              'The pooled front is a union of independently seeded searches, not an uncertainty estimate. '
              'Test HPO points are evaluated only after selection; missing test scores are left blank. '
              'GPU determinism and complete certification are unverified.', '']
    (destination / 'summary.md').write_text('\n'.join(lines))
    print(f'Wrote {destination}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('config.json'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--trials', type=int, default=100)
    parser.add_argument('--eval-pairs', type=int, default=1000)
    parser.add_argument('--selection-split', choices=('holdout', 'test'), default='holdout')
    parser.add_argument('--sampler', choices=('botorch', 'nsgaii'), default='botorch')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, help='Task batch size')
    parser.add_argument('--property-batch-size', type=int, help='Property domains per task update')
    parser.add_argument('--property', choices=('arithmetic', 'swap'), help='Override the property in the config')
    from CTRAIN.properties.training import TRAINING_METHODS
    parser.add_argument('--method', choices=TRAINING_METHODS)
    parser.add_argument('--training-options', type=json.loads)
    parser.add_argument('--data-root', type=Path, help='Override the cached MNIST location in the config')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--ce-only', action='store_true', help='Tune learning rate with a zero property penalty')
    parser.add_argument('--summarize', type=Path, help='Combine completed seed_0, seed_1 and seed_2 directories')
    parser.add_argument('--ce-study', type=Path, help='Include a matched CE-only study in --summarize')
    parser.add_argument('--allow-partial', action='store_true', help='Snapshot finished trials from running studies; requires --summarize and a fresh --output')
    options = parser.parse_args(argv)
    if options.summarize is not None:
        export_comparison(options.summarize, options.ce_study, allow_partial=options.allow_partial, output=options.output)
        return
    if options.ce_study is not None:
        parser.error('--ce-study requires --summarize')
    if options.allow_partial:
        parser.error('--allow-partial requires --summarize')
    if options.output is None or options.trials < 1 or options.eval_pairs < 1:
        parser.error('--output is required; trials and eval-pairs must be positive')
    baseline_argv = ['--config', str(options.config), '--output', str(options.output),
                     '--seed', str(options.seed), '--device', options.device]
    if options.method is not None:
        baseline_argv.extend(['--method', options.method])
    if options.training_options is not None:
        baseline_argv.extend(['--training-options', json.dumps(options.training_options)])
    if options.batch_size is not None:
        baseline_argv.extend(['--batch-size', str(options.batch_size)])
    if options.property_batch_size is not None:
        baseline_argv.extend(['--property-batch-size', str(options.property_batch_size)])
    if options.property is not None:
        baseline_argv.extend(['--property', options.property])
    if options.data_root is not None:
        baseline_argv.extend(['--data-root', str(options.data_root)])
    if options.smoke:
        baseline_argv.append('--smoke')
        options.eval_pairs = 8
    args = parse_args(baseline_argv)
    if args.task != 'classification':
        parser.error('HPO requires classification')
    args.test_pairs = options.eval_pairs
    warm_max = min(5, args.epochs - 1)
    ramp_max = min(args.epochs if args.smoke else args.epochs // 2, args.epochs - warm_max)
    if args.smoke:
        args.warm_up_epochs = min(args.warm_up_epochs, warm_max)
        args.weight_ramp_epochs = min(args.weight_ramp_epochs, ramp_max)
    search_space = dict(lr=[1e-4, 1e-2], property_weight=[1e-3, 100.],
                        warm_up_epochs=[0, warm_max], weight_ramp_epochs=[0, ramp_max],
                        bound_method=['IBP', 'CROWN-IBP', 'CROWN'])
    if not (search_space['lr'][0] <= args.lr <= search_space['lr'][1]
            and search_space['property_weight'][0] <= args.property_weight <= search_space['property_weight'][1]
            and args.warm_up_epochs <= search_space['warm_up_epochs'][1]
            and args.weight_ramp_epochs <= search_space['weight_ramp_epochs'][1]):
        parser.error('Fixed baseline parameters must lie inside the declared search space')
    if options.method is not None or args.training_options or args.method not in ('IBP', 'CROWN-IBP', 'CROWN'):
        search_space['bound_method'] = [args.method]
    search_space.update(PROPERTY_METHOD_SEARCH_SPACE.get(args.method, {}))
    if options.ce_only:
        search_space = {'lr': search_space['lr']}
    if options.dry_run:
        print(json.dumps(dict(parameters={key: str(value) if isinstance(value, Path) else value
                                           for key, value in vars(args).items()},
                              trials=options.trials, sampler=options.sampler, eval_pairs=options.eval_pairs,
                              selection_split=options.selection_split, search_space=search_space,
                              training_mode='ce_only' if options.ce_only else 'certified',
                              paper=PAPER, data_seed=DATA_SEED), indent=2))
        return
    run_study(args, options, search_space)


if __name__ == '__main__':
    main()
