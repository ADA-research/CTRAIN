"""Evaluate every assigned ACAS obligation, tune repairs, and verify selected weights."""
import argparse
import contextlib
import copy
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import time

import optuna
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from CTRAIN.model_wrappers.certified import PropertyModelWrapper
from CTRAIN.properties.training import TRAINING_METHODS, resolve_training_options
from CTRAIN.bound.properties import PropertyBounds
from CTRAIN.data_loaders import load_vnnlib_problem
from CTRAIN.eval.properties import (evaluate_cells, export_property_graph,
                                    find_counterexample, initial_partition, refine_partition)
from CTRAIN.properties import Box, PropertySpec, load_vnnlib_property
from CTRAIN.train.certified.losses.properties import advisory_cross_entropy
from case_studies.common import PROPERTY_METHOD_SEARCH_SPACE, sample_training_options, source_provenance
from case_studies.acasxu.evaluate import sample_inputs


def write_json(path, value):
    # Readers never see an interrupted JSON write.
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def benchmark_cases(root):
    cases = {}
    with (root / 'instances.csv').open() as file:
        for row in csv.reader(file):
            if len(row) != 3 or not float(row[2]) > 0:
                raise ValueError('Invalid benchmark instance')
            model, specification = root / row[0], root / row[1]
            match = re.fullmatch(r'ACASXU_run2a_([1-5]_[1-9])_batch_2000.onnx', model.name)
            if (match is None or not model.is_file() or not specification.is_file()
                    or root.resolve() not in model.resolve().parents
                    or root.resolve() not in specification.resolve().parents):
                raise ValueError(f'Invalid or missing benchmark artifacts: {row}')
            network = match[1]
            case = cases.setdefault(network, dict(network=network, model=str(model.resolve()), specifications=[]))
            if any(entry['path'] == str(specification.resolve()) for entry in case['specifications']):
                raise ValueError('Duplicate benchmark obligation')
            case['specifications'].append(dict(path=str(specification.resolve()), timeout=float(row[2])))
    if not cases:
        raise ValueError('Empty benchmark')
    return [cases[key] for key in sorted(cases)]


def predictions(model, inputs):
    reference = next(model.parameters())
    with torch.no_grad():
        output = torch.cat([model(chunk.to(reference)).cpu() for chunk in inputs.split(1024)])
    if not torch.isfinite(output).all():
        raise FloatingPointError('Non-finite predictions')
    return output


def fidelity_inputs(specs, reference, count, seed):
    """Half operational envelope, half equally sampled assigned property boxes."""
    domains = [box for spec in specs for box in spec.domains]
    envelope = Box(torch.stack([b.lower for b in domains]).amin(0),
                   torch.stack([b.upper for b in domains]).amax(0))
    # sample_inputs only needs domains, so the formula is immaterial here.
    broad = PropertySpec('fidelity_envelope', specs[0].formula, [envelope])
    chunks = [sample_inputs(broad, reference, count // 2, seed)]
    per_box = max(1, (count - count // 2 + len(domains) - 1) // len(domains))
    regional = torch.cat([sample_inputs(spec, reference, per_box, seed + i + 1)
                          for i, spec in enumerate(specs)])[:count - count // 2]
    return torch.cat([*chunks, regional]).cpu(), envelope.to_dict()


def sampling_metrics(model, original, specs, inputs, broad):
    rows = []
    for spec, points in zip(specs, inputs):
        output, source = predictions(model, points), predictions(original, points)
        safe = spec.formula.evaluate(output).flatten()
        rows.append(dict(property=spec.name, samples=len(points), violations=int((~safe).sum()),
            satisfaction=float(safe.double().mean()),
            advisory_agreement=float((output.argmin(1) == source.argmin(1)).double().mean()),
            score_mse=float((output - source).square().mean()),
            counterexample=points[~safe][0].tolist() if not safe.all() else None))
    broad_output, broad_source = predictions(model, broad), predictions(original, broad)
    return dict(properties=rows, minimum_property_satisfaction=min(r['satisfaction'] for r in rows),
        advisory_agreement=sum(r['advisory_agreement'] for r in rows) / len(rows),
        envelope_advisory_agreement=float((broad_output.argmin(1) == broad_source.argmin(1)).double().mean()))


def objective_values(metrics):
    return (metrics['minimum_property_satisfaction'],
            (metrics['advisory_agreement'] + metrics['envelope_advisory_agreement']) / 2)


def fixed_cover(specs, cells_per_box):
    cells = initial_partition(specs)
    while True:
        rows = [dict(minimum_lower_bound=0., outcome='unknown') for _ in cells]
        children = refine_partition(cells, rows, specs, split_count=len(cells),
                                    max_cells=cells_per_box, max_depth=20)
        if len(children) == len(cells):
            return cells
        cells = children


def cover_metrics(bounds, cells, specs):
    rows, seconds = evaluate_cells(bounds, cells, 'CROWN', batch_size=32)
    per_property = []
    for p, spec in enumerate(specs):
        relevant = [(c, r) for c, r in zip(cells, rows) if c['property'] == p]
        total = sum(c['probability'] for c, _ in relevant)
        fraction = sum(c['probability'] for c, r in relevant if r['outcome'] == 'certified') / total
        per_property.append(dict(property=spec.name, certified_fraction=fraction,
                                 certified=all(r['outcome'] == 'certified' for _, r in relevant)))
    return dict(certified=all(r['certified'] for r in per_property), properties=per_property,
                cells=rows, seconds=seconds)


def verify(model, onnx_path, case, specs, metrics, directory, timeout):
    """Use original VNNLIB files; SAT without a validated witness stays unresolved."""
    from CTRAIN.complete_verification.abCROWN.util import get_abcrown_standard_conf
    from CTRAIN.complete_verification.abCROWN.verify import limited_abcrown_eval
    directory.mkdir(parents=True, exist_ok=False)
    bounds = PropertyBounds(model, tuple(specs[0].domains[0].lower.shape), specs,
                            {'conv_mode': 'matrix', 'relu': 'adaptive', 'property_bound': 'atomic'})
    root_rows, _ = evaluate_cells(bounds, initial_partition(specs), 'CROWN')
    results = []
    for p, (spec, entry, sampled) in enumerate(zip(specs, case['specifications'], metrics['properties'])):
        witness = sampled['counterexample']
        if witness is None:
            for box in spec.domains:
                witness = find_counterexample(model, spec.formula, box)
                if witness is not None:
                    break
        roots = [row for row in root_rows if row['property'] == spec.name]
        bounded = all(row['outcome'] == 'certified' for row in roots)
        if witness is not None and bounded:
            raise AssertionError('Counterexample contradicts CROWN certificate')
        result = dict(property=spec.name, specification=entry['path'],
                      outcome='violated' if witness is not None else ('certified' if bounded else 'unknown'),
                      counterexample=witness, root_bounds=roots)
        if result['outcome'] == 'unknown':
            budget = timeout if timeout is not None else entry['timeout']
            configuration = get_abcrown_standard_conf(timeout=budget, no_cores=1)
            # Use the bundled ACAS input-splitting policy with a CPU-sized batch.
            configuration['solver'].update(batch_size=256, bound_prop_method='crown')
            configuration['bab']['branching']['method'] = 'naive'
            configuration['bab']['branching']['input_split'].update(enable=True, reorder_bab=True)
            configuration['attack'].update(pgd_order='after', pgd_restarts=100)
            (directory / f'{spec.name}_solver_config.yaml').write_text(yaml.safe_dump(configuration))
            with (directory / f'{spec.name}_solver.log').open('w') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                elapsed, status = limited_abcrown_eval(config=configuration, seed=0,
                    instance=entry['path'], model_name=None, model_onnx_path=str(onnx_path.resolve()),
                    input_shape=[1, *spec.domains[0].lower.shape], timeout=budget, no_cores=1)
            result.update(solver_status=status, solver_seconds=elapsed, timeout=budget,
                          outcome='certified' if status == 'unsat' else 'unknown')
        results.append(result)
        write_json(directory / 'results.json', dict(properties=results, certified=False))
        print(f'{spec.name}: {result["outcome"]}', flush=True)
    report = dict(properties=results, certified=all(r['outcome'] == 'certified' for r in results),
                  numerical_scope='floating-point CROWN/abCROWN; no directed-rounding proof',
                  sat_policy='SAT without a validated source-model witness is unresolved')
    write_json(directory / 'results.json', report)
    return report


def run(args):
    case = next(c for c in benchmark_cases(args.benchmark) if c['network'] == args.network)
    if args.output.exists():
        raise FileExistsError('Use a fresh output directory; prior experiments are preserved')
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    original, first, provenance = load_vnnlib_problem(case['model'], case['specifications'][0]['path'])
    original.to(args.device).eval()
    shape = tuple(first.domains[0].lower.shape)
    specs = [first, *[load_vnnlib_property(e['path'], input_shape=shape) for e in case['specifications'][1:]]]
    all_specs = [load_vnnlib_property(p, input_shape=shape) for p in sorted((args.benchmark / 'vnnlib').glob('*.vnnlib'))]
    reference = next(original.parameters())
    inputs = {split: [sample_inputs(s, reference, args.samples, seed).cpu() for s in specs]
              for split, seed in [('selection', args.seed + 200), ('test', args.seed + 300)]}
    training, envelope = fidelity_inputs(all_specs, reference, args.train_samples, args.seed + 100)
    # Add assigned regions explicitly: narrow obligations must appear in fidelity training.
    assigned, _ = fidelity_inputs(specs, reference, args.train_samples, args.seed + 101)
    training = torch.cat((training[:args.train_samples // 2], assigned[args.train_samples // 2:]))
    broad = {split: fidelity_inputs(all_specs, reference, args.samples * 2, seed)[0][:args.samples]
             for split, seed in [('selection', args.seed + 400), ('test', args.seed + 500)]}
    args.output.mkdir(parents=True, exist_ok=False)
    specification_directory = args.output / 'specifications'
    specification_directory.mkdir()
    # abCROWN caches parsed VNNLIB beside its input; each job owns its copies.
    for entry in case['specifications']:
        source = Path(entry['path'])
        destination = specification_directory / source.name
        shutil.copyfile(source, destination)
        entry.update(benchmark_source=str(source), path=str(destination.resolve()))
    methods = [args.training_method] if getattr(args, 'training_method', None) else ['IBP', 'CROWN-IBP', 'CROWN']
    configuration = dict(parameters={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        case=case, artifacts=provenance, source=source_provenance(),
        benchmark_sha256=hashlib.sha256((args.benchmark / 'instances.csv').read_bytes()).hexdigest(),
        artifact_sha256={str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in [Path(case['model']), *sorted((args.benchmark / 'vnnlib').glob('*.vnnlib')),
                                  *[Path(e['path']) for e in case['specifications']]]},
        objectives=['minimum per-property sampled satisfaction', 'mean of assigned-property and envelope advisory agreement'],
        selection='complete fixed-cover certification first, then worst sampled satisfaction, then advisory agreement; test is unused',
        fidelity_envelope=envelope, fidelity_sampling='half envelope, half assigned boxes; original argmin labels',
        epoch_selection='final epoch, no validation-selected weights',
        bound_semantics='sound atomic affine LiRPA bounds with monotone Boolean reduction',
        search_space=dict(lr=[1e-5, 3e-3], property_weight=[1e-4, 100.], margin=[1e-5, 1e-2],
                          weight_ramp_epochs=[0, args.epochs // 2], bound_method=methods,
                          property_batch_size=[8, 32]))
    configuration['search_space'].update(PROPERTY_METHOD_SEARCH_SPACE.get(methods[0], {}))
    configuration['source']['sha256']['case_studies/acasxu/hpo.py'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    write_json(args.output / 'config.json', configuration)
    baseline = sampling_metrics(original, original, specs, inputs['selection'], broad['selection'])
    before = verify(original, Path(case['model']), case, specs, baseline, args.output / 'before', args.solver_timeout)
    write_json(args.output / 'baseline.json', dict(selection=baseline, verification=before))
    if before['certified']:
        final = args.output / 'final'
        final.mkdir()
        torch.save({k: v.detach().cpu() for k, v in original.state_dict().items()}, final / 'model.pt')
        write_json(final / 'config.json', dict(parameters=None, reason='all original obligations certified'))
        test = sampling_metrics(original, original, specs, inputs['test'], broad['test'])
        if test['minimum_property_satisfaction'] != 1:
            raise AssertionError('Test witness contradicts original certificate')
        write_json(final / 'results.json', dict(test=test, verification=before, unchanged=True))
        write_json(args.output / 'results.json', dict(network=args.network, status='original_certified',
                   completed_trials=0, final=str(final.resolve()), test=test, verification=before))
        return
    labels = predictions(original, training).argmin(1)
    dataset = TensorDataset(training, labels)
    cells = fixed_cover(specs, args.selection_cells)
    records = []

    def objective(trial):
        torch.manual_seed(args.seed)
        directory = args.output / 'trials' / f'{trial.number:03d}'
        directory.mkdir(parents=True, exist_ok=False)
        parameters = dict(lr=trial.suggest_float('lr', 1e-5, 3e-3, log=True),
            property_weight=trial.suggest_float('property_weight', 1e-4, 100., log=True),
            margin=trial.suggest_float('margin', 1e-5, 1e-2, log=True),
            weight_ramp_epochs=trial.suggest_int('weight_ramp_epochs', 0, args.epochs // 2),
            bound_method=trial.suggest_categorical('bound_method', methods),
            property_batch_size=trial.suggest_categorical('property_batch_size', [8, 32]))
        parameters['training_options'] = sample_training_options(
            trial, parameters['bound_method'], getattr(args, 'training_options', {}))
        write_json(directory / 'config.json', parameters)
        model = copy.deepcopy(original)
        wrapper_parameters = dict(parameters)
        method = wrapper_parameters.pop('bound_method')
        wrapper = PropertyModelWrapper(model, shape, properties=specs, task_loss=advisory_cross_entropy,
            num_epochs=args.epochs, device=args.device, refinement=True, max_cells=args.training_cells,
            bound_opts={'conv_mode': 'matrix', 'relu': 'adaptive', 'property_bound': 'atomic'},
            training_method=method,
            bound_method=method if method in ('IBP', 'CROWN-IBP', 'CROWN') else 'CROWN', **wrapper_parameters)
        loader = DataLoader(dataset, batch_size=128, shuffle=True,
                            generator=torch.Generator().manual_seed(args.seed))
        start = time.monotonic()
        wrapper.train_model(loader)
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, directory / 'model.pt')
        metrics = sampling_metrics(model, original, specs, inputs['selection'], broad['selection'])
        cover = cover_metrics(wrapper.bounds, cells, specs)
        write_json(directory / 'selection_cover.json', cover)
        record = dict(trial=trial.number, parameters=parameters, selection=metrics,
                      cover_certified=cover['certified'], training_seconds=time.monotonic() - start)
        write_json(directory / 'history.json', wrapper.history)
        write_json(directory / 'results.json', record)
        records.append(record)
        write_json(args.output / 'all_trials.json', records)
        trial.set_user_attr('cover_certified', cover['certified'])
        return objective_values(metrics)

    study = optuna.create_study(study_name=f'acasxu_{args.network}', directions=['maximize', 'maximize'],
        sampler=optuna.samplers.TPESampler(seed=args.seed, n_startup_trials=10),
        storage=f'sqlite:///{args.output.resolve()}/optuna_study.db')
    for weight in (.001, .1, 10.):
        study.enqueue_trial(dict(lr=1e-4, property_weight=weight, margin=.001,
                                 weight_ramp_epochs=0, bound_method=methods[0] if getattr(args, 'training_method', None) else 'CROWN-IBP', property_batch_size=32))
    study.optimize(objective, n_trials=args.trials, gc_after_trial=True)
    selected = max(records, key=lambda r: (r['cover_certified'], *objective_values(r['selection'])))
    final = args.output / 'final'
    final.mkdir()
    model = copy.deepcopy(original)
    model.load_state_dict(torch.load(args.output / 'trials' / f'{selected["trial"]:03d}' / 'model.pt',
                                    map_location=args.device, weights_only=True))
    model.eval()
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, final / 'model.pt')
    write_json(final / 'config.json', dict(selected_trial=selected['trial'], training=selected['parameters'],
        model=case['model'], specification=case['specifications'][0]['path'], artifacts=provenance,
        additional_specifications=[e['path'] for e in case['specifications'][1:]],
        additional_specification_sha256={e['path']: configuration['artifact_sha256'][e['path']] for e in case['specifications'][1:]}))
    probes = torch.cat([points[:8] for points in inputs['test']]).to(reference)
    export_property_graph(model, final / 'model.onnx', shape, probes)
    test = sampling_metrics(model, original, specs, inputs['test'], broad['test'])
    verification = verify(model, final / 'model.onnx', case, specs, test, final / 'verification', args.solver_timeout)
    result = dict(network=args.network, status='repaired_certified' if verification['certified'] else 'repair_incomplete',
        completed_trials=len(records), selected_trial=selected['trial'], parameters=selected['parameters'],
        pareto_trials=[t.number for t in study.best_trials], selection=selected['selection'], test=test,
        verification=verification, final=str(final.resolve()),
        model_sha256=hashlib.sha256((final / 'model.pt').read_bytes()).hexdigest(),
        onnx_sha256=hashlib.sha256((final / 'model.onnx').read_bytes()).hexdigest())
    write_json(final / 'results.json', result)
    write_json(args.output / 'results.json', result)


def collect(suite):
    manifest = suite / 'submission.json'
    cases = json.loads(manifest.read_text())['cases'] if manifest.is_file() else benchmark_cases(suite)
    results, missing = [], []
    for case in cases:
        path = suite / case['network'] / 'results.json'
        if path.is_file():
            result = json.loads(path.read_text())
            properties = result['verification']['properties']
            expected = {Path(e['path']).stem for e in case['specifications']}
            if (result['network'] != case['network'] or len(properties) != len(expected)
                    or {p['property'] for p in properties} != expected
                    or result['verification']['certified'] != all(p['outcome'] == 'certified' for p in properties)):
                raise ValueError('Incomplete or mismatched network result')
            results.append(result)
        else:
            missing.append(case['network'])
    incomplete = [r['network'] for r in results if not r['verification']['certified']]
    report = dict(total_models=len(cases), total_obligations=sum(len(c['specifications']) for c in cases),
        completed_models=len(results), missing_models=missing, incomplete_repairs=incomplete,
        all_properties_certified=not missing and not incomplete, results=results)
    write_json(suite / 'results.json', report)
    lines = ['| Network | Status | Trials | Worst test satisfaction | Test advisory accordance |', '|---|---|---|---|---|']
    for r in results:
        lines.append(f'| {r["network"]} | {r["status"]} | {r["completed_trials"]} | '
                     f'{r["test"]["minimum_property_satisfaction"]:.2%} | {objective_values(r["test"])[1]:.2%} |')
    lines += ['', f'Missing networks: {missing}', f'Uncertified repairs: {incomplete}',
              'Sampled satisfaction and advisory accordance are empirical. Unknown solver results are not certificates.']
    (suite / 'summary.md').write_text('\n'.join(lines) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', type=Path)
    parser.add_argument('--network')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--collect', type=Path)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--trials', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--training-method', choices=TRAINING_METHODS)
    parser.add_argument('--training-options', type=json.loads, default={})
    parser.add_argument('--samples', type=int, default=32768)
    parser.add_argument('--train-samples', type=int, default=4096)
    parser.add_argument('--selection-cells', type=int, default=64)
    parser.add_argument('--training-cells', type=int, default=256)
    parser.add_argument('--solver-timeout', type=float)
    args = parser.parse_args()
    if args.training_options and args.training_method is None:
        parser.error('--training-options requires --training-method')
    if args.training_method is not None:
        try:
            resolve_training_options(args.training_method, args.training_options)
        except ValueError as error:
            parser.error(str(error))
    if args.collect:
        collect(args.collect)
        return
    if args.benchmark is None or args.output is None or args.network is None:
        parser.error('Supply --benchmark, --network and --output')
    if (min(args.trials, args.epochs, args.samples, args.selection_cells, args.training_cells) < 1
            or args.train_samples < 2 or (args.solver_timeout is not None and not 0 < args.solver_timeout < float('inf'))):
        parser.error('Invalid sample, training or verification budget')
    if args.network not in {c['network'] for c in benchmark_cases(args.benchmark)}:
        parser.error('Network is not in the benchmark')
    run(args)


if __name__ == '__main__':
    main()
