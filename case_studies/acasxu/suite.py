"""Jointly train selected ACAS networks on all their assigned properties."""
import argparse
import csv
import json
from CTRAIN.properties.training import TRAINING_METHODS, resolve_training_options
from pathlib import Path
import re
import subprocess
import sys
from case_studies.common import source_provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='Fresh suite output directory')
    parser.add_argument('--cases', nargs='+', default=['2_1:2', '3_9:2', '4_5:2', '5_9:2'],
                        help='NETWORK:PROPERTY pairs, e.g. 4_5:2; all properties assigned to each selected network are trained')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--training-options', type=json.loads, default={})
    parser.add_argument('--training-method', choices=TRAINING_METHODS, default='CROWN-IBP')
    parser.add_argument('--evaluation-methods', nargs='+',
                        choices=['IBP', 'CROWN-IBP', 'CROWN', 'ALPHA-CROWN'],
                        default=['CROWN-IBP', 'CROWN'],
                        help='Comparison methods (default: CROWN-IBP CROWN); ALPHA-CROWN is opt-in')
    parser.add_argument('--samples', type=int, default=32768)
    parser.add_argument('--max-cells', type=int, default=256)
    parser.add_argument('--alpha-iterations', type=int, default=100)
    parser.add_argument('--dry-run', action='store_true', help='Validate files and print commands without training')
    args = parser.parse_args()
    try:
        resolve_training_options(args.training_method, args.training_options)
    except ValueError as error:
        parser.error(str(error))
    if min(args.epochs, args.samples, args.max_cells, args.alpha_iterations) < 1:
        parser.error('epochs, samples, max-cells and alpha-iterations must be positive')
    if len(set(args.cases)) != len(args.cases) or len(set(args.evaluation_methods)) != len(args.evaluation_methods):
        parser.error('cases and evaluation methods must not contain duplicates')
    if args.output.exists():
        parser.error('output directory already exists; choose a fresh destination')
    benchmark = args.benchmark.resolve()
    if not (benchmark / 'instances.csv').is_file():
        parser.error('benchmark must contain instances.csv')
    with (benchmark / 'instances.csv').open() as file:
        assigned = {(Path(row[0]).name, Path(row[1]).name) for row in csv.reader(file) if row}
    root = Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    commands = []
    networks = set()
    for case in args.cases:
        if re.fullmatch(r'[1-5]_[1-9]:(?:[1-9]|10)', case) is None:
            parser.error(f'invalid case {case!r}; use NETWORK:PROPERTY')
        network, prop = case.split(':')
        if network in networks:
            parser.error(f'network {network} was selected more than once')
        networks.add(network)
        model = benchmark / 'onnx' / f'ACASXU_run2a_{network}_batch_2000.onnx'
        specification = benchmark / 'vnnlib' / f'prop_{prop}.vnnlib'
        if (model.name, specification.name) not in assigned:
            parser.error(f'{case} is not an assigned benchmark model/property pair')
        specifications = sorted({name for model_name, name in assigned if model_name == model.name},
                                key=lambda name: int(Path(name).stem.split('_')[-1]))
        additional = [benchmark / 'vnnlib' / name for name in specifications if name != specification.name]
        if not model.is_file() or not specification.is_file() or any(not path.is_file() for path in additional):
            parser.error(f'missing model or specification for {case}')
        run = output / f'acasxu_{network}_all_properties_seed{args.seed}'
        commands.append([sys.executable, '-m', 'case_studies.acasxu.run',
            '--model', str(model), '--specification', str(specification), '--output', str(run),
            '--epochs', str(args.epochs), '--seed', str(args.seed), '--device', args.device,
            '--method', args.training_method, '--training-options', json.dumps(args.training_options)])
        if additional:
            commands[-1] += ['--additional-specifications', *map(str, additional)]
        cover = None
        for method in args.evaluation_methods:
            destination = run / f'comparison_{method.lower().replace("-", "_")}'
            command = [sys.executable, '-m', 'case_studies.acasxu.evaluate',
                '--run', str(run), '--benchmark', str(benchmark), '--output', str(destination),
                '--assigned-only', '--method', method, '--device', args.device,
                '--seed', str(args.seed), '--samples', str(args.samples),
                '--max-cells', str(args.max_cells), '--alpha-iterations', str(args.alpha_iterations)]
            if method == 'ALPHA-CROWN' and cover is not None:
                command += ['--cover-from', str(cover)]
            elif method != 'ALPHA-CROWN':
                cover = destination / 'results.json'
            commands.append(command)
    manifest = dict(cases=args.cases, training_scope='all properties assigned in instances.csv',
                    commands=commands, source=source_provenance())
    if args.dry_run:
        print(json.dumps(manifest, indent=2)); return
    output.mkdir(parents=True, exist_ok=False)
    (output / 'suite.json').write_text(json.dumps(manifest, indent=2))
    for command in commands:
        print('Running: ' + ' '.join(command), flush=True)
        subprocess.run(command, cwd=root, check=True)


if __name__ == '__main__':
    main()
