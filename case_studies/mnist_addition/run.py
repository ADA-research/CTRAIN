"""Experimental MNIST arithmetic consistency and classification swap equivariance.

This is a new MNIST-based experiment, not a reproduction of a named benchmark.
The three class predictions are learned independently. Training algorithms
and affine bounds are supplied by CTRAIN's existing property workflow.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time

import auto_LiRPA
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from CTRAIN.model_wrappers.certified import PropertyModelWrapper
from CTRAIN.properties.training import TRAINING_METHODS, resolve_training_options
from CTRAIN.data_loaders.data_loaders import MNIST_MEAN, MNIST_STD
from CTRAIN.model_definitions import MNISTAdditionClassifier, MNISTAdditionRegressor, MNISTSwapGraph
from CTRAIN.properties import Box, PropertySpec, Outputs, And, Or
from CTRAIN.train.certified.losses.properties import mnist_addition_cross_entropy
from case_studies.common import source_provenance
from case_studies.mnist_addition.data import paired_mnist_dataset, MNISTPropertyDataset


INPUT_SHAPE = (2, 28, 28)


def pixel_box(image_pair, epsilon):
    """Clip the raw-pixel box first, then transform it into loader coordinates."""
    if (image_pair.shape != INPUT_SHAPE or not math.isfinite(epsilon) or not 0 < epsilon <= 1
            or not torch.isfinite(image_pair).all()
            or not ((image_pair >= 0) & (image_pair <= 1)).all()):
        raise ValueError('Expected a finite [2,28,28] image pair in [0,1] and epsilon in (0,1]')
    image_pair = image_pair.double()
    return Box(((image_pair - epsilon).clamp(0, 1) - MNIST_MEAN) / MNIST_STD,
               ((image_pair + epsilon).clamp(0, 1) - MNIST_MEAN) / MNIST_STD)


def write_vnnlib(path, box):
    """Unsafe complement of the 100 consistent argmax triples; first-index ties."""
    lower, upper = box.lower.flatten().tolist(), box.upper.flatten().tolist()
    lines = ['; Experimental MNIST independent digit/digit/sum logits (10,10,19).',
             '(set-logic QF_LRA)']
    lines += [f'(declare-const X_{i} Real)' for i in range(len(lower))]
    lines += [f'(declare-const Y_{i} Real)' for i in range(39)]
    for i, (lo, hi) in enumerate(zip(lower, upper)):
        lines += [f'(assert (>= X_{i} {lo:.17g}))', f'(assert (<= X_{i} {hi:.17g}))']
    def winner(start, size, selected):
        # torch.argmax chooses the smallest index among equal maxima.
        return '(and ' + ' '.join(
            f'({">" if other < selected else ">="} Y_{start + selected} Y_{start + other})'
            for other in range(size) if other != selected) + ')'
    valid = ['(and ' + ' '.join((winner(0, 10, a), winner(10, 10, b),
                               winner(20, 19, a + b))) + ')'
             for a in range(10) for b in range(10)]
    lines += ['(assert (not (or ' + ' '.join(valid) + ')))']
    # Files are artifacts of a fresh run, never silently overwritten.
    with Path(path).open('x') as stream:
        stream.write('\n'.join(lines) + '\n')


def _winner(start, size, selected):
    outputs = Outputs()
    return And(*[(outputs[start + selected] > outputs[start + other]
                  if other < selected else outputs[start + selected] >= outputs[start + other])
                 for other in range(size) if other != selected])


def addition_formula(task='classification', tau=None):
    """Argmax consistency, or the one-sided affine regression residual bound."""
    outputs = Outputs()
    if task == 'regression':
        return outputs[2] - outputs[0] - outputs[1] <= tau
    return Or(*[And(_winner(0, 10, a), _winner(10, 10, b), _winner(20, 19, a + b))
                for a in range(10) for b in range(10)])


def swap_formula():
    """Match winning classes across exchanged operand heads and unchanged sum heads."""
    matches = []
    for original, swapped, size in ((0, 49, 10), (10, 39, 10), (20, 59, 19)):
        matches.append(Or(*[And(_winner(original, size, digit), _winner(swapped, size, digit))
                            for digit in range(size)]))
    return And(*matches)


def specifications(inputs, epsilon, task='classification', tau=None, property_kind='arithmetic'):
    """Raw pixel boxes for arithmetic, or exact normalized inputs for swap."""
    if property_kind == 'swap':
        return PropertySpec('swap_equivariance', swap_formula(),
                            [Box(pair.double(), pair.double()) for pair in inputs])
    return PropertySpec('arithmetic_consistency' if task == 'classification' else 'arithmetic_upper_residual',
                        addition_formula(task, tau),
                        [pixel_box(pair, epsilon) for pair in inputs])


def task_metrics(model, dataset, device, batch_size, tau=None, swap=False):
    """Clean classification/regression metrics; these are not certificates."""
    evaluation_model = MNISTSwapGraph(model) if swap else model
    with torch.no_grad():
        output = torch.cat([evaluation_model(x.to(device)).cpu() for x, _ in
                            DataLoader(dataset, batch_size=batch_size,
                                       generator=torch.Generator().manual_seed(0))])
    if swap:
        output, swapped = output.split(39, dim=1)
    target = dataset.tensors[1]
    if output.shape[1] == 3:
        error = output - target.float()
        residual = output[:, 2] - output[:, 0] - output[:, 1]
        return dict(output_order=['a', 'b', 'sum'],
                    mse_per_output=error.square().mean(0).tolist(),
                    mae_per_output=error.abs().mean(0).tolist(),
                    mean_signed_residual=float(residual.mean()),
                    maximum_signed_residual=float(residual.max()),
                    empirical_property_satisfied_fraction=float((residual <= tau).float().mean()))
    heads = output.split((10, 10, 19), dim=1)
    predicted = torch.stack([head.argmax(dim=1) for head in heads], dim=1)
    residual = predicted[:, 2] - predicted[:, 0] - predicted[:, 1]
    correct = predicted == target
    metrics = dict(output_order=['a', 'b', 'sum'],
                cross_entropy_per_output=[float(nn.functional.cross_entropy(head, target[:, i]))
                                          for i, head in enumerate(heads)],
                accuracy_per_output=correct.float().mean(dim=0).tolist(),
                triple_accuracy=float(correct.all(dim=1).float().mean()),
                mean_absolute_class_residual=float(residual.abs().float().mean()),
                maximum_absolute_class_residual=int(residual.abs().max()),
                empirical_consistent_fraction=float((residual == 0).float().mean()))
    if swap:
        swapped_predictions = torch.stack([head.argmax(1) for head in swapped.split((10, 10, 19), 1)], 1)
        matches = predicted == swapped_predictions[:, [1, 0, 2]]
        metrics.update(swap_agreement_per_output=matches.float().mean(0).tolist(),
                       empirical_swap_equivariant_fraction=float(matches.all(1).float().mean()))
    return metrics


def certify(wrapper, spec, method):
    """Fresh affine-spec bounds for every original box, without input refinement."""
    report = wrapper.certify_properties(properties=[spec], method=method, refine=False)
    rows = report['cells']
    count = sum(row['outcome'] == 'certified' for row in rows)
    report.update(certified_count=count, total_boxes=len(rows), certified_fraction=count / len(rows),
                scope='swap-equivariant class predictions at the declared normalized input pairs'
                      if spec.name == 'swap_equivariance' else
                      f'all inputs in each declared clipped normalized pixel box; {spec.name}',
                argmax_ties='smallest class index; certificates require positive margins',
                bound_strategy='affine leaf bounds with monotone Boolean composition; no input refinement')
    if spec.name not in ('arithmetic_consistency', 'swap_equivariance'):
        report.pop('argmax_ties')
    return report


def mnist_source(args, train, seed, smoke_size):
    if not args.smoke:
        source = torchvision.datasets.MNIST(str(args.data_root), train=train, download=args.download)
        return source.data, source.targets
    generator = torch.Generator().manual_seed(seed)
    labels = torch.arange(smoke_size) % 10
    images = (labels[:, None, None] / 9 * 230
              + torch.rand(smoke_size, 28, 28, generator=generator) * 25).to(torch.uint8)
    return images, labels


def make_wrapper(args, spec, parameters, state, directory=None, property_dataset=None):
    regression = args.task == 'regression'
    model = (MNISTAdditionRegressor if regression else MNISTAdditionClassifier)(args.width)
    model.load_state_dict(state)
    spec = PropertySpec(spec.name, spec.formula, spec.domains[:1])
    method = parameters.get('bound_method', args.method)
    wrapper = PropertyModelWrapper(model, INPUT_SHAPE, properties=[spec],
        verification_model=MNISTSwapGraph(model) if args.property == 'swap' else None,
        task_loss=nn.MSELoss() if regression else mnist_addition_cross_entropy, num_epochs=args.epochs, device=args.device,
        lr=parameters['lr'], property_weight=parameters['property_weight'],
        warm_up_epochs=parameters.get('warm_up_epochs', args.warm_up_epochs),
        weight_ramp_epochs=parameters['weight_ramp_epochs'], bound_method=method if method in ('IBP', 'CROWN-IBP', 'CROWN') else args.eval_method,
        training_method=method, training_options=parameters.get('training_options', getattr(args, 'training_options', {})),
        margin=.001, property_batch_size=args.property_batch_size,
        checkpoint_save_path=None if directory is None else directory / 'checkpoints',
        checkpoint_save_interval=args.epochs, bound_opts={'property_bound': 'atomic'},
        property_dataset=property_dataset)
    return wrapper


def train_trial(args, dataset, spec, parameters, initial, directory, *, metric_dataset=None, metric_split='heldout'):
    # Paired initialization, batches and property-cell sampling across all arms.
    torch.manual_seed(args.seed + 2)
    if args.task == 'regression':
        dataset = TensorDataset(dataset.tensors[0], dataset.tensors[1].float())
    wrapper = make_wrapper(args, spec, parameters, initial, directory,
                           property_dataset=MNISTPropertyDataset(dataset, args.epsilon, count=args.property_boxes))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(args.seed + 2))
    def epoch_metrics(model):
        metrics = task_metrics(model, metric_dataset, args.device, args.batch_size, args.tau,
                               swap=args.property == 'swap')
        keys = ('accuracy_per_output', 'triple_accuracy', 'swap_agreement_per_output',
                'empirical_swap_equivariant_fraction', 'mse_per_output', 'mae_per_output',
                'empirical_property_satisfied_fraction')
        return {f'{metric_split}_{key}': metrics[key] for key in keys if key in metrics}
    start = time.monotonic()
    wrapper.train_model(loader, epoch_metrics=epoch_metrics if metric_dataset is not None else None)
    return wrapper, time.monotonic() - start


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, help='JSON parameters; command-line flags override file values')
    parser.add_argument('--output', type=Path, default=Path('/tmp/ctrain-mnist-addition-ce'))
    parser.add_argument('--data-root', type=Path, default=Path('data'))
    parser.add_argument('--download', action='store_true', help='Explicitly download missing MNIST files')
    parser.add_argument('--skip-ce-baseline', action='store_true', help='Skip the task-only baseline')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--train-pairs', type=int, default=60000,
                        help='Training pairs; must cover every training image in each position')
    parser.add_argument('--test-pairs', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=64, help='Task batch size')
    parser.add_argument('--property-batch-size', type=int, default=64,
                        help='Property domains sampled with replacement per task update; also bound-evaluation batch size')
    parser.add_argument('--width', type=int, default=64)
    parser.add_argument('--task', choices=('classification', 'regression'), default='classification')
    parser.add_argument('--property', choices=('arithmetic', 'swap'), default='arithmetic',
                        help='Swap checks class equivariance at exact input pairs (epsilon=0)')
    parser.add_argument('--tau', type=float, help='Regression residual bound: sum - a - b <= tau')
    parser.add_argument('--property-boxes', type=int, help='Training-pair boxes; default: all pairs; not an HPO parameter')
    parser.add_argument('--epsilon', type=float, default=.02, help='Radius in unnormalized [0,1] pixel units')
    parser.add_argument('--property-weight', type=float, default=10.)
    parser.add_argument('--warm-up-epochs', type=int, default=0, help='Initial task-only epochs')
    parser.add_argument('--weight-ramp-epochs', type=int, default=0, help='Property-weight ramp epochs after warm-up')
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--method', choices=TRAINING_METHODS, default='CROWN-IBP')
    parser.add_argument('--training-options', type=json.loads, default={},
                        help='JSON parameters for the selected training method')
    parser.add_argument('--eval-method', choices=('IBP', 'CROWN-IBP', 'CROWN'), default='CROWN')
    parser.add_argument('--smoke', action='store_true', help='Two-epoch synthetic execution check, not MNIST evidence')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    if args.config is not None:
        try:
            settings = json.loads(args.config.read_text())
        except (OSError, json.JSONDecodeError) as error:
            parser.error(f'Cannot read config: {error}')
        if not isinstance(settings, dict):
            parser.error('Config must be a JSON object')
        unknown = set(settings) - (vars(args).keys() - {'config'})
        if unknown:
            parser.error(f'Unknown config options: {", ".join(sorted(unknown))}')
        config_argv = []
        for name, value in settings.items():
            option = '--' + name.replace('_', '-')
            if name == 'training_options':
                if not isinstance(value, dict):
                    parser.error('training_options must be a JSON object')
                config_argv.extend((option, json.dumps(value)))
            elif isinstance(getattr(args, name), bool):
                if type(value) is not bool:
                    parser.error(f'{name} must be a JSON boolean')
                if value:
                    config_argv.append(option)
            else:
                if value is None or isinstance(value, (bool, list, dict)):
                    parser.error(f'{name} must be a scalar parameter')
                config_argv.extend((option, str(value)))
        # Reuse argparse's type/choice validation for file settings as well.
        args = parser.parse_args(config_argv + (sys.argv[1:] if argv is None else list(argv)))
    try:
        resolve_training_options(args.method, args.training_options)
    except ValueError as error:
        parser.error(str(error))
    for name in ('epochs', 'train_pairs', 'test_pairs', 'batch_size', 'property_batch_size', 'width'):
        if getattr(args, name) < 1:
            parser.error(f'{name} must be positive')
    for name in ('property_weight', 'lr'):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f'{name} must be positive and finite')
    if not math.isfinite(args.epsilon) or not 0 <= args.epsilon <= 1:
        parser.error('epsilon must be finite and in [0,1]')
    if args.property == 'swap':
        if args.task != 'classification':
            parser.error('--property swap requires --task classification')
        args.epsilon = 0.
    elif args.epsilon == 0:
        parser.error('Arithmetic properties require positive epsilon')
    if args.seed < 0 or args.seed >= 2 ** 32 - 3:
        parser.error('seed must be in [0, 2**32 - 3)')
    if (args.warm_up_epochs < 0 or args.weight_ramp_epochs < 0
            or args.warm_up_epochs + args.weight_ramp_epochs > args.epochs):
        parser.error('warm-up and weight ramp must be nonnegative and fit epochs')
    if args.task == 'regression':
        args.tau = .1 if args.tau is None else args.tau
        if not math.isfinite(args.tau) or args.tau < 0:
            parser.error('tau must be finite and nonnegative')
    elif args.tau is not None:
        parser.error('--tau requires --task regression')
    if args.property_boxes is not None and not 1 <= args.property_boxes <= args.train_pairs:
        parser.error('property-boxes must be between 1 and train-pairs')
    if args.smoke:
        args.epochs, args.train_pairs, args.test_pairs, args.width = 2, 32, 16, 8
        if args.property_boxes is not None:
            args.property_boxes = min(args.property_boxes, args.train_pairs)
        args.warm_up_epochs = min(args.warm_up_epochs, args.epochs)
        args.weight_ramp_epochs = min(args.weight_ramp_epochs, args.epochs - args.warm_up_epochs)
    elif args.train_pairs < 60000:
        parser.error('train-pairs must be at least 60000 to cover every MNIST training image')
    return args


def main(argv=None):
    args = parse_args(argv)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    if args.config is not None:
        config['parameter_file_sha256'] = hashlib.sha256(args.config.read_bytes()).hexdigest()
    config.update(case='experimental_mnist_arithmetic_consistency', established_benchmark=False,
                  outputs=['a_logits[0:10]', 'b_logits[10:20]', 'sum_logits[20:39]'],
                  task_loss='mean of three cross-entropy losses; integer targets [a,b,a+b]',
                  property='argmax(sum_logits) == argmax(a_logits) + argmax(b_logits)',
                  normalization=dict(mean=MNIST_MEAN, std=MNIST_STD, transform='(uint8/255-mean)/std',
                                     epsilon_coordinates='raw pixels; clip before normalization', augmentation=False),
                  selection='final epoch; no validation/test model selection',
                  training_properties='lazy box around every training pair; no training VNNLIB files',
                  comparison=('CE + certified property penalty only' if args.skip_ce_baseline else
                              'CE-only vs CE + certified property penalty; paired initialization and batches'))
    if args.task == 'regression':
        config.update(case='experimental_mnist_arithmetic_regression',
                      outputs=['a', 'b', 'sum'], task_loss='mean squared error against [a,b,a+b]',
                      property='sum - a - b <= tau',
                      comparison='MSE + property penalty only' if args.skip_ce_baseline else
                                 'MSE-only vs MSE + property penalty; paired initialization and batches')
    else:
        config.pop('tau')
    if args.property == 'swap':
        config.update(case='experimental_mnist_swap_equivariance',
                      property='predicted operands exchange and predicted sum agrees after swapping',
                      property_domains='singleton normalized input pairs',
                      verification_graph='concatenate f(x) and f(swap(x)); shared model weights')
        config['normalization']['epsilon_coordinates'] = 'exact normalized inputs; no perturbation'
    config['training_properties'] = ('lazy box around every training pair' if args.property_boxes is None
                                     else f'lazy boxes around the first {args.property_boxes} training pairs')
    if args.property == 'swap':
        config['training_properties'] = ('exact point for every training pair' if args.property_boxes is None
                                         else f'exact points for the first {args.property_boxes} training pairs')
    config['batch_sampling'] = 'property_batch_size random domains per task update, with replacement from the selected training-pair pool'
    if args.dry_run:
        print(json.dumps(config, indent=2))
        return
    if args.output.exists():
        argparse.ArgumentParser().error('output must be a fresh directory')
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    pools, property_inputs, provenance = {}, {}, {}
    for split, offset in (('train', 0), ('test', 1)):
        count = args.train_pairs if split == 'train' else args.test_pairs
        images, labels = mnist_source(args, split == 'train', args.seed + offset, count)
        pools[split], indices = paired_mnist_dataset(images, labels, count, args.seed + offset,
                                                   cover_all=split == 'train')
        # One domain initializes the formula; the lazy dataset supplies all training boxes.
        domain_count = 1 if split == 'train' else args.test_pairs
        property_inputs[split] = (pools[split].tensors[0][:domain_count] if args.property == 'swap'
                                  else images[indices[:domain_count]].float() / 255)
        provenance[split] = dict(original_split=split,
            pairing='independent shuffled complete passes' if split == 'train' else 'with replacement',
            unique_images_per_position=[int(indices[:, side].unique().numel()) for side in range(2)],
            pair_indices=indices.tolist(), images_sha256=hashlib.sha256(images.numpy().tobytes()).hexdigest(),
            labels_sha256=hashlib.sha256(labels.numpy().tobytes()).hexdigest())
    args.output.mkdir(parents=True, exist_ok=False)
    specs = {}
    for split in ('train', 'test'):
        specs[split] = specifications(property_inputs[split], args.epsilon, args.task, args.tau, args.property)
        provenance[split].update(property_pair_indices=list(range(args.test_pairs if split == 'test' else (args.property_boxes or args.train_pairs))),
                                 specification_role='formula template; selected training boxes generated on demand'
                                 if split == 'train' else 'evaluation boxes')
    initial_state = (MNISTAdditionRegressor if args.task == 'regression' else MNISTAdditionClassifier)(args.width).state_dict()
    torch.save(initial_state, args.output / 'initial_model.pt')
    config.update(data=dict(kind='synthetic execution fixture' if args.smoke else 'MNIST official train/test splits',
                            provenance=provenance), source=source_provenance(),
                  environment=dict(python=platform.python_version(), torch=torch.__version__,
                                   torchvision=torchvision.__version__, auto_lirpa=auto_LiRPA.__version__,
                                   threads=1))
    (args.output / 'config.json').write_text(json.dumps(config, indent=2, allow_nan=False))
    results = {}
    arms = [('certified', args.property_weight)]
    if not args.skip_ce_baseline:
        arms.insert(0, ('task_only' if args.task == 'regression' else 'ce_only', 0.))
    for name, weight in arms:
        print(f'Training {name}: property_weight={weight:g}', flush=True)
        directory = args.output / name
        directory.mkdir()
        parameters = dict(lr=args.lr, property_weight=weight, warm_up_epochs=args.warm_up_epochs,
                          weight_ramp_epochs=args.weight_ramp_epochs)
        wrapper, training_seconds = train_trial(args, pools['train'], specs['train'],
                                               parameters, initial_state, directory,
                                               metric_dataset=pools['test'], metric_split='test')
        (directory / 'training_config.json').write_text(json.dumps(wrapper.configuration, indent=2))
        model = wrapper.original_model
        model.eval()
        result = dict(training_seconds=training_seconds, history=wrapper.history,
                      metrics={split: task_metrics(model, pools[split], args.device, args.batch_size, args.tau, swap=args.property == 'swap')
                               for split in ('train', 'test')},
                      verification={'test': certify(wrapper, specs['test'], args.eval_method)})
        result['verification']['train'] = wrapper.certify_properties(method=args.eval_method)
        torch.save(model.cpu().state_dict(), directory / 'model.pt')
        (directory / 'results.json').write_text(json.dumps(result, indent=2, allow_nan=False))
        results[name] = dict(test_metrics=result['metrics']['test'],
                            train_certified_fraction=result['verification']['train']['certified_fraction'],
                            test_certified_fraction=result['verification']['test']['certified_fraction'],
                            training_seconds=training_seconds)
    summary = dict(case=config['case'], smoke=args.smoke, epsilon=args.epsilon, property=config['property'], results=results)
    (args.output / 'results.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
