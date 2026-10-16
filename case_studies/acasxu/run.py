"""Repair a user-selected ONNX/VNNLIB pair, an ACAS preset, or a smoke fixture."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from CTRAIN.model_wrappers.certified import PropertyModelWrapper
from CTRAIN.properties.training import TRAINING_METHODS, resolve_training_options
from CTRAIN.properties import Box, load_vnnlib_property
from CTRAIN.data_loaders.acasxu import acasxu_property, download_acasxu, load_acasxu, SOURCE
from CTRAIN.data_loaders import load_vnnlib_problem
from CTRAIN.train.certified.losses.properties import advisory_cross_entropy
from case_studies.common import source_provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', type=Path)
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--revision')
    parser.add_argument('--case', choices=('7', '8'), default='7', help='ACAS preset when using --artifacts or --smoke')
    parser.add_argument('--model', type=Path, help='User-selected local ONNX model')
    parser.add_argument('--specification', type=Path, help='User-selected local VNNLIB file')
    parser.add_argument('--additional-specifications', nargs='+', type=Path, default=[],
                        help='Additional VNNLIB properties to train jointly on their own input domains')
    parser.add_argument('--input-shape', nargs='+', type=int, help='Override dynamic ONNX input dimensions, excluding batch')
    parser.add_argument('--output-semantics', choices=('unsafe', 'safe'), default='unsafe')
    parser.add_argument('--output', type=Path, default=Path('/tmp/ctrain-property-acasxu'))
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--warm-up-epochs', type=int, default=0)
    parser.add_argument('--weight-ramp-epochs', type=int, default=0)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--complete', action='store_true')
    parser.add_argument('--method', choices=TRAINING_METHODS, default='CROWN-IBP')
    parser.add_argument('--training-options', type=json.loads, default={},
                        help='JSON parameters for the selected training method')
    args = parser.parse_args()
    budget = 2 if args.smoke else args.epochs
    if (args.warm_up_epochs < 0 or args.weight_ramp_epochs < 0
            or args.warm_up_epochs + args.weight_ramp_epochs > budget):
        parser.error('warm-up and weight ramp must be nonnegative and fit the epoch budget')
    try:
        resolve_training_options(args.method, args.training_options)
    except ValueError as error:
        parser.error(str(error))
    custom = args.model is not None or args.specification is not None
    if custom and (args.model is None or args.specification is None):
        parser.error('Supply --model and --specification together')
    if args.additional_specifications and not custom:
        parser.error('--additional-specifications requires --model/--specification')
    if custom:
        paths = [args.specification.resolve(), *[path.resolve() for path in args.additional_specifications]]
        if len(set(paths)) != len(paths):
            parser.error('Specification files must be distinct')
    if custom and (args.artifacts is not None or args.download or args.smoke):
        parser.error('--model/--specification cannot be combined with presets, downloads or --smoke')
    if not custom and args.output_semantics != 'unsafe':
        parser.error('--output-semantics applies to user-selected specifications')
    if not custom and args.input_shape is not None:
        parser.error('--input-shape applies to user-selected networks')
    config = dict(case=f'acasxu_property_{args.case}', source=SOURCE, seed=args.seed,
                  warm_up_epochs=args.warm_up_epochs, weight_ramp_epochs=args.weight_ramp_epochs,
                  epochs=2 if args.smoke else args.epochs, smoke=args.smoke, method=args.method, training_options=args.training_options,
                  protocol='repair; cross-entropy fidelity to source-model advisories',
                  fidelity_loss='cross_entropy(-scores, source_scores.argmin(dim=1))',
                  fidelity_samples=dict(train=64 if args.smoke else 4096,
                                        validation=32 if args.smoke else 2048),
                  fidelity_seed=args.seed + 100,
                  input_coordinates='released ONNX/VNN-LIB coordinates; no additional normalization',
                  output_convention='lowest score selects advisory; unsafe ties excluded')
    if custom:
        config.update(case='user_selected', source='local files',
                      model=str(args.model), specification=str(args.specification),
                      additional_specifications=[str(path.resolve()) for path in args.additional_specifications],
                      output_semantics=args.output_semantics,
                      output_convention='specified by VNNLIB; lowest score selects fidelity advisory')
    if args.epochs < 1:
        parser.error('epochs must be positive')
    if args.dry_run:
        print(json.dumps(config, indent=2)); return
    if args.download:
        if args.artifacts is None:
            parser.error('--download requires --artifacts')
        download_acasxu(args.artifacts, args.revision)
    torch.set_num_threads(1); torch.manual_seed(args.seed)
    if custom:
        model, spec, manifest = load_vnnlib_problem(args.model, args.specification,
            input_shape=args.input_shape, output_semantics=args.output_semantics)
        box = spec.domains[0]
        config['artifacts'] = manifest
    elif args.smoke:
        model = nn.Sequential(nn.Linear(5, 8), nn.ReLU(), nn.Linear(8, 5))
        box = Box([-1] * 5, [1] * 5)
        spec = acasxu_property(args.case, box)
        config['protocol'] = 'synthetic execution check; not released ACAS Xu'
    else:
        if args.artifacts is None:
            parser.error('Supply explicitly acquired --artifacts, or use --smoke')
        model, spec, manifest = load_acasxu(args.artifacts, args.case)
        box = spec.domains[0]
        config['artifacts'] = manifest
    specs = [spec]
    if args.additional_specifications:
        config['additional_specification_sha256'] = {}
        for path in args.additional_specifications:
            config['additional_specification_sha256'][str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
            specs.append(load_vnnlib_property(path, input_shape=tuple(box.lower.shape),
                                             output_semantics=args.output_semantics))
    domains = [domain for prop in specs for domain in prop.domains]
    domain_properties = torch.tensor([index for index, prop in enumerate(specs) for _ in prop.domains])
    model.eval()
    generator = torch.Generator().manual_seed(args.seed + 100)
    count = 96 if args.smoke else 6144
    split = 64 if args.smoke else 4096
    if len(domains) > min(split, count - split):
        raise ValueError('Too many input boxes to cover in both fidelity data splits')
    regions = torch.arange(count) % len(domains)
    lower = torch.stack([b.lower for b in domains])[regions]
    upper = torch.stack([b.upper for b in domains])[regions]
    data = (lower + torch.rand((count, *box.lower.shape), generator=generator,
                              dtype=torch.double) * (upper - lower)).to(next(model.parameters()))
    with torch.no_grad():
        source_scores = model(data)
        target = source_scores.argmin(dim=1)
    train = DataLoader(TensorDataset(data[:split], target[:split]), batch_size=64, shuffle=False)
    validation = DataLoader(TensorDataset(data[split:], target[split:]), batch_size=64)
    args.output.mkdir(parents=True, exist_ok=False)
    wrapper = PropertyModelWrapper(model, tuple(box.lower.shape), properties=specs, task_loss=advisory_cross_entropy,
        num_epochs=config['epochs'], device=args.device, bound_method=args.method if args.method in ('IBP', 'CROWN-IBP', 'CROWN') else 'CROWN',
        training_method=args.method, training_options=args.training_options,
        warm_up_epochs=args.warm_up_epochs, weight_ramp_epochs=args.weight_ramp_epochs,
        refinement=True, property_batch_size=8, max_cells=32 if args.smoke else 1024,
        checkpoint_save_path=args.output / 'checkpoints')
    config.update(training=wrapper.configuration, source_code=source_provenance(),
                  fidelity_sampling='equal frequency per original input box across all training properties')
    (args.output / 'config.json').write_text(json.dumps(config, indent=2))
    before = wrapper.certify_properties(refine=False)
    wrapper.train_model(train, validation)
    after = wrapper.certify_properties(method=wrapper.bound_method, refine=not args.smoke, complete=args.complete,
                                       results_path=args.output / 'verification')
    model.cpu().eval()
    with torch.no_grad():
        prediction = model(data[split:])
        metrics = dict(advisory_ce=float(advisory_cross_entropy(prediction, target[split:])),
            score_mse=float(nn.functional.mse_loss(prediction, source_scores[split:])),
            advisory_agreement=float((prediction.argmin(1) == target[split:]).float().mean()))
        property_metrics = {}
        memberships = domain_properties[regions]
        for index, prop in enumerate(specs):
            mask = memberships[split:] == index
            property_metrics[prop.name] = dict(train_samples=int((memberships[:split] == index).sum()),
                validation_samples=int(mask.sum()),
                advisory_ce=float(advisory_cross_entropy(prediction[mask], target[split:][mask])),
                score_mse=float(nn.functional.mse_loss(prediction[mask], source_scores[split:][mask])),
                advisory_agreement=float((prediction[mask].argmin(1) == target[split:][mask]).float().mean()))
    torch.save(model.state_dict(), args.output / 'model.pt')
    (args.output / 'results.json').write_text(json.dumps(dict(metrics=metrics, property_metrics=property_metrics,
        history=wrapper.history,
        before=before, after=after), indent=2, allow_nan=False))
    print(json.dumps(dict(metrics=metrics, certified=after['certified']), indent=2))


if __name__ == '__main__':
    main()
