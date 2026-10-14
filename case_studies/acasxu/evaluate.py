"""Compare a repair run against its original model on every ACAS property."""
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import torch
from torch import nn
from CTRAIN import PropertyModelWrapper
from CTRAIN.data_loaders import load_vnnlib_problem
from CTRAIN.properties import Box, load_vnnlib_property
from CTRAIN.eval.properties import find_counterexample
from case_studies.common import source_provenance


def sample_inputs(spec, reference, count, seed):
    """Use identical points before/after, strictly within each declared box."""
    generator = torch.Generator().manual_seed(seed)
    chunks = []
    for box in spec.domains:
        lower, upper = box.lower.to(reference), box.upper.to(reference)
        lower = torch.where(lower.double() < box.lower.to(lower.device),
                            torch.nextafter(lower, torch.full_like(lower, float('inf'))), lower)
        upper = torch.where(upper.double() > box.upper.to(upper.device),
                            torch.nextafter(upper, torch.full_like(upper, -float('inf'))), upper)
        if (lower > upper).any():
            raise ValueError('No representable points in this input domain')
        random = torch.rand((count, *lower.shape), generator=generator, dtype=torch.double)
        chunks.append((box.lower + random * (box.upper - box.lower)).to(reference).clamp(lower, upper))
    return torch.cat(chunks)


def evaluate(model, spec, inputs, method, max_cells, alpha_iterations=100, cover=None):
    with torch.no_grad():
        output = torch.cat([model(chunk) for chunk in inputs.split(1024)])
        if not torch.isfinite(output).all():
            raise FloatingPointError('Non-finite evaluation predictions')
        satisfied = spec.formula.evaluate(output).flatten()
        margin = spec.formula.margin(output).flatten()
    witness = inputs[~satisfied][:1].detach().cpu().tolist()
    if not witness:
        for box in spec.domains:
            found = find_counterexample(model, spec.formula, box)
            if found is not None:
                witness = [found]
                break
    wrapper = PropertyModelWrapper(model, tuple(spec.domains[0].lower.shape),
        properties=[spec], task_loss=nn.MSELoss(), num_epochs=1,
        device=inputs.device, property_batch_size=32, max_cells=max_cells,
        bound_opts={'conv_mode': 'matrix', 'relu': 'adaptive',
                    'optimize_bound_args': {'iteration': alpha_iterations}})
    if cover is not None:
        wrapper.partition = []
        for row in cover['cells']:
            property_index, domain, path = row['id'].split(':')
            if property_index != '0' or row['property'] != spec.name:
                raise ValueError('Saved cover does not match this property')
            wrapper.partition.append(dict(property=0, domain=int(domain), path=path,
                depth=len(path), probability=1 / len(spec.domains) / 2 ** len(path),
                box=Box(**row['domain'])))
    # certify_properties validates the full cover and recomputes every bound.
    certificate = wrapper.certify_properties(method=method, refine=cover is None)
    if witness and certificate['certified']:
        raise AssertionError('Witness contradicts the bound certificate')
    result = dict(outcome='violated' if witness else ('certified' if certificate['certified'] else 'unknown'),
                  sampled_violations=int((~satisfied).sum()), sampled_inputs=len(inputs),
                  sample_minimum_margin=float(margin.min()),
                  counterexample=witness[0] if witness else None, certificate=certificate)
    return result, output, satisfied


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--benchmark', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--samples', type=int, default=32768, help='Random points per original input box')
    parser.add_argument('--max-cells', type=int, default=256, help='Certification budget per input box')
    parser.add_argument('--method', choices=('IBP', 'CROWN-IBP', 'CROWN', 'ALPHA-CROWN'), default='CROWN-IBP')
    parser.add_argument('--alpha-iterations', type=int, default=100)
    parser.add_argument('--assigned-only', action='store_true', help='Evaluate only benchmark obligations for this network')
    parser.add_argument('--cover-from', type=Path, help='Reuse validated input covers from a previous comparison, with fresh bounds and no further refinement')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()
    if args.samples < 1 or args.max_cells < 1 or args.alpha_iterations < 1:
        parser.error('samples, max-cells and alpha-iterations must be positive')
    directory = args.output or args.run / 'property_comparison'
    directory.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    config = json.loads((args.run / 'config.json').read_text())
    for path in config.get('additional_specifications', []):
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != config['additional_specification_sha256'][path]:
            raise ValueError(f'Training specification hash mismatch: {path}')
    original, _, provenance = load_vnnlib_problem(config['model'], config['specification'],
        expected_sha256=config['artifacts']['sha256'])
    repaired = copy.deepcopy(original)
    weights = args.run / 'model.pt'
    repaired.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True))
    original.to(args.device).eval(); repaired.to(args.device).eval()
    reference = next(original.parameters())
    shape = tuple(provenance['input_shape'])
    with (args.benchmark / 'instances.csv').open() as file:
        applicable = {Path(row[1]).name for row in csv.reader(file)
                      if Path(row[0]).name == Path(config['model']).name}
    files = sorted((args.benchmark / 'vnnlib').glob('prop_*.vnnlib'),
                   key=lambda path: int(path.stem.split('_')[-1]))
    if args.assigned_only:
        files = [path for path in files if path.name in applicable]
    if not files:
        raise ValueError('No ACAS VNNLIB properties found')
    report = dict(run=str(args.run.resolve()), benchmark=str(args.benchmark.resolve()),
                  original=provenance, repaired_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
                  method=args.method, max_cells=args.max_cells, samples_per_box=args.samples,
                  seed=args.seed, device=args.device, alpha_iterations=args.alpha_iterations,
                  assigned_only=args.assigned_only,
                  instances_sha256=hashlib.sha256((args.benchmark / 'instances.csv').read_bytes()).hexdigest(),
                  source=source_provenance(), properties=[])
    previous = None
    if args.cover_from is not None:
        previous = json.loads(args.cover_from.read_text())
        if (previous['original']['sha256'] != provenance['sha256'] or
                previous['repaired_sha256'] != report['repaired_sha256']):
            raise ValueError('Saved covers refer to different model artifacts')
        report['cover_from'] = str(args.cover_from.resolve())
        report['cover_sha256'] = hashlib.sha256(args.cover_from.read_bytes()).hexdigest()
    for path in files:
        spec = load_vnnlib_property(path, input_shape=shape)
        inputs = sample_inputs(spec, reference, args.samples, args.seed)
        saved = None if previous is None else next(entry for entry in previous['properties']
            if entry['sha256'] == hashlib.sha256(path.read_bytes()).hexdigest())
        before, output_before, safe_before = evaluate(original, spec, inputs, args.method, args.max_cells,
            args.alpha_iterations, None if saved is None else saved['before']['certificate'])
        after, output_after, safe_after = evaluate(repaired, spec, inputs, args.method, args.max_cells,
            args.alpha_iterations, None if saved is None else saved['after']['certificate'])
        entry = dict(property=spec.name, benchmark_applicable=path.name in applicable,
                     source=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                     domains=len(spec.domains), before=before, after=after,
                     newly_violating_inputs=int((safe_before & ~safe_after).sum()),
                     fixed_inputs=int((~safe_before & safe_after).sum()),
                     advisory_agreement=float((output_before.argmin(1) == output_after.argmin(1)).float().mean()),
                     score_mse=float(nn.functional.mse_loss(output_after, output_before)))
        report['properties'].append(entry)
        (directory / 'results.json').write_text(json.dumps(report, indent=2, allow_nan=False))
        print(f'{spec.name} applicable={entry["benchmark_applicable"]}: '
              f'{before["outcome"]} -> {after["outcome"]}; sampled violations '
              f'{before["sampled_violations"]} -> {after["sampled_violations"]} / {len(inputs)}; '
              f'advisory agreement={entry["advisory_agreement"]:.3%}', flush=True)
    lines = ['| Property | Assigned to network | Before | After | Sample violations before/after | Advisory agreement |',
             '|---|---|---|---|---|---|']
    for entry in report['properties']:
        before, after = entry['before'], entry['after']
        lines.append(f'| {entry["property"]} | {entry["benchmark_applicable"]} | '
                     f'{before["outcome"]} | {after["outcome"]} | '
                     f'{before["sampled_violations"]}/{after["sampled_violations"]} '
                     f'of {before["sampled_inputs"]} | {entry["advisory_agreement"]:.2%} |')
    lines += ['', 'Unknown means unresolved within the bound/refinement budget; sampling is not a proof.',
              'Unassigned properties are diagnostic tests, not benchmark obligations for this network.',
              'Sampling weights original boxes equally, including overlapping boxes. Certificates cover every box.',
              'Numerical scope: floating-point PyTorch/LiRPA, without a directed-rounding arithmetic proof.']
    (directory / 'summary.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
