"""Decompose saved arithmetic certificates and seek changed, consistent triples."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import torch


HEADS = (10, 10, 19)


def predictions(output):
    return torch.stack([head.argmax(1) for head in output.split(HEADS, 1)], 1)


def target_margin(output, targets):
    """Smallest winning margin over the requested heads; -1 ignores a head."""
    margins = []
    for h, head in enumerate(output.split(HEADS, 1)):
        selected = targets[:, h].clamp_min(0)
        competitors = head.scatter(1, selected[:, None], -torch.inf)
        margin = head.gather(1, selected[:, None]).squeeze(1) - competitors.amax(1)
        margins.append(torch.where(targets[:, h] >= 0, margin, torch.inf))
    return torch.stack(margins, 1).amin(1)


def search_targets(model, nominal, lower, upper, targets, steps, restarts, observe):
    """Projected targeted input search, observing every iterate, including starts."""
    for restart in range(restarts):
        point = nominal.clone() if restart == 0 else lower + torch.rand_like(lower) * (upper - lower)
        for step in range(steps + 1):
            point = point.detach().requires_grad_(True)
            output = model(point)
            observe(point.detach(), output.detach())
            if step == steps:
                break
            gradient, = torch.autograd.grad(target_margin(output, targets).sum(), point)
            point = (point + .1 * (upper - lower) * gradient.sign()).clamp(lower, upper)


def margin_rows(labels):
    rows, columns = [], []
    for h, (start, size) in enumerate(zip((0, 10, 20), HEADS)):
        for competitor in range(size):
            if competitor == int(labels[h]):
                continue
            row = labels.new_zeros(39, dtype=torch.float32)
            row[start + labels[h]], row[start + competitor] = 1, -1
            rows.append(row)
            columns.append((h, competitor))
    return torch.stack(rows), columns


def self_test():
    torch.manual_seed(3)
    # Two operand classes switch together with the sum: a genuine consistent change.
    class Consistent(torch.nn.Module):
        def forward(self, x):
            t = x.flatten(1)[:, 0]
            output = t.new_full((len(t), 39), -10.)
            output[:, 0], output[:, 1] = -t, t
            output[:, 10], output[:, 11] = -t, t
            output[:, 20], output[:, 22] = -t, t
            return output
    model = Consistent()
    nominal = torch.tensor([[-.5]])
    points = []
    search_targets(model, nominal, torch.tensor([[-1.]]), torch.tensor([[1.]]),
                   torch.tensor([[1, 1, 2]]), 20, 2,
                   lambda x, y: points.append((x.clone(), predictions(y))))
    assert any(p.tolist() == [[1, 1, 2]] for x, p in points)
    assert all((x >= -1).all() and (x <= 1).all() for x, p in points)
    output = model(torch.tensor([[-.5], [.5]]))
    assert predictions(output).tolist() == [[0, 0, 0], [1, 1, 2]]
    assert (target_margin(output[1:], torch.tensor([[1, 1, 2]])) > 0).all()
    assert (target_margin(output[1:], torch.tensor([[1, -1, -1]])) > 0).all()
    rows, columns = margin_rows(torch.tensor([0, 0, 0]))
    assert rows.shape == (36, 39) and len(columns) == 36
    assert float((output[:1] @ rows.T).amin()) == float(target_margin(output[:1], torch.tensor([[0, 0, 0]])))
    print('Synthetic changed-consistent witness, margins and box containment: PASS')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-root', type=Path, help='HPO root containing seed_N directories')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--trial', type=int, help='Selected trial with saved test verification')
    parser.add_argument('--data-root', type=Path, help='Override the saved MNIST cache location')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--restarts', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--limit', type=int, help='Execution check only; otherwise evaluate all saved test pairs')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.study_root is None or args.trial is None or min(args.seed, args.trial) < 0:
        parser.error('Supply --study-root, --trial and a nonnegative seed/trial')
    if args.output is None or min(args.steps, args.restarts, args.batch_size) < 1 or (args.limit is not None and args.limit < 1):
        parser.error('Require a fresh --output and positive search/batch/limit values')
    if args.output.exists():
        raise FileExistsError('Output must be fresh; existing experiments are preserved')
    root = args.study_root.resolve()
    from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
    from torchvision.datasets import MNIST
    from CTRAIN.bound.properties import box_endpoints
    from CTRAIN.data_loaders.data_loaders import MNIST_MEAN, MNIST_STD
    from CTRAIN.model_definitions import MNISTAdditionClassifier
    from CTRAIN.properties import Box
    source = Path(sys.modules['CTRAIN'].__file__).resolve().parent.parent

    started = time.monotonic()
    torch.set_num_threads(1)
    torch.manual_seed(0)
    seed_directory = root / f'seed_{args.seed}'
    directory = seed_directory / 'trials' / f'{args.trial:03d}'
    config_path, report_path = seed_directory / 'config.json', directory / 'test_verification.json'
    config = json.loads(config_path.read_text())
    saved = json.loads(report_path.read_text())
    record = json.loads((directory / 'results.json').read_text())
    assert saved['method'] == 'CROWN', 'This diagnostic requires a saved CROWN test cover'
    assert saved['certified_count'] == sum(cell['outcome'] == 'certified' for cell in saved['cells'])
    assert len(saved['cells']) == len(config['data']['test']['pair_indices'])
    data = MNIST(args.data_root or config['parameters']['data_root'], train=False, download=False)
    provenance = config['data']['test']
    for tensor, key in ((data.data, 'images_sha256'), (data.targets, 'labels_sha256')):
        assert hashlib.sha256(tensor.numpy().tobytes()).hexdigest() == provenance[key]
    indices = torch.tensor(provenance['pair_indices'])
    raw = data.data[indices].float() / 255
    nominal = ((raw - MNIST_MEAN) / MNIST_STD).to(args.device)
    operands = data.targets[indices]
    truth = torch.cat((operands, operands.sum(1, keepdim=True)), 1)
    model = MNISTAdditionClassifier(config['parameters']['width']).to(args.device).eval()
    model.load_state_dict(torch.load(directory / 'model.pt', map_location=args.device, weights_only=True))
    model.requires_grad_(False)
    with torch.no_grad():
        clean = predictions(model(nominal)).cpu()
    assert abs(float((clean == truth).all(1).float().mean()) - record['test']['metrics']['triple_accuracy']) < 1e-6
    assert abs(float((clean[:, 0] + clean[:, 1] == clean[:, 2]).float().mean()) - record['test']['metrics']['empirical_consistent_fraction']) < 1e-6
    count = len(nominal) if args.limit is None else min(args.limit, len(nominal))
    nominal, clean, truth, indices = nominal[:count], clean[:count], truth[:count], indices[:count]
    cells = saved['cells'][:count]
    assert [cell['id'] for cell in cells] == [f'0:{i}:' for i in range(count)]
    boxes = [Box(**cell['domain']) for cell in cells]
    reference = next(model.parameters())
    lower, upper = box_endpoints(boxes, reference)
    exact_lower = torch.stack([box.lower for box in boxes]).to(args.device)
    exact_upper = torch.stack([box.upper for box in boxes]).to(args.device)
    attack_lower, attack_upper = exact_lower.to(reference), exact_upper.to(reference)
    attack_lower = torch.where(attack_lower.double() < exact_lower, torch.nextafter(attack_lower, torch.full_like(attack_lower, torch.inf)), attack_lower)
    attack_upper = torch.where(attack_upper.double() > exact_upper, torch.nextafter(attack_upper, torch.full_like(attack_upper, -torch.inf)), attack_upper)
    # Loader rounding can put saturated pixels one ULP outside exact saved boxes.
    attack_nominal = nominal.clamp(attack_lower, attack_upper)
    with torch.no_grad():
        assert torch.equal(predictions(model(attack_nominal)).cpu(), clean)
    args.output.mkdir(parents=True, exist_ok=False)
    bounded = BoundedModule(model, nominal[:1], device=args.device,
                            bound_opts={'param': {'auto_requires_grad': False}})
    bounded.eval()
    margins = []
    with torch.no_grad():
        for start in range(0, count, args.batch_size):
            stop = min(start + args.batch_size, count)
            C = torch.stack([margin_rows(label)[0] for label in clean[start:stop]]).to(reference)
            ptb = PerturbationLpNorm(norm=float('inf'), x_L=lower[start:stop], x_U=upper[start:stop])
            lb, _ = bounded.compute_bounds(x=(BoundedTensor((lower[start:stop] + upper[start:stop]) / 2, ptb),),
                                           C=C, method='CROWN', bound_upper=False)
            margins.append(lb.cpu())
    margins = torch.cat(margins)
    assert torch.isfinite(margins).all(), 'Nonfinite CROWN margins require investigation'
    per_head = torch.stack([chunk.amin(1) > 0 for chunk in margins.split((9, 9, 18), 1)], 1)
    fixed = per_head.all(1)
    certified = torch.tensor([cell['outcome'] == 'certified' for cell in cells])
    print(f'CROWN: {int(fixed.sum())}/{count} fixed triples; {int((certified & fixed).sum())}/{int(certified.sum())} saved certificates explained by fixed predictions', flush=True)

    # Only targets not excluded by sound CROWN margins need input search.
    single_jobs, joint_jobs = [], []
    for i, label in enumerate(clean):
        allowed = [{int(value)} for value in label]
        for lb, (h, competitor) in zip(margins[i], margin_rows(label)[1]):
            if not torch.isfinite(lb) or lb <= 0:
                allowed[h].add(competitor)
                target = [-1, -1, -1]
                target[h] = competitor
                single_jobs.append((i, target))
        for a in sorted(allowed[0]):
            for b in sorted(allowed[1]):
                if a + b in allowed[2] and [a, b, a + b] != label.tolist():
                    joint_jobs.append((i, [a, b, a + b]))
    witnesses = {}
    for name, jobs in (('single_head', single_jobs), ('consistent_triple', joint_jobs)):
        print(f'Search {name}: {len(jobs)} targets, {args.steps} steps, {args.restarts} starts', flush=True)
        for start in range(0, len(jobs), args.batch_size):
            chunk = jobs[start:start + args.batch_size]
            ids = torch.tensor([i for i, target in chunk], device=args.device)
            targets = torch.tensor([target for i, target in chunk], device=args.device)
            base = clean[ids.cpu()].to(args.device)
            def observe(point, output):
                pred = predictions(output)
                changed = (pred != base).any(1)
                consistent = pred[:, 0] + pred[:, 1] == pred[:, 2]
                for j in changed.nonzero().flatten().tolist():
                    i = int(ids[j])
                    kind = 'changed_consistent' if bool(consistent[j]) else 'changed_inconsistent'
                    if (i, kind) not in witnesses:
                        witnesses[i, kind] = point[j].detach().cpu()
            search_targets(model, attack_nominal[ids], attack_lower[ids], attack_upper[ids],
                           targets, args.steps, args.restarts, observe)
            if start % (10 * args.batch_size) == 0:
                print(f'  {name}: {min(start + len(chunk), len(jobs))}/{len(jobs)} targets; {len(witnesses)} witnesses', flush=True)
    witness_rows = []
    for (i, kind), point in sorted(witnesses.items()):
        assert (point.double() >= exact_lower[i].cpu()).all() and (point.double() <= exact_upper[i].cpu()).all()
        with torch.no_grad():
            output = model(point[None].to(args.device))
            pred = predictions(output).cpu()[0]
        assert (pred != clean[i]).any()
        assert bool(pred[0] + pred[1] == pred[2]) == (kind == 'changed_consistent')
        witness_rows.append(dict(pair=i, kind=kind, nominal_prediction=clean[i].tolist(),
            changed_prediction=pred.tolist(), saved_consistency_certificate=bool(certified[i]),
            minimum_winning_margin=float(target_margin(output, pred[None].to(args.device))),
            raw_pixel_linf=float(((point - nominal[i].cpu()).abs() * MNIST_STD).max())))
    rows = []
    for i in range(count):
        rows.append(dict(pair=i, source_indices=indices[i].tolist(), target=truth[i].tolist(),
            nominal_prediction=clean[i].tolist(), empirical_consistent=bool(clean[i, 0] + clean[i, 1] == clean[i, 2]),
            saved_consistency_certificate=bool(certified[i]), crown_fixed_heads=per_head[i].tolist(),
            crown_minimum_margins=[float(chunk.amin()) for chunk in margins[i].split((9, 9, 18))],
            crown_fixed_triple=bool(fixed[i]),
            witnessed_changed_consistent=(i, 'changed_consistent') in witnesses,
            witnessed_changed_inconsistent=(i, 'changed_inconsistent') in witnesses))
    summary = dict(total_pairs=count, saved_consistency_certified=int(certified.sum()),
        crown_fixed_triples=int(fixed.sum()), saved_certified_and_fixed=int((certified & fixed).sum()),
        crown_fixed_and_consistent=sum(row['crown_fixed_triple'] and row['empirical_consistent'] for row in rows),
        saved_certified_without_fixed_certificate=int((certified & ~fixed).sum()),
        changed_consistent=sum(row['witnessed_changed_consistent'] for row in rows),
        changed_consistent_from_consistent=sum(row['empirical_consistent'] and row['witnessed_changed_consistent'] for row in rows),
        saved_certified_with_consistent_change=sum(row['saved_consistency_certificate'] and row['witnessed_changed_consistent'] for row in rows),
        changed_inconsistent=sum(row['witnessed_changed_inconsistent'] for row in rows),
        saved_certified_with_any_change=sum(row['saved_consistency_certificate'] and
            (row['witnessed_changed_consistent'] or row['witnessed_changed_inconsistent']) for row in rows),
        unresolved_prediction_stability=sum(not row['crown_fixed_triple'] and
            not row['witnessed_changed_consistent'] and not row['witnessed_changed_inconsistent'] for row in rows))
    paths = [config_path, report_path, directory / 'model.pt', Path(__file__),
             source / 'CTRAIN/bound/properties.py', source / 'CTRAIN/properties/__init__.py',
             source / 'CTRAIN/model_definitions/models_properties.py']
    result = dict(summary=summary, pairs=rows, witnesses=witness_rows,
        provenance={str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        parameters={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        search_targets=dict(single_head=len(single_jobs), consistent_triple=len(joint_jobs)),
        seconds=time.monotonic() - started,
        certificate_interpretation='Saved max-of-mins affine-leaf certification proves one fixed consistent winning triple per certified box; it does not establish flexible consistent predictions.',
        limits='Floating-point CROWN bounds. Failed bounds and unsuccessful PGD are unresolved, not proof of prediction stability. A changed-consistent witness proves consistency at that point only, not throughout its box.')
    (args.output / 'results.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    torch.save(witnesses, args.output / 'witnesses.pt')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
