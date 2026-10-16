"""A small controller trained and certified through CTRAIN's property API."""
import argparse
import heapq
import json
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from CTRAIN.model_wrappers.certified import PropertyModelWrapper
from CTRAIN.properties.training import TRAINING_METHODS, resolve_training_options
from CTRAIN.properties import Outputs, And, Or, Box, PropertySpec
from CTRAIN.model_definitions.models_properties import ControllerMLP, ControllerDynamics
from case_studies.common import source_provenance


def state_boxes():
    return [Box([.3, -1], [1, 1]), Box([-1, -1], [-.3, 1]),
            Box([-1, .3], [1, 1]), Box([-1, -1], [1, -.3])]


def controller_property():
    y = Outputs()
    formula = And(y[0] >= -.1, y[0] <= .1, y[1] >= -.1, y[1] <= .1,
                  y[2] >= -1, y[2] <= 1, y[3] >= -1, y[3] <= 1,
                  Or(y[2] >= .3, y[2] <= -.3, y[3] >= .3, y[3] <= -.3))
    domains = [Box(torch.cat((b.lower, torch.tensor([-.01, -.01]))),
                   torch.cat((b.upper, torch.tensor([.01, .01])))) for b in state_boxes()]
    return PropertySpec('robust_invariant_ring', formula, domains)


def segment_safe(a, b):
    entry, exit = 0., 1.
    for coordinate in range(2):
        delta = float(b[coordinate] - a[coordinate])
        if abs(delta) < 1e-12:
            if abs(float(a[coordinate])) >= .3:
                return True
        else:
            t1, t2 = ((-.3 - float(a[coordinate])) / delta,
                      (.3 - float(a[coordinate])) / delta)
            entry, exit = max(entry, min(t1, t2)), min(exit, max(t1, t2))
    return entry >= exit - 1e-10


def waypoint(position):
    points = [position, position.new_tensor([.75, .75])]
    points += [position.new_tensor([a, b]) for a, b in ((-.35, -.35), (-.35, .35), (.35, -.35), (.35, .35))]
    distances, parents, queue = {0: 0.}, {}, [(0., 0)]
    while queue:
        cost, i = heapq.heappop(queue)
        if cost != distances[i]:
            continue
        if i == 1:
            while parents[i] != 0:
                i = parents[i]
            return points[i]
        for j, point in enumerate(points):
            if j != i and segment_safe(points[i], point):
                candidate = cost + float((points[i] - point).norm())
                if candidate < distances.get(j, float('inf')):
                    distances[j], parents[j] = candidate, i
                    heapq.heappush(queue, (candidate, j))
    raise ValueError('No safe waypoint path')


def teacher(data, mode='waypoint'):
    if mode == 'safety':
        # A simple continuous reference for the certification-focused run.
        return .04 * (data / .3).clamp(-1, 1) - .06 * data
    actions = []
    for position in data:
        direction = waypoint(position) - position
        preferred = position + direction * min(1., .08 / max(float(direction.norm()), 1e-12))
        candidates = []
        for box in state_boxes():
            lower = torch.maximum(box.lower.to(position) + .0101, position - .0999)
            upper = torch.minimum(box.upper.to(position) - .0101, position + .0999)
            if (lower <= upper).all():
                candidates.append(torch.maximum(lower, torch.minimum(upper, preferred)))
        if not candidates:
            raise ValueError('Teacher has no disturbance-safe reachable successor')
        successor = min(candidates, key=lambda p: float((p - preferred).square().sum()))
        actions.append(successor - position)
    return torch.stack(actions)


def dataset(count, seed, mode):
    generator = torch.Generator().manual_seed(seed)
    boxes = state_boxes()
    data = []
    for i in range(count):
        box = boxes[i % len(boxes)]
        data.append((box.lower + torch.rand(2, generator=generator, dtype=torch.double)
                     * (box.upper - box.lower)).float())
    data = torch.stack(data)
    targets = teacher(data, mode)
    successor = data + targets
    assert (targets.abs() <= .1 + 1e-7).all()
    assert (successor.abs() <= .99 + 1e-6).all()
    assert (successor.abs().amax(dim=1) >= .31 - 1e-6).all()
    return TensorDataset(data, targets)


def rollouts(model, data, seed=3, steps=100):
    generator = torch.Generator().manual_seed(seed)
    position = data.clone()
    safe = torch.ones(len(data), dtype=torch.bool)
    reached = torch.zeros_like(safe)
    with torch.no_grad():
        for _ in range(steps):
            disturbance = (torch.rand(position.shape, generator=generator) * 2 - 1) * .01
            position = position + model(position) + disturbance
            safe &= (position.abs() <= 1).all(dim=1) & (position.abs().amax(dim=1) >= .3)
            reached |= (position - position.new_tensor([.75, .75])).norm(dim=1) < .1
    return dict(empirical_safe_fraction=float(safe.float().mean()),
                empirical_goal_reached_fraction=float(reached.float().mean()), steps=steps)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('/tmp/ctrain-property-controller'))
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--warm-up-epochs', type=int, default=0)
    parser.add_argument('--weight-ramp-epochs', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--teacher', choices=('waypoint', 'safety'), default='waypoint')
    parser.add_argument('--method', choices=TRAINING_METHODS, default='CROWN-IBP')
    parser.add_argument('--training-options', type=json.loads, default={},
                        help='JSON parameters for the selected training method')
    parser.add_argument('--complete', action='store_true')
    parser.add_argument('--max-cells', type=int, default=1024)
    args = parser.parse_args()
    budget = 2 if args.smoke else args.epochs
    if (args.warm_up_epochs < 0 or args.weight_ramp_epochs < 0
            or args.warm_up_epochs + args.weight_ramp_epochs > budget):
        parser.error('warm-up and weight ramp must be nonnegative and fit the epoch budget')
    try:
        resolve_training_options(args.method, args.training_options)
    except ValueError as error:
        parser.error(str(error))
    if args.epochs < 1 or args.max_cells < 1:
        parser.error('epochs and max-cells must be positive')
    spec = controller_property()
    config = dict(case='controller', seed=args.seed, teacher=args.teacher, method=args.method,
                  warm_up_epochs=args.warm_up_epochs, weight_ramp_epochs=args.weight_ramp_epochs,
                  epochs=2 if args.smoke else args.epochs, smoke=args.smoke, training_options=args.training_options, properties=[spec.to_dict()])
    if args.dry_run:
        print(json.dumps(config, indent=2)); return
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1); torch.manual_seed(args.seed)
    training = dataset(64 if args.smoke else 1024, args.seed, args.teacher)
    validation = dataset(32 if args.smoke else 256, args.seed + 1, args.teacher)
    model = ControllerMLP(coordinate_features=True)
    wrapper = PropertyModelWrapper(model, (2,), properties=[spec], task_loss=nn.MSELoss(),
        num_epochs=config['epochs'], verification_model=ControllerDynamics(model),
        verification_input_shape=(4,), device=args.device,
        training_method=args.method, training_options=args.training_options,
        warm_up_epochs=args.warm_up_epochs, weight_ramp_epochs=args.weight_ramp_epochs,
        property_weight=20., refinement=True, property_batch_size=16,
        max_cells=args.max_cells, checkpoint_save_path=args.output / 'checkpoints')
    config.update(training=wrapper.configuration, source=source_provenance())
    (args.output / 'config.json').write_text(json.dumps(config, indent=2))
    initial = wrapper.certify_properties(refine=False)
    wrapper.train_model(DataLoader(training, batch_size=64), DataLoader(validation, batch_size=64))
    result = wrapper.certify_properties(method=args.method if args.method in ('IBP', 'CROWN-IBP', 'CROWN') else 'CROWN', refine=not args.smoke, complete=args.complete,
        results_path=args.output / 'verification')
    with torch.no_grad():
        model.cpu().eval()
        error = float(nn.functional.mse_loss(model(validation.tensors[0]), validation.tensors[1]))
    metrics = rollouts(model, validation.tensors[0][:32])
    metrics.update(held_out_mse=error, initial_certified=initial['certified'],
                   final_certified=result['certified'])
    torch.save(model.state_dict(), args.output / 'model.pt')
    (args.output / 'results.json').write_text(json.dumps(dict(metrics=metrics,
        history=wrapper.history, verification=result), indent=2, allow_nan=False))
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
