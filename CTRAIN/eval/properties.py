"""Cover-preserving refinement and fresh whole-domain property certification."""
import math
import time
from fractions import Fraction
from pathlib import Path
import torch
from CTRAIN.properties import Box, PropertyGraph, PropertySpec
from CTRAIN.util.util import preserve_model_state, export_onnx


def initial_partition(specs):
    count = sum(len(s.domains) for s in specs)
    return [dict(property=p, domain=d, path='', depth=0, probability=1 / count, box=box)
            for p, spec in enumerate(specs) for d, box in enumerate(spec.domains)]


def certify_property_dataset(wrapper, method='CROWN', results_path=None):
    """Fresh full-box bounds over every region of a lazy property dataset.

    Return aggregate results without materializing all box endpoints or changing
    the random sampling stream. Detailed per-cell/refinement reports use the
    ordinary explicit-property API.
    """
    start = time.monotonic()
    report = evaluate_property_dataset(wrapper.bounds, wrapper.property_dataset, method,
                                       wrapper.property_batch_size)
    report.update(properties=[dict(name=wrapper.properties[0].name, certified=report['certified'])],
                  property_dataset=wrapper.configuration['property_dataset'],
                  method=method, seconds=time.monotonic() - start)
    if results_path is not None:
        import json
        directory = Path(results_path)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / 'results.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    return report


def evaluate_property_dataset(bounds, dataset, method='CROWN', batch_size=32):
    """Aggregate fresh bounds over a lazy box pool without consuming RNG state."""
    from CTRAIN.bound.properties import CERTIFICATION_METHODS
    if method not in CERTIFICATION_METHODS:
        raise ValueError('Invalid certification method')
    if type(batch_size) is not int or batch_size < 1 or not len(dataset):
        raise ValueError('A nonempty dataset and positive batch_size are required')
    count, certified_count, minimum = len(dataset), 0, float('inf')
    with preserve_model_state(bounds.model, *bounds.modules), torch.no_grad():
        bounds.model.eval()
        for offset in range(0, count, batch_size):
            boxes = [dataset[i] for i in range(offset, min(offset + batch_size, count))]
            lower = bounds(0, boxes, method).detach()
            finite = torch.isfinite(lower).all(dim=1)
            certified_count += int((finite & (lower > 0).all(dim=1)).sum())
            minimum = min(minimum, float(lower.min()) if finite.all() else -float('inf'))
    return dict(certified=certified_count == count, certified_count=certified_count,
                  total_boxes=count, certified_fraction=certified_count / count,
                  minimum_lower_bound=minimum if math.isfinite(minimum) else None)


def partition_state(cells):
    return [dict(c, box=c['box'].to_dict()) for c in cells]


def restore_partition(state):
    return [dict(c, box=Box(**c['box'])) for c in state]


def validate_partition(cells, specs):
    """Check saved dyadic leaves really cover every original region."""
    roots = [(p, d) for p, spec in enumerate(specs) for d in range(len(spec.domains))]
    if not cells or any((c['property'], c['domain']) not in roots for c in cells):
        raise ValueError('Invalid partition roots')
    for p, d in roots:
        leaves = [c for c in cells if (c['property'], c['domain']) == (p, d)]
        paths = [c['path'] for c in leaves]
        if not paths or len(set(paths)) != len(paths) or any(
                a != b and b.startswith(a) for a in paths for b in paths) or sum(
                (Fraction(1, 2 ** len(path)) for path in paths), Fraction()) != 1:
            raise ValueError('Partition does not cover an original domain exactly')
        root = specs[p].domains[d]
        for cell in leaves:
            path = cell['path']
            if any(bit not in '01' for bit in path) or cell['depth'] != len(path) or not math.isclose(
                    cell['probability'], 1 / len(roots) / 2 ** len(path), rel_tol=1e-12, abs_tol=0):
                raise ValueError('Invalid partition depth/probability')
            lower, upper = root.lower.clone(), root.upper.clone()
            for bit in path:
                width = root.upper - root.lower
                ratio = torch.where(width > 0, (upper - lower) / width.clamp_min(torch.finfo(torch.float64).tiny), 0.)
                dim = int(ratio.flatten().argmax())
                mid = (lower.flatten()[dim] + upper.flatten()[dim]) / 2
                if not lower.flatten()[dim] < mid < upper.flatten()[dim]:
                    raise ValueError('Invalid partition split')
                (upper if bit == '0' else lower).flatten()[dim] = mid
            if not torch.equal(cell['box'].lower, lower) or not torch.equal(cell['box'].upper, upper):
                raise ValueError('Partition endpoints do not match their declared lineage')


def evaluate_cells(bounds, cells, method, batch_size=32):
    start = time.monotonic()
    rows = [None] * len(cells)
    with torch.no_grad():
        for p in range(len(bounds.specs)):
            indices = [i for i, c in enumerate(cells) if c['property'] == p]
            for offset in range(0, len(indices), batch_size):
                chunk = indices[offset:offset + batch_size]
                lower = bounds(p, [cells[i]['box'] for i in chunk], method).detach().cpu()
                for i, value in zip(chunk, lower):
                    values = value.tolist()
                    certified = all(math.isfinite(v) and v > 0 for v in values)
                    rows[i] = dict(property=bounds.specs[p].name,
                                   id=f"{p}:{cells[i]['domain']}:{cells[i]['path']}",
                                   domain=cells[i]['box'].to_dict(),
                                   lower_bounds=[v if math.isfinite(v) else None for v in values],
                                   minimum_lower_bound=min(values) if all(math.isfinite(v) for v in values) else None,
                                   outcome='certified' if certified else 'unknown',
                                   graph='pytorch-lirpa')
    return rows, time.monotonic() - start


def refine_partition(cells, rows, specs, split_count=8, max_cells=1024, max_depth=12):
    counts = {}
    for cell in cells:
        key = cell['property'], cell['domain']
        counts[key] = counts.get(key, 0) + 1
    candidates = sorted(range(len(cells)), key=lambda i: (
        rows[i]['minimum_lower_bound'] if rows[i]['minimum_lower_bound'] is not None else -float('inf'), i))
    replacements = {}
    for index in candidates:
        cell = cells[index]
        key = cell['property'], cell['domain']
        if rows[index]['outcome'] == 'certified' or cell['depth'] >= max_depth or counts[key] >= max_cells:
            continue
        root = specs[key[0]].domains[key[1]]
        width = cell['box'].upper - cell['box'].lower
        root_width = root.upper - root.lower
        ratio = torch.where(root_width > 0, width / root_width.clamp_min(torch.finfo(torch.float64).tiny), 0.)
        dimension = int(ratio.flatten().argmax())
        if not ratio.flatten()[dimension] > 0:
            continue
        lower, upper = cell['box'].lower.clone(), cell['box'].upper.clone()
        midpoint = (lower.flatten()[dimension] + upper.flatten()[dimension]) / 2
        if not lower.flatten()[dimension] < midpoint < upper.flatten()[dimension]:
            continue
        left_upper, right_lower = upper.clone(), lower.clone()
        left_upper.flatten()[dimension] = midpoint
        right_lower.flatten()[dimension] = midpoint
        replacements[index] = [dict(cell, box=box, path=cell['path'] + str(side),
                                    depth=cell['depth'] + 1, probability=cell['probability'] / 2)
                               for side, box in enumerate((Box(lower, left_upper), Box(right_lower, upper)))]
        counts[key] += 1
        if len(replacements) >= split_count:
            break
    return [child for i, cell in enumerate(cells) for child in replacements.get(i, [cell])]


def find_counterexample(model, formula, box, steps=30):
    reference = next(model.parameters(), None)
    if reference is None:
        reference = next(model.buffers(), torch.zeros(()))
    lower, upper = box.lower.to(reference), box.upper.to(reference)
    # Attack points must be inside the declared box, not its outward dtype enclosure.
    lower = torch.where(lower.double() < box.lower.to(lower.device),
                        torch.nextafter(lower, torch.full_like(lower, float('inf'))), lower)
    upper = torch.where(upper.double() > box.upper.to(upper.device),
                        torch.nextafter(upper, torch.full_like(upper, -float('inf'))), upper)
    if (lower > upper).any():
        return None
    with preserve_model_state(model), torch.enable_grad():
        model.eval()
        point = ((lower + upper) / 2).unsqueeze(0)
        for _ in range(steps + 1):
            point = point.detach().requires_grad_(True)
            output = model(point)
            if not formula.evaluate(output).all():
                return point.detach().cpu()[0].tolist()
            margin = formula.margin(output)
            if not margin.requires_grad:
                break
            gradient = torch.autograd.grad(margin.sum(), point, allow_unused=True)[0]
            if gradient is None:
                break
            point = torch.maximum(lower, torch.minimum(upper,
                point - .1 * (upper - lower) * gradient.sign()))
    return None


def export_property_graph(graph, path, input_shape, probes):
    """Resolve constant aliases for the installed verifier's ONNX loader."""
    import onnx
    import onnxoptimizer
    import onnxruntime as ort
    with preserve_model_state(graph), torch.no_grad():
        graph.eval()
        export_onnx(graph, str(path), 1, input_shape)
        # PyTorch deduplicates equal coefficients using Identity nodes. The
        # bundled onnx2pytorch loader requires Gemm coefficients as initializers.
        exported = onnxoptimizer.optimize(onnx.load(path),
            ['extract_constant_to_initializer', 'eliminate_identity'])
        onnx.checker.check_model(exported)
        onnx.save(exported, path)
        session = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
        actual = session.run(None, {session.get_inputs()[0].name: probes.cpu().numpy()})[0]
        torch.testing.assert_close(torch.from_numpy(actual), graph(probes).detach().cpu(),
                                   rtol=1e-4, atol=1e-5)


def complete_cells(wrapper, cells, rows, directory, timeout, properties=None):
    from CTRAIN.complete_verification.abCROWN.verify import limited_abcrown_eval
    from CTRAIN.complete_verification.abCROWN.util import get_abcrown_standard_conf
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    exported = {}
    for cell, row in zip(cells, rows):
        if row['outcome'] == 'certified':
            continue
        p = cell['property']
        spec = (wrapper.properties if properties is None else properties)[p]
        witness = find_counterexample(wrapper.verification_model, spec.formula, cell['box'])
        if witness is not None:
            row.update(outcome='violated', counterexample=witness)
            continue
        if p not in exported:
            graph = PropertyGraph(wrapper.verification_model, spec.formula, relu=True,
                                  output_width=wrapper.bounds.width)
            path = directory / f'property_{p}.onnx'
            with preserve_model_state(graph), torch.no_grad():
                graph.eval()
                all_probes = []
                for box in spec.domains:
                    lower, upper = box.lower.to(wrapper.reference), box.upper.to(wrapper.reference)
                    probes = torch.stack((lower, upper, (lower + upper) / 2))
                    torch.testing.assert_close(graph(probes), spec.formula.margin(
                        wrapper.verification_model(probes)), rtol=1e-5, atol=1e-6)
                    all_probes.append(probes)
                export_property_graph(graph, path, wrapper.verification_input_shape, torch.cat(all_probes))
            exported[p] = path
        filename = directory / (row['id'].replace(':', '_') + '.vnnlib')
        lower, upper = cell['box'].lower.flatten(), cell['box'].upper.flatten()
        lines = [f'(declare-const X_{i} Real)' for i in range(lower.numel())]
        lines += ['(declare-const Y_0 Real)']
        for i, (lo, hi) in enumerate(zip(lower.tolist(), upper.tolist())):
            lines += [f'(assert (>= X_{i} {lo:.17g}))', f'(assert (<= X_{i} {hi:.17g}))']
        lines += ['(assert (<= Y_0 0))']
        filename.write_text('\n'.join(lines) + '\n')
        config = get_abcrown_standard_conf(timeout=timeout, no_cores=1)
        config['data']['num_outputs'] = 1
        config['general']['double_fp'] = wrapper.reference.dtype == torch.float64
        elapsed, status = limited_abcrown_eval(
            config=config, seed=0,
            instance=str(filename.resolve()), model_name=None,
            model_onnx_path=str(exported[p].resolve()),
            input_shape=[1, *wrapper.verification_input_shape], timeout=timeout, no_cores=1)
        # The existing runner does not return a witness. SAT alone is not a
        # validated counterexample to the source graph.
        row.update(solver_status=status, solver_seconds=elapsed,
                   outcome='certified' if status == 'unsat' else ('timeout' if status == 'timeout' else 'unknown'))
        if row['outcome'] == 'certified':
            row['graph'] = 'onnx-relu'


def certify_properties(wrapper, method='CROWN', refine=True, complete=False,
                       max_cells=None, max_depth=None, results_path=None, timeout=60,
                       properties=None):
    """Recompute bounds on the training cover or on separate evaluation specs.

    Explicit ``properties`` start from their original boxes and use fresh bound
    graphs with the wrapper's options. Evaluation never changes training specs
    or the stored partition.
    """
    from CTRAIN.bound.properties import CERTIFICATION_METHODS
    if method not in CERTIFICATION_METHODS or type(refine) is not bool or type(complete) is not bool:
        raise ValueError('Invalid certification method or flags')
    if wrapper.property_dataset is not None and properties is None:
        if refine or complete:
            raise ValueError('Lazy property datasets support full-box bounds without refinement '
                             'or complete verification; supply explicit properties for those modes')
        return certify_property_dataset(wrapper, method, results_path)
    max_cells = wrapper.max_cells if max_cells is None else max_cells
    max_depth = wrapper.max_depth if max_depth is None else max_depth
    if type(max_cells) is not int or max_cells < 1 or type(max_depth) is not int or max_depth < 0:
        raise ValueError('Invalid refinement budget')
    if complete and (results_path is None or not math.isfinite(timeout) or timeout <= 0):
        raise ValueError('Complete verification requires results_path and positive finite timeout')
    start = time.monotonic()
    specs, bounds = wrapper.properties, wrapper.bounds
    cells = list(wrapper.partition)
    if properties is not None:
        from CTRAIN.bound.properties import PropertyBounds
        specs = tuple(properties)
        if (not specs or any(not isinstance(p, PropertySpec) for p in specs)
                or len({p.name for p in specs}) != len(specs)):
            raise ValueError('Evaluation properties require nonempty specs with unique names')
        bounds = PropertyBounds(wrapper.verification_model, wrapper.verification_input_shape,
                                specs, wrapper.configuration['bound_opts'] or None)
        cells = initial_partition(specs)
    validate_partition(cells, specs)
    while True:
        rows, _ = evaluate_cells(bounds, cells, method, wrapper.property_batch_size)
        children = refine_partition(cells, rows, specs, wrapper.split_count,
                                    max_cells, max_depth) if refine else cells
        if len(children) == len(cells):
            break
        cells = children
    if complete:
        complete_cells(wrapper, cells, rows, results_path, timeout, specs)
    summaries = [dict(name=spec.name, certified=all(r['outcome'] == 'certified'
                    for r in rows if r['property'] == spec.name)) for spec in specs]
    return dict(certified=all(p['certified'] for p in summaries), properties=summaries,
                cells=rows, method=method, seconds=time.monotonic() - start)
