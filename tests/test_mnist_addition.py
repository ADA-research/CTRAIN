"""Full MNIST coverage, normalized boxes, independent CE heads and argmax logic."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile

import torch
from torch import nn
from torch.utils.data import DataLoader

from CTRAIN import PropertyModelWrapper
from CTRAIN.bound.properties import PropertyBounds
from CTRAIN.data_loaders.data_loaders import MNIST_MEAN, MNIST_STD
from CTRAIN.model_definitions import MNISTAdditionClassifier
from CTRAIN.properties import Box, Outputs, PropertySpec, _AffineFormula, load_vnnlib_property
from CTRAIN.train.certified.losses.properties import property_loss, mnist_addition_cross_entropy

from case_studies.mnist_addition.run import main as run_benchmark, pixel_box, write_vnnlib, certify
from case_studies.mnist_addition.data import paired_mnist_dataset


def main():
    torch.set_num_threads(1)
    images = torch.arange(10, dtype=torch.uint8)[:, None, None].expand(10, 28, 28).clone()
    labels = torch.arange(10)
    normalized, covered = paired_mnist_dataset(images, labels, 15, 3, cover_all=True)
    repeated, repeated_indices = paired_mnist_dataset(images, labels, 15, 3, cover_all=True)
    assert torch.equal(covered, repeated_indices)
    assert torch.equal(normalized.tensors[0], repeated.tensors[0])
    for side in range(2):
        assert torch.equal(covered[:10, side].sort().values, torch.arange(10))
    torch.testing.assert_close(normalized.tensors[0][:, :, 0, 0],
                               (covered.float() / 255 - MNIST_MEAN) / MNIST_STD)
    targets = normalized.tensors[1]
    assert targets.dtype == torch.long
    assert torch.equal(targets[:, :2], labels[covered])
    assert torch.equal(targets[:, 2], targets[:, :2].sum(dim=1))
    try:
        paired_mnist_dataset(images, labels, 9, 3, cover_all=True)
    except ValueError:
        pass
    else:
        raise AssertionError('Incomplete training coverage was accepted')

    classifier = MNISTAdditionClassifier(8)
    logits = classifier(normalized.tensors[0])
    ce = mnist_addition_cross_entropy(logits, targets)
    expected_ce = sum(nn.functional.cross_entropy(head, targets[:, i])
                      for i, head in enumerate(logits.split((10, 10, 19), dim=1))) / 3
    torch.testing.assert_close(ce, expected_ce)
    ce.backward()
    for gradient in classifier[-1].weight.grad.split((10, 10, 19), dim=0):
        assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    before = classifier(normalized.tensors[0]).detach()
    with torch.no_grad():
        classifier[-1].bias[20:] += 1
    after = classifier(normalized.tensors[0])
    torch.testing.assert_close(after[:, :20], before[:, :20])
    torch.testing.assert_close(after[:, 20:], before[:, 20:] + 1)

    pair = torch.zeros(2, 28, 28, dtype=torch.double)
    pair[0, 0, 0], pair[1, 0, 0] = 1., .5
    box = pixel_box(pair, .2)
    torch.testing.assert_close(box.lower[0, 0, 0], torch.tensor((.8-MNIST_MEAN)/MNIST_STD, dtype=torch.double))
    torch.testing.assert_close(box.upper[0, 0, 0], torch.tensor((1-MNIST_MEAN)/MNIST_STD, dtype=torch.double))
    torch.testing.assert_close(box.lower[1, 0, 0], torch.tensor((.3-MNIST_MEAN)/MNIST_STD, dtype=torch.double))
    torch.testing.assert_close(box.upper[1, 0, 0], torch.tensor((.7-MNIST_MEAN)/MNIST_STD, dtype=torch.double))
    torch.testing.assert_close(box.lower[0, 0, 1], torch.tensor(-MNIST_MEAN/MNIST_STD, dtype=torch.double))

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        path = root / 'property.vnnlib'
        write_vnnlib(path, box)
        spec = load_vnnlib_property(path, input_shape=(2, 28, 28))
        torch.testing.assert_close(spec.domains[0].lower, box.lower)
        torch.testing.assert_close(spec.domains[0].upper, box.upper)
        # Exhaust all possible unique winning triples, then exercise ties.
        triples = torch.cartesian_prod(torch.arange(10), torch.arange(10), torch.arange(19))
        outputs = torch.full((len(triples), 39), -1.)
        rows = torch.arange(len(triples))
        for column, start in enumerate((0, 10, 20)):
            outputs[rows, start + triples[:, column]] = 1.
        expected = triples[:, 2] == triples[:, 0] + triples[:, 1]
        assert torch.equal(spec.formula.evaluate(outputs).flatten(), expected)
        tied = torch.randint(0, 3, (300, 39), generator=torch.Generator().manual_seed(5)).float()
        winners = torch.stack([head.argmax(1) for head in tied.split((10, 10, 19), 1)], 1)
        assert torch.equal(spec.formula.evaluate(tied).flatten(), winners[:, 2] == winners[:, 0] + winners[:, 1])
        assert spec.formula.evaluate(torch.zeros(1, 39)).item()

        # Compilation shares repeated winner conditions, retaining exact margins.
        compiled = _AffineFormula(spec.formula, 39, next(classifier.parameters()))
        assert compiled.weight.shape == (522, 39)
        probes = normalized.tensors[0][:4]
        prediction = classifier(probes)
        atomic = nn.functional.linear(prediction, compiled.weight, compiled.bias)
        torch.testing.assert_close(compiled.reduce(atomic), spec.formula.margin(prediction), atol=1e-6, rtol=1e-6)

        # Wrong digit predictions may still be arithmetically consistent.
        # A constant (2,3,5) classifier proves consistency throughout the box.
        model = nn.Sequential(nn.Flatten(), nn.Linear(1568, 39))
        with torch.no_grad():
            model[-1].weight.zero_()
            model[-1].bias.fill_(-3.)
            model[-1].bias[[2, 13, 25]] = 3.
        bounds = PropertyBounds(model, (2, 28, 28), [spec], bound_opts={'property_bound': 'atomic'})
        for method in ('IBP', 'CROWN-IBP', 'CROWN'):
            lower = bounds(0, list(spec.domains), method)
            torch.testing.assert_close(lower, torch.tensor([[6.]]), atol=1e-5, rtol=1e-5)
        wrapper = PropertyModelWrapper(model, (2, 28, 28), properties=[spec],
            task_loss=mnist_addition_cross_entropy, num_epochs=1, property_batch_size=1,
            bound_opts={'property_bound': 'atomic'})
        report = certify(wrapper, spec, 'CROWN')
        assert report['certified'] and report['certified_count'] == 1
        heldout = PropertySpec('heldout', spec.formula, [Box([-1.] * 1568, [1.] * 1568)])
        # Reject a shape mismatch before computing evaluation bounds.
        try:
            wrapper.certify_properties(properties=[heldout], refine=False)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid held-out input shape accepted')
        heldout = PropertySpec('heldout', spec.formula, [Box(torch.zeros(2, 28, 28), torch.ones(2, 28, 28))])
        partition, configuration = wrapper.partition, json.dumps(wrapper.configuration)
        assert certify(wrapper, heldout, 'CROWN')['properties'] == [dict(name='heldout', certified=True)]
        wrong_sum = PropertySpec('wrong_sum', Outputs()[26] > Outputs()[25], heldout.domains)
        assert not wrapper.certify_properties(properties=[wrong_sum], refine=False)['certified']
        assert wrapper.partition is partition and json.dumps(wrapper.configuration) == configuration
        assert wrapper.properties[0].name == spec.name
        with torch.no_grad():
            model[-1].bias[26] = 3.25  # winning tuple (2,3,6) violates the relation
        assert not spec.formula.evaluate(model(probes[:1])).item()
        model.zero_grad(set_to_none=True)
        penalty = property_loss(bounds(0, list(spec.domains), 'CROWN-IBP'))
        penalty.backward()
        assert torch.isfinite(model[-1].weight.grad).all()
        assert model[-1].bias.grad[26] > 0 and model[-1].bias.grad[25] < 0
        assert not certify(wrapper, spec, 'CROWN')['certified']

        # Soundness on a box with changing winners: (0,0,0) switches to
        # (1,0,1). Atomic composition may be unknown despite true consistency.
        switching = nn.Linear(1, 39).double()
        with torch.no_grad():
            switching.weight.zero_()
            switching.bias.fill_(-10.)
            switching.bias[[0, 1, 10, 20, 21]] = 0.
            switching.weight[[0, 20], 0] = -1.
            switching.weight[[1, 21], 0] = 1.
        changing_spec = PropertySpec('changing', spec.formula, [Box([-1.], [1.])])
        changing_bounds = PropertyBounds(switching, (1,), [changing_spec],
                                        bound_opts={'property_bound': 'atomic'})
        points = torch.linspace(-1, 1, 101, dtype=torch.double).unsqueeze(1)
        assert spec.formula.evaluate(switching(points)).all()
        margins = spec.formula.margin(switching(points))
        for method in ('IBP', 'CROWN-IBP', 'CROWN'):
            lower = changing_bounds(0, changing_spec.domains, method)
            torch.testing.assert_close(lower, torch.tensor([[-2.]], dtype=torch.double))
            assert (margins >= lower).all()

        # A zero property weight must give the ordinary CE/Adam update.
        baseline, ordinary = MNISTAdditionClassifier(8), MNISTAdditionClassifier(8)
        ordinary.load_state_dict(baseline.state_dict())
        optimizer = torch.optim.Adam(ordinary.parameters(), lr=.001)
        optimizer.zero_grad()
        mnist_addition_cross_entropy(ordinary(normalized.tensors[0]), targets).backward()
        optimizer.step()
        wrapper = PropertyModelWrapper(baseline, (2, 28, 28), properties=[spec],
            task_loss=mnist_addition_cross_entropy, num_epochs=1, property_weight=0.,
            property_batch_size=1, bound_opts={'property_bound': 'atomic'})
        with contextlib.redirect_stdout(io.StringIO()):
            wrapper.train_model(DataLoader(normalized, batch_size=len(normalized)))
        for key, value in ordinary.state_dict().items():
            torch.testing.assert_close(baseline.state_dict()[key], value, rtol=0, atol=0)

        # Paired arms and CPU repeatability; no dataset downloads or strict determinism.
        for name in ('first', 'repeat'):
            with contextlib.redirect_stdout(io.StringIO()):
                run_benchmark(['--smoke', '--output', str(root / name)])
        first = json.loads((root / 'first/results.json').read_text())
        repeat = json.loads((root / 'repeat/results.json').read_text())
        assert first['smoke'] and set(first['results']) == {'ce_only', 'certified'}
        config = json.loads((root / 'first/config.json').read_text())
        assert 'tau' not in config
        assert config['normalization']['std'] == MNIST_STD
        assert config['data']['provenance']['train']['unique_images_per_position'] == [32, 32]
        for arm in first['results']:
            assert first['results'][arm]['test_metrics'] == repeat['results'][arm]['test_metrics']
            assert first['results'][arm]['test_certified_fraction'] == repeat['results'][arm]['test_certified_fraction']
            report = json.loads((root / f'first/{arm}/results.json').read_text())
            assert report['verification']['train']['total_boxes'] == 32
            assert report['verification']['test']['total_boxes'] == 16
            assert len(report['history']) == 2
            weights = torch.load(root / f'first/{arm}/model.pt', weights_only=True)
            again = torch.load(root / f'repeat/{arm}/model.pt', weights_only=True)
            assert all(torch.equal(weights[key], again[key]) for key in weights)
        with contextlib.redirect_stdout(io.StringIO()):
            run_benchmark(['--smoke', '--skip-ce-baseline', '--output', str(root / 'property_only')])
        property_only = json.loads((root / 'property_only/results.json').read_text())
        assert set(property_only['results']) == {'certified'}
        assert not (root / 'property_only/ce_only').exists()
        skipped = torch.load(root / 'property_only/certified/model.pt', weights_only=True)
        paired = torch.load(root / 'first/certified/model.pt', weights_only=True)
        assert all(torch.equal(skipped[key], paired[key]) for key in paired)
        parameters = root / 'parameters.json'
        settings = dict(seed=5, epochs=3, weight_ramp_epochs=1, train_pairs=60000,
                        method='CROWN', download=False, output=str(root / 'configured'))
        parameters.write_text(json.dumps(settings))
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            run_benchmark(['--config', str(parameters), '--seed', '11', '--dry-run'])
        resolved = json.loads(output.getvalue())
        assert resolved['seed'] == 11 and resolved['epochs'] == 3 and resolved['method'] == 'CROWN'
        assert not (root / 'configured').exists()
        with contextlib.redirect_stdout(io.StringIO()):
            run_benchmark(['--config', str(parameters), '--smoke', '--seed', '11'])
        for arm in ('ce_only', 'certified'):
            training = json.loads((root / f'configured/{arm}/training_config.json').read_text())
            assert training['weight_ramp_epochs'] == 1
        for invalid in ([], dict(unknown=1), dict(epochs=1.5), dict(method='invalid'),
                        dict(download='false'), dict(tau=.25), dict(train_pairs=59999),
                        dict(epochs=3, weight_ramp_epochs=4)):
            parameters.write_text(json.dumps(invalid))
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                try:
                    run_benchmark(['--config', str(parameters), '--dry-run'])
                except SystemExit as error:
                    assert error.code == 2
                else:
                    raise AssertionError(f'Invalid config accepted: {invalid}')
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            try:
                run_benchmark(['--smoke', '--output', str(root / 'first')])
            except SystemExit as error:
                assert error.code == 2
            else:
                raise AssertionError('Existing run was not rejected')
    print('MNIST classification, full coverage and coupled argmax verification checks passed.')


if __name__ == '__main__':
    main()
