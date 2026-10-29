"""ACAS source conventions and library ONNX/verifier regressions."""
import copy
import hashlib
import io
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch
import torch
from torch import nn
from CTRAIN import PropertyModelWrapper
from CTRAIN.properties import Outputs, Or, Box, PropertySpec, PropertyGraph
from CTRAIN.eval.properties import complete_cells, initial_partition, evaluate_cells, export_property_graph
from CTRAIN.util.util import export_onnx
from CTRAIN.util import load_onnx_model
from CTRAIN.train.certified.losses.properties import advisory_cross_entropy
from CTRAIN.data_loaders.acasxu import acasxu_property, load_acasxu, download_acasxu, _crosscheck, FILES, SOURCE


def main():
    torch.set_num_threads(1)
    y = torch.tensor([[0., 1., 2., 3., 4.], [3., 3., 3., 0., 1.], [0.] * 5])
    labels = y.argmin(dim=1)
    assert labels.dtype == torch.long and labels.tolist() == [0, 3, 0]
    costs = y.clone().requires_grad_()
    loss = advisory_cross_entropy(costs, labels)
    torch.testing.assert_close(loss, nn.functional.cross_entropy(-costs, labels))
    # Common score offsets preserve the policy and its CE, unlike score MSE.
    torch.testing.assert_close(loss, advisory_cross_entropy(costs + 10, labels))
    assert loss < advisory_cross_entropy(-costs, labels)
    loss.backward()
    assert torch.isfinite(costs.grad).all()
    assert (costs.grad[torch.arange(len(labels)), labels] > 0).all()
    assert ((costs.detach() - .1 * costs.grad).gather(1, labels[:, None])
            < costs.detach().gather(1, labels[:, None])).all()
    box = Box([-1] * 5, [1] * 5)
    assert acasxu_property('7', box).formula.evaluate(y).flatten().tolist() == [True, False, False]
    assert acasxu_property('8', box).formula.evaluate(y).flatten().tolist() == [True, False, False]
    safe = ['or', ['and', *[['<=', 'Y_3', f'Y_{i}'] for i in range(5) if i != 3]],
                  ['and', *[['<=', 'Y_4', f'Y_{i}'] for i in range(5) if i != 4]]]
    _crosscheck('7', [safe])
    unsafe8 = ['or', *[
                           ['and', ['<=', f'Y_{i}', 'Y_0'], ['<=', f'Y_{i}', 'Y_1']]
                           for i in (2, 3, 4)]]
    _crosscheck('8', [unsafe8])
    symbol = Outputs()
    model = nn.Linear(1, 1)
    with torch.no_grad():
        model.weight.fill_(1); model.bias.zero_()
    spec = PropertySpec('overlap', Or(symbol[0] <= .2, symbol[0] >= -.2), [Box([-1], [1])])
    wrapper = PropertyModelWrapper(model, (1,), properties=[spec], task_loss=nn.MSELoss(), num_epochs=1)
    with tempfile.TemporaryDirectory() as directory:
        graph = PropertyGraph(model, spec.formula, relu=True)
        onnx_path = Path(directory) / 'property.onnx'
        probes = torch.linspace(-1, 1, 101).unsqueeze(1)
        export_property_graph(graph, onnx_path, (1,), probes)
        import onnxruntime as ort
        session = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
        torch.testing.assert_close(graph(probes), torch.from_numpy(session.run(None,
            {session.get_inputs()[0].name: probes.numpy()})[0]))
        # Exercise the installed verifier's ONNX converter without running a solver.
        import onnx, onnx2pytorch
        converted = onnx2pytorch.ConvertModel(onnx.load(onnx_path), experimental=True)
        torch.testing.assert_close(converted(probes), graph(probes))
        # Verify the library's affine/ReLU adapter on a sequential fixture.
        sequential = nn.Sequential(nn.Linear(5, 8), nn.ReLU(), nn.Linear(8, 5))
        path = Path(directory) / 'acas_fixture.onnx'
        export_onnx(sequential, str(path), 1, (5,))
        adapted = load_onnx_model(path, torch.ones(3, 5))
        torch.testing.assert_close(adapted(torch.ones(3, 5)), sequential(torch.ones(3, 5)))
        fixed_path = Path(directory) / 'fixed_batch.onnx'
        torch.onnx.export(sequential, torch.ones(2, 5), str(fixed_path), opset_version=17)
        fixed = load_onnx_model(fixed_path, torch.ones(3, 5))
        torch.testing.assert_close(fixed(torch.ones(3, 5)), sequential(torch.ones(3, 5)))
        # Fixture artifacts exercise the real loader without claiming released weights.
        def expression(item):
            return '(' + ' '.join(expression(v) if isinstance(v, list) else v for v in item) + ')'
        declarations = [f'(declare-const {prefix}_{i} Real)' for prefix in ('X', 'Y') for i in range(5)]
        constraints = [f'(assert ({op} X_{i} {v}))' for i in range(5) for op, v in (('>=', -1), ('<=', 1))]
        artifacts = Path(directory) / 'artifacts'
        artifacts.mkdir()
        for filename in FILES[:2]:
            (artifacts / filename).write_bytes(path.read_bytes())
        for case, predicate in (('7', safe), ('8', unsafe8)):
            (artifacts / f'prop_{case}.vnnlib').write_text('\n'.join(
                declarations + constraints + [f'(assert {expression(predicate)})']))
        payload = {filename: (artifacts / filename).read_bytes() for filename in FILES}
        revision = 'a' * 40
        # Explicit download pins every URL, returns a complete manifest, and refuses overwrite.
        urls = []
        def response(url, timeout):
            urls.append(url)
            return io.BytesIO(payload[url.rsplit('/', 1)[1]])
        downloaded = Path(directory) / 'downloaded'
        with patch('CTRAIN.data_loaders.acasxu.urlopen', side_effect=response):
            manifest = download_acasxu(downloaded, revision)
        assert len(urls) == 4 and all(f'/{revision}/' in url for url in urls)
        assert manifest['source'] == SOURCE
        def rejects(error, operation):
            try:
                operation()
            except error:
                return
            raise AssertionError(f'Expected {error.__name__}')
        rejects(FileExistsError, lambda: download_acasxu(downloaded, revision))
        rejects(ValueError, lambda: download_acasxu(Path(directory) / 'invalid', 'main'))
        for case in ('7', '8'):
            loaded, loaded_spec, metadata = load_acasxu(downloaded, case)
            assert metadata == manifest and loaded_spec.name == f'acasxu_{case}'
            torch.testing.assert_close(loaded(torch.ones(3, 5)), sequential(torch.ones(3, 5)))
        training = PropertyModelWrapper(loaded, (5,), properties=[loaded_spec],
            task_loss=advisory_cross_entropy, num_epochs=1, property_batch_size=2)
        for method in ('IBP', 'CROWN-IBP', 'CROWN'):
            loaded.zero_grad(set_to_none=True)
            (-training.bounds(0, loaded_spec.domains, method).sum()).backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in loaded.parameters())
        data = torch.zeros(4, 5)
        target = sequential(data).detach().argmin(dim=1)
        from torch.utils.data import DataLoader, TensorDataset
        training.train_model(DataLoader(TensorDataset(data, target), batch_size=4))
        # Hash failures are rejected before parsing/conversion.
        filename = downloaded / FILES[0]
        filename.write_bytes(filename.read_bytes() + b'corrupted')
        rejects(ValueError, lambda: load_acasxu(downloaded))
        filename.write_bytes(payload[FILES[0]])
        source = downloaded / 'prop_7.vnnlib'
        source.write_text('\n'.join(declarations + constraints + ['(assert (<= Y_0 Y_1))']))
        manifest['sha256'][source.name] = hashlib.sha256(source.read_bytes()).hexdigest()
        (downloaded / 'manifest.json').write_text(json.dumps(manifest))
        rejects(ValueError, lambda: load_acasxu(downloaded))
        rejects(ValueError, lambda: acasxu_property('9', box))
        rejects(ValueError, lambda: load_onnx_model(path, torch.zeros(3, 5, dtype=torch.float64)))
        with patch('onnx2pytorch.ConvertModel', return_value=nn.Linear(5, 5)):
            rejects(AssertionError, lambda: load_onnx_model(path, torch.ones(3, 5)))
        cells = initial_partition([spec])
        rows, _ = evaluate_cells(wrapper.bounds, cells, 'CROWN')
        assert rows[0]['outcome'] == 'unknown'
        for status, expected in (('unsat', 'certified'), ('sat', 'unknown'), ('timeout', 'timeout'), ('unknown', 'unknown')):
            outcome = copy.deepcopy(rows)
            with patch('CTRAIN.complete_verification.abCROWN.verify.limited_abcrown_eval', return_value=(.1, status)):
                complete_cells(wrapper, cells, outcome, directory, 10)
            assert outcome[0]['outcome'] == expected
        violated = PropertySpec('unsafe', symbol[0] >= .5, [Box([-1], [1])])
        bad = PropertyModelWrapper(model, (1,), properties=[violated], task_loss=nn.MSELoss(), num_epochs=1)
        result = bad.certify_properties(refine=False, complete=True, results_path=directory)
        assert result['cells'][0]['outcome'] == 'violated'
        result = wrapper.certify_properties(properties=[violated], refine=False,
                                            complete=True, results_path=directory)
        assert result['cells'][0]['property'] == 'unsafe' and result['cells'][0]['outcome'] == 'violated'
        assert wrapper.properties[0].name == 'overlap'
    if '--complete' in sys.argv:
        with tempfile.TemporaryDirectory() as directory:
            result = wrapper.certify_properties(method='CROWN', refine=False,
                complete=True, results_path=directory, timeout=30)
            assert result['certified']
            assert result['cells'][0]['graph'] == 'onnx-relu'
            assert result['cells'][0]['solver_status'] == 'unsat'
        print('Actual alpha-beta-CROWN overlapping-disjunction proof passed')
    print('ACAS conventions, ONNX adapters, and verifier outcome checks passed')


if __name__ == '__main__':
    main()
