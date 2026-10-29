"""CPU export regression (checkpoint formats are exercised by the evaluation notebook); no dataset download or solver."""
import tempfile
import warnings
from pathlib import Path
from unittest.mock import patch

import onnx
import onnxruntime as ort
import torch
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
from CTRAIN.model_definitions import CNN7_Shi
from CTRAIN.model_wrappers.model_wrapper import CTRAINWrapper
from CTRAIN.util import export_onnx


class CancellationModel(torch.nn.Module):
    """FP32 loses the unit term; float64 retains it."""
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1e8, 1., -1e8]))

    def forward(self, x):
        return (x * self.weight).sum(dim=1, keepdim=True)


def check_float64_fallback(directory):
    x = torch.ones(1, 3)
    native = CancellationModel()
    bounded = BoundedModule(CancellationModel().eval(), x, device='cpu')
    # Exercise the same cached tensor subclass left by verification prechecks.
    with torch.no_grad():
        bounded.compute_bounds(x=(BoundedTensor(x, PerturbationLpNorm(
            norm=float('inf'), eps=.1)),), method='IBP')
    assert any(isinstance(getattr(node, 'center', None), BoundedTensor)
               for node in bounded.modules())
    for model in (native, bounded):
        model.train()
        assert model(x).item() == 0.
        next(model.parameters()).requires_grad_(False)
        modes = [module.training for module in model.modules()]
        gradients = [p.requires_grad for p in model.parameters()]
        locations = [(p.dtype, p.device) for p in model.parameters()]
        before = {key: value.clone() for key, value in model.state_dict().items()}
        for actual, accepted in ((1., True), (2., False), (float('nan'), False),
                                 (float('inf'), False)):
            session = type('Session', (), {
                'get_inputs': lambda self: [type('Input', (), {'name': 'input'})()],
                'run': lambda self, *args: [torch.tensor([[actual]]).numpy()],
            })()
            with patch('torch.randn', return_value=x), \
                 patch('onnxruntime.InferenceSession', return_value=session), \
                 warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                try:
                    export_onnx(model, str(Path(directory) / 'fallback.onnx'), 1, (3,))
                except RuntimeError as error:
                    assert not accepted, str(error)
                    assert ('float64 reference also disagrees' in str(error)
                            if actual == 2. else 'non-finite values' in str(error))
                else:
                    assert accepted, 'Incorrect export accepted by float64 fallback'
                    assert any('passed against a CPU float64 reference' in str(w.message)
                               and w.category is RuntimeWarning for w in caught)
            assert [module.training for module in model.modules()] == modes
            assert all(torch.equal(value, before[key]) for key, value in model.state_dict().items())
            assert [p.requires_grad for p in model.parameters()] == gradients
            assert [(p.dtype, p.device) for p in model.parameters()] == locations
        with patch('torch.randn', return_value=x), \
             patch('onnxruntime.InferenceSession', return_value=session), \
             patch('CTRAIN.util.util.copy.deepcopy', side_effect=RuntimeError('unsupported copy')):
            # Use finite outputs to exercise an unavailable float64 reference.
            session.run = lambda *args: [torch.tensor([[1.]]).numpy()]
            try:
                export_onnx(model, str(Path(directory) / 'unsupported.onnx'), 1, (3,))
            except RuntimeError as error:
                assert 'CPU float64 parity reference unavailable' in str(error)
            else:
                raise AssertionError('Unavailable float64 reference accepted')


def main():
    torch.manual_seed(1)
    model = CNN7_Shi((3, 32, 32), width=2, linear_size=4).eval()
    wrapper = CTRAINWrapper(model, eps=2/255, input_shape=(3, 32, 32), device='cpu')
    x = torch.rand(1, 3, 32, 32)
    expected = wrapper.bounded_model(x).detach()
    with tempfile.TemporaryDirectory() as directory:
        check_float64_fallback(directory)
        for candidate in (model, wrapper.bounded_model):
            path = str(Path(directory) / 'cpu.onnx')
            with patch('torch.cuda.is_available', return_value=True), \
                 patch('CTRAIN.util.util.copy.deepcopy', side_effect=AssertionError('Unexpected fallback')):
                export_onnx(candidate, path, 1, [1, 3, 32, 32])
            assert any(node.op_type == 'BatchNormalization' for node in onnx.load(path).graph.node)
            session = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
            output = session.run(None, {session.get_inputs()[0].name: x.numpy()})[0]
            assert torch.allclose(torch.from_numpy(output), expected, atol=1e-5, rtol=1e-4)
        model.train()
        model.layers[1].eval()
        modes = [module.training for module in model.modules()]
        before = {key: value.clone() for key, value in model.state_dict().items()}
        fake_session = type('Session', (), {
            'get_inputs': lambda self: [type('Input', (), {'name': 'input'})()],
            'run': lambda self, *args: [expected.numpy() + 100.],
        })()
        with patch('onnxruntime.InferenceSession', return_value=fake_session):
            try:
                export_onnx(model, str(Path(directory) / 'invalid.onnx'), 1, [1, 3, 32, 32])
            except RuntimeError as error:
                assert 'changed model outputs' in str(error)
            else:
                raise AssertionError('Export parity mismatch accepted')
        assert [module.training for module in model.modules()] == modes
        assert all(torch.equal(value, before[key]) for key, value in model.state_dict().items())
    print('CNN7 CPU ONNX parity checks passed')


if __name__ == '__main__':
    main()
