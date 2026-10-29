"""Real CPU checks of the bundled NeuralSAT; requires CTRAIN's dependencies."""
import tempfile
from pathlib import Path

import torch
from CTRAIN.complete_verification.neuralsat import neuralsat_eval, validate_neuralsat_command
from CTRAIN.complete_verification.abCROWN.util import write_adversarial_robustness_vnnlib
from CTRAIN.util import export_onnx


def main():
    command = validate_neuralsat_command(None)
    with tempfile.TemporaryDirectory(prefix='ctrain-neuralsat-real-') as directory:
        root = Path(directory)
        model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(1, 2)).eval()
        with torch.no_grad():
            model[1].weight.copy_(torch.tensor([[1.], [-1.]]))
            model[1].bias.copy_(torch.tensor([1., 0.]))
        export_onnx(model, str(root / 'model.onnx'), 1, [1, 1, 1])
        for label, expected in [(0, 'unsat'), (1, 'sat')]:
            spec = root / f'{label}.vnnlib'
            write_adversarial_robustness_vnnlib(spec, 'Known linear property',
                                               torch.tensor([[0., 1.]]), label, 2)
            log = root / f'{label}.log'
            runtime, status = neuralsat_eval(root / 'model.onnx', spec, command,
                                            device='cpu', timeout=60, log_path=log)
            if status != expected:
                print(log.read_text())
            assert status == expected, (status, expected)
            print(f'Bundled NeuralSAT: {expected} in {runtime:.2f}s')
    print('Bundled NeuralSAT real CPU checks passed')


if __name__ == '__main__':
    main()
