"""Load trainable ONNX models using the installed converter and runtime."""
from pathlib import Path
import copy
import torch


def load_onnx_model(path, probes, *, rtol=1e-4, atol=1e-5):
    """Convert a single-input/output ONNX graph and check prediction parity.

    Uses the installed `onnx2pytorch` converter rather than implementing ONNX
    operators in CTRAIN. The returned model is trainable, on CPU and in eval
    mode. Conversion support does not imply auto_LiRPA support for every graph;
    the property wrapper checks that separately when constructing its bounds.

    Args:
        path (str | pathlib.Path): Local ONNX model file.
        probes (torch.Tensor): Nonempty finite batch of representative inputs,
            in the original graph's coordinates and input dtype. Include domain
            boundaries where possible. Probes are evaluated on CPU. Fixed ONNX
            batch sizes are handled by chunking and padding the final chunk.
        rtol (float): Relative tolerance for the ONNX Runtime comparison.
        atol (float): Absolute tolerance for the ONNX Runtime comparison.

    Returns:
        torch.nn.Module: Converted model with trainable parameters.

    Raises:
        ValueError: Probes or graph input/output structure are unsupported.
        AssertionError: Converted predictions disagree with ONNX Runtime.

    The comparison is a numerical check at supplied inputs, not an equivalence
    proof. No normalization or advisory interpretation is added.
    """
    import onnx
    import onnxruntime as ort
    from onnx2pytorch import ConvertModel
    if not isinstance(probes, torch.Tensor) or probes.ndim < 2 or not len(probes) or not (
            probes.is_floating_point() and torch.isfinite(probes).all()):
        raise ValueError('probes must be a nonempty finite floating-point input batch')
    path = Path(path)
    graph = onnx.load(path)
    onnx.checker.check_model(graph)
    session = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
    if len(session.get_inputs()) != 1 or len(session.get_outputs()) != 1:
        raise ValueError('load_onnx_model requires a single input and output')
    inputs = probes.detach().cpu()
    expected_type = {torch.float32: 'tensor(float)', torch.float64: 'tensor(double)'}.get(inputs.dtype)
    if session.get_inputs()[0].type != expected_type:
        raise ValueError('Probe dtype must match the original ONNX input dtype')
    # The installed converter wraps read-only ONNX NumPy arrays as parameters.
    # Give optimizer updates owned, writable tensor storage.
    model = copy.deepcopy(ConvertModel(graph, experimental=True)).to(dtype=inputs.dtype).eval()
    with torch.no_grad():
        batch_size = session.get_inputs()[0].shape[0]
        batch_size = batch_size if isinstance(batch_size, int) and batch_size > 0 else len(inputs)
        predictions = []
        for chunk in inputs.split(batch_size):
            padded = torch.cat((chunk, chunk[-1:].expand(batch_size - len(chunk), *chunk.shape[1:])))
            output = session.run(None, {session.get_inputs()[0].name: padded.numpy()})[0]
            if output.ndim < 1 or output.shape[0] != batch_size:
                raise ValueError('ONNX output must preserve the input batch dimension')
            predictions.append(torch.from_numpy(output[:len(chunk)]))
        expected = torch.cat(predictions)
        actual = model(inputs)
        if not isinstance(actual, torch.Tensor):
            raise ValueError('Converted model must return a single tensor')
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    return model
