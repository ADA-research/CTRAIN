"""User-selected ONNX/VNNLIB pairs for property training."""
import hashlib
from pathlib import Path
import torch
from CTRAIN.properties.vnnlib import _read_vnnlib
from CTRAIN.util.onnx import load_onnx_model


def load_vnnlib_problem(model_path, specification_path, *, name=None,
                        input_shape=None, output_semantics='unsafe', expected_sha256=None):
    """Load an ONNX network and VNNLIB specification for property-aware training.

    The loader constructs the desired output formula and finite input boxes,
    converts the model and checks predictions at box endpoints, midpoints and
    128 deterministic random inputs per box. File paths and hashes record the
    provenance of the resulting training problem.

    Args:
        model_path (str | pathlib.Path): Local single-input/output ONNX model.
        specification_path (str | pathlib.Path): Local bounded-box VNNLIB file.
        name (str | None): Property name, defaulting to the specification stem.
        input_shape (tuple[int, ...] | None): Shape without batch. Inferred from
            static ONNX dimensions, otherwise defaults to a vector. Override
            this when dynamic image dimensions need an explicit shape.
        output_semantics (str): `'unsafe'` complements source output assertions;
            `'safe'` treats them as desired conditions. Defaults to `'unsafe'`.
        expected_sha256 (dict | None): Optional expected digests with exactly
            the keys `'model'` and `'specification'`. Checked before file parsing.

    Returns:
        (tuple): `(model, property_spec, provenance)`. Model is trainable on CPU
            in eval mode. Provenance records resolved paths, SHA256 hashes, input
            shape and output semantics. Recorded hashes alone do not authenticate
            an upstream source. Input normalisation remains the caller's choice.

    Raises:
        ValueError: Hashes, shapes, input dtype or VNNLIB constructs unsupported.
        AssertionError: Converted predictions fail ONNX Runtime parity.

    ONNX float32/float64 inputs and batched vector outputs are supported. Bound
    operator support is checked later by `PropertyModelWrapper`. See
    `load_vnnlib_property` for the supported specification subset.
    """
    import onnx
    paths = dict(model=Path(model_path), specification=Path(specification_path))
    hashes = {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in paths.items()}
    if expected_sha256 is not None and expected_sha256 != hashes:
        raise ValueError('Expected model/specification SHA256 hashes do not match')
    graph = onnx.load(paths['model'])
    initializers = {value.name for value in graph.graph.initializer}
    inputs = [value for value in graph.graph.input if value.name not in initializers]
    if len(inputs) != 1:
        raise ValueError('Property problems require a single ONNX input')
    dtype = {onnx.TensorProto.FLOAT: torch.float32,
             onnx.TensorProto.DOUBLE: torch.float64}.get(inputs[0].type.tensor_type.elem_type)
    if dtype is None:
        raise ValueError('ONNX input must be float32 or float64')
    dimensions = [dim.dim_value for dim in inputs[0].type.tensor_type.shape.dim][1:]
    if input_shape is None and dimensions and all(dim > 0 for dim in dimensions):
        input_shape = tuple(dimensions)
    spec, width = _read_vnnlib(paths['specification'], name, input_shape, output_semantics)
    generator = torch.Generator().manual_seed(17)
    batches = []
    for box in spec.domains:
        random = box.lower + torch.rand((128, *box.lower.shape), generator=generator,
                                       dtype=torch.double) * (box.upper - box.lower)
        batches.append(torch.cat((torch.stack((box.lower, box.upper, (box.lower + box.upper) / 2)), random)))
    probes = torch.cat(batches).to(dtype)
    model = load_onnx_model(paths['model'], probes)
    with torch.no_grad():
        if model(probes).shape != (len(probes), width):
            raise ValueError('ONNX output width must match the declared VNNLIB outputs')
    provenance = dict(paths={key: str(path.resolve()) for key, path in paths.items()},
                      sha256=hashes, input_shape=list(box.lower.shape), output_semantics=output_semantics)
    return model, spec, provenance
