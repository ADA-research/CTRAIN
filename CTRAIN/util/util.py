import torch
import os
import random
import numpy as np
import onnx
import inspect
import copy
import warnings
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def preserve_model_state(*models):
    """Restore module modes and parameter gradient flags, including on failure."""
    modes = {module: module.training for model in models for module in model.modules()}
    gradients = {param: param.requires_grad for model in models for param in model.parameters()}
    try:
        yield
    finally:
        for module, training in modes.items():
            module.training = training
        for param, requires_grad in gradients.items():
            param.requires_grad_(requires_grad)


def loader_radius(eps, loader, data):
    """Convert a raw-pixel radius to the loader's model input coordinates."""
    radius = torch.as_tensor(eps, device=data.device, dtype=data.dtype)
    if getattr(loader, 'normalised', False):
        std = torch.as_tensor(loader.std, device=data.device, dtype=data.dtype)
        if not torch.isfinite(std).all() or (std <= 0).any():
            raise ValueError('Loader standard deviation must be finite and positive')
        radius = radius / std
    if radius.ndim == 1 and data.ndim > 2:
        radius = radius.reshape(-1, *([1] * (data.ndim - 2)))
    return radius


def loader_limits(loader, data):
    return tuple(torch.as_tensor(getattr(loader, name), device=data.device, dtype=data.dtype)
                 for name in ('min', 'max'))


def export_onnx(model, file_name, batch_size, input_shape):
    """
    Exports a PyTorch model to the ONNX format.

    If FP32 parity fails, retry against a separate CPU float64 reference
    with the same tolerance. Acceptance through this fallback emits a warning.

    Args:
        model (torch.nn.Module): The PyTorch model to be exported.
        file_name (str): The file path where the ONNX model will be saved.
        batch_size (int): The batch size for the input tensor.
        input_shape (tuple): The shape of the input tensor. If the shape has 4 dimensions, 
                             the first dimension is assumed to be the batch size.

    Returns:
        None
    """
    reference = next(model.parameters(), None)
    if reference is None:
        reference = next(model.buffers(), torch.empty(0))
    device, dtype = reference.device, reference.dtype
    # Input to the model
    if len(input_shape) == 4:
        batch_size = input_shape[0]
        input_shape = input_shape[1:4]
    x = torch.randn(batch_size, *input_shape, device=device, dtype=dtype, requires_grad=False)

    export_kwargs = {
        "export_params": True,
        "opset_version": 18,
        "do_constant_folding": False,
        "input_names": ["input"],
        "output_names": ["output"],
        "dynamic_axes": {"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        "training": torch.onnx.TrainingMode.EVAL,
        "verbose": False,
    }
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = False

    # Export the model
    with preserve_model_state(model), torch.no_grad():
        model.eval()
        expected = model(x).detach()
        torch.onnx.export(model, x, file_name, **export_kwargs)
        remove_training_mode_attr(file_name, file_name)
        import onnxruntime as ort
        session = ort.InferenceSession(str(file_name), providers=["CPUExecutionProvider"])
        actual = session.run(None, {session.get_inputs()[0].name: x.cpu().numpy()})[0]
        if actual.shape != tuple(expected.shape) or not np.isfinite(actual).all():
            raise RuntimeError("ONNX export changed model outputs (invalid shape or non-finite values)")
        if not np.allclose(expected.cpu().numpy(), actual, rtol=1e-4, atol=1e-5):
            error = np.max(np.abs(expected.cpu().numpy() - actual))
            failure = f"ONNX export changed model outputs (max error {error:g})"
            if dtype != torch.float32:
                raise RuntimeError(failure)
            try:
                # Never cast the caller or the exported graph to float64.
                from auto_LiRPA import BoundedTensor
                # Bound prechecks leave BoundedTensor centers cached on LiRPA
                # nodes. The unperturbed reference only needs their nominal data.
                memo = {id(value): value.detach().as_subclass(torch.Tensor).clone()
                        for module in model.modules() for value in vars(module).values()
                        if isinstance(value, BoundedTensor)}
                reference_model = copy.deepcopy(model, memo).to(device='cpu', dtype=torch.float64)
                reference_model.eval()
                reference_output = reference_model(x.to(device='cpu', dtype=torch.float64)).detach().cpu().numpy()
            except Exception as exc:
                raise RuntimeError(f"{failure}; CPU float64 parity reference unavailable: {exc}") from exc
            reference_error = np.max(np.abs(reference_output - actual))
            if not np.allclose(reference_output, actual, rtol=1e-4, atol=1e-5):
                raise RuntimeError(f"{failure}; CPU float64 reference also disagrees "
                                   f"(max error {reference_error:g})")
            warnings.warn(
                f"ONNX export failed FP32 parity (max error {error:g}) but passed against "
                f"a CPU float64 reference (max error {reference_error:g}) with unchanged "
                "rtol=1e-4, atol=1e-5. Continuing with the original FP32 ONNX model; "
                "this numerical parity check is not a robustness certificate.",
                RuntimeWarning, stacklevel=2,
            )
    print(f"Model exported to {file_name}")
    
def remove_training_mode_attr(onnx_path, output_path):
    model = onnx.load(onnx_path)
    for node in model.graph.node:
        if node.op_type in ["BatchNormalization", "Dropout"]:
            # Filter out 'training_mode' attributes
            cleaned_attrs = [attr for attr in node.attribute if attr.name != "training_mode"]
            # Clear and repopulate attributes
            del node.attribute[:]
            node.attribute.extend(cleaned_attrs)
    onnx.save(model, output_path)
    print(f"Cleaned model saved to: {output_path}")

def save_checkpoint(model, optimizer, loss, epoch, results_path, overwrite=False, lr_scheduler=None, selection_state=None,
                    *, extra_state=None, filename=None):
    """
    Saves the model checkpoint to the specified path.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer used for training the model.
        loss (float): The loss value at the time of saving the checkpoint.
        epoch (int): The current epoch number.
        results_path (str): The directory path where the checkpoint will be saved.
        lr_scheduler (optional): Save scheduler state for exact LR resumption.
        selection_state (dict, optional): Persistent best score, weights and epoch.
        extra_state (dict, optional): Method-specific trajectory data. Common keys
            cannot be replaced.
        filename (str, optional): Explicit filename for compatibility with existing
            checkpoint APIs. Defaults to ``{epoch}_checkpoint.pt``.

    Raises:
        FileExistsError: If a checkpoint for the given epoch already exists at the specified path.
    """
    path = Path(results_path) / (f'{epoch}_checkpoint.pt' if filename is None else filename)
    if path.exists() and not overwrite:
        raise FileExistsError("Checkpoint already exists!")
    checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }
    if lr_scheduler is not None:
        checkpoint['lr_scheduler_state_dict'] = lr_scheduler.state_dict()
    if selection_state is not None:
        checkpoint['selection_state'] = selection_state
    if extra_state is not None:
        if checkpoint.keys() & extra_state.keys():
            raise ValueError('Method-specific checkpoint state cannot replace common fields')
        checkpoint.update(extra_state)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb' if overwrite else 'xb') as output:
        torch.save(checkpoint, output)

def construct_c(data, target, n_classes):
    """
    Constructs a tensor `c` based on the input data, target labels, and the number of classes.
    This is used to calculate the margins between classes during bound calculation.

    Args:
        data (torch.Tensor): The input data tensor.
        target (torch.Tensor): The target labels tensor.
        n_classes (int): The number of classes.

    Returns:
        torch.Tensor: A tensor `c` of shape (batch_size, n_classes - 1, n_classes) where `batch_size` is the size of the input data.
    """
    c = torch.eye(n_classes).type_as(data)[target].unsqueeze(1) - torch.eye(n_classes).type_as(data).unsqueeze(0)
    # remove specifications to self
    I = (~(target.data.unsqueeze(1) == torch.arange(n_classes).type_as(target.data).unsqueeze(0)))
    c = (c[I].view(data.size(0), n_classes - 1, n_classes))
    return c

def seed_ctrain(seed=42):
    """
    Set the seed for random number generation in Python, NumPy, and PyTorch to ensure reproducibility.
    Parameters:
    seed (int): The seed value to use for random number generation. Default is 42.
    This function sets the seed for the following:
    - Python's built-in random module
    - NumPy's random module
    - PyTorch's random number generators (both CPU and CUDA, if available)
    Additionally, if CUDA is available, it sets the following PyTorch settings:
    - torch.backends.cudnn.deterministic to True
    - torch.backends.cudnn.benchmark to False
    These settings ensure that the results are deterministic and reproducible.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
