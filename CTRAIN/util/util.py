import torch
import os
import random
import numpy as np
import onnx

def export_onnx(model, file_name, batch_size, input_shape):
    """
    Exports a PyTorch model to the ONNX format.

    Args:
        model (torch.nn.Module): The PyTorch model to be exported.
        file_name (str): The file path where the ONNX model will be saved.
        batch_size (int): The batch size for the input tensor.
        input_shape (tuple): The shape of the input tensor. If the shape has 4 dimensions, 
                             the first dimension is assumed to be the batch size.

    Returns:
        None
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        # Input to the model
    if len(input_shape) == 4:
        batch_size = input_shape[0]
        input_shape = input_shape[1:4]
    x = torch.randn(batch_size, *input_shape, requires_grad=False).to(device)
    torch_out = model(x)

    # Export the model
    torch.onnx.export(model,               # model being run
                    x,                         # model input (or a tuple for multiple inputs)
                    file_name,   # where to save the model (can be a file or file-like object)
                    export_params=True,        # store the trained parameter weights inside the model file
                    opset_version=18,          # the ONNX version to export the model to
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names = ['input'],   # the model's input names
                    output_names = ['output'], # the model's output names
                    dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                    'output' : {0 : 'batch_size'}},
                    training=torch.onnx.TrainingMode.EVAL,
                    verbose=False)
    remove_training_mode_attr(file_name, file_name)
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

def save_checkpoint(model, optimizer, loss, epoch, results_path):
    """
    Saves the model checkpoint to the specified path.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer used for training the model.
        loss (float): The loss value at the time of saving the checkpoint.
        epoch (int): The current epoch number.
        results_path (str): The directory path where the checkpoint will be saved.

    Raises:
        AssertionError: If a checkpoint for the given epoch already exists at the specified path.
    """
    if os.path.exists(f"{results_path}/{epoch}_checkpoint.pt"):
        assert False, "Checkpoint already exists!"
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, f"{results_path}/{epoch}_checkpoint.pt")

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