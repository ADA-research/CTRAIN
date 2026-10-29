"""L1 parameter regularisation for certified training."""
import torch
import torch.nn as nn


def get_l1_reg(model, device='cuda'):
    """Sum absolute parameter values to obtain the L1 training regulariser.

    Args:
        model (torch.nn.Module): The neural network model to regularize. IMPORTANT: Don't
            pass the bounded model here.
        device (str, optional): The device to perform the computation on.

    Returns:
        (torch.Tensor): The L1 regularisation loss.
    """
    loss = torch.zeros(()).to(device)
    # only regularise Linear and Convolutional layers
    for module in model.modules():
        if isinstance(module, nn.Linear):
            loss += torch.abs(module.weight).sum()
        elif isinstance(module, nn.Conv2d):
            loss += torch.abs(module.weight).sum()
    return loss