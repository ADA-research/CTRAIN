from auto_LiRPA.perturbations import PerturbationLpNorm

import numpy as np
import torch

from CTRAIN.bound.ibp import bound_ibp
from CTRAIN.attacks import pgd_attack


def bound_sabr(hardened_model, original_model, data, target, eps, subselection_ratio, device='cuda', n_classes=10, x_L=None, x_U=None, data_min=None, data_max=None, n_steps=8, step_size=.5, restarts=1, early_stopping=True, intermediate_bound_model=None, decay_factor=0.1, decay_checkpoints=(4,7), return_adv_output=False):
    """
    Compute the lower and upper bounds of the model's output using the SABR method.

    Parameters:
        hardened_model (autoLiRPA.BoundedModule): The auto_LiRPA model.
        original_model (torch.nn.Module): The original neural network model.
        data (Tensor): The input data tensor.
        target (Tensor): The target labels tensor.
        eps (float): The epsilon value for perturbation.
        subselection_ratio (float): The ratio for subselection of the epsilon for the IBP bounding during SABR.
        device (str, optional): The device to run the computation on. Default is 'cuda'.
        n_classes (int, optional): The number of classes for classification. Default is 10.
        x_L (Tensor, optional): The lower bound of the input data. Default is None.
        x_U (Tensor, optional): The upper bound of the input data. Default is None.
        data_min (Tensor, optional): The minimum value of the input data. Default is None.
        data_max (Tensor, optional): The maximum value of the input data. Default is None.
        n_steps (int, optional): The number of steps for the attack. Default is 8.
        step_size (float, optional): The step size for the attack. Default is 0.5.
        restarts (int, optional): The number of restarts for the attack. Default is 1.
        early_stopping (bool, optional): Whether to use early stopping. Default is True.
        intermediate_bound_model (torch.nn.Module, optional): The intermediate bound model. If provided, the SABR bounds of the intermediate model are returned. This is needed during STAPS bound calculation. Default is None.
        decay_factor (float, optional): The decay factor for the attack. Default is 0.1.
        decay_checkpoints (tuple, optional): The decay checkpoints for the attack. Default is (4, 7).
        return_adv_output (bool, optional): Whether to return the adversarial output. Default is False.

    Returns:
        (Tuple[Tensor, Tensor, Tensor]): The lower and upper bounds of the model's output, and the adversarial output if return_adv_output is True.
    """
    hardened_model.eval()
    original_model.eval()
    
    propagation_inputs, tau, x_adv = get_propagation_region(
        model=hardened_model,
        data=data,
        data_min=data_min,
        data_max=data_max,
        target=target,
        eps=eps if (x_L is None and x_U is None) else None,
        subselection_ratio=subselection_ratio,
        n_steps=n_steps,
        step_size=step_size,
        restarts=restarts,
        early_stopping=early_stopping,
        x_L=x_L,
        x_U=x_U,
        decay_checkpoints=decay_checkpoints, 
        decay_factor=decay_factor
    )
    
    hardened_model.train()
    original_model.train()
    
    ptb = PerturbationLpNorm(
        eps=tau,
        norm=np.inf,
        x_L=torch.clamp(propagation_inputs - tau, data_min, data_max).to(device),
        x_U=torch.clamp(propagation_inputs + tau, data_min, data_max).to(device)
    )
    
    # Pass input through network to set batch statistics
    adv_output = hardened_model(x_adv)    
    
    # Use intermediate_bound_model if provided and return intermediate bounds (as needed by STAPS), otherwise use hardened_model
    lb, ub = bound_ibp(
        model=hardened_model if intermediate_bound_model is None else intermediate_bound_model,
        ptb=ptb,
        data=data,
        # data=propagation_inputs,
        # Only provide target if intermediate_bound_model is not used (as we are not interested in final bound margins)
        target=target if intermediate_bound_model is None else None,
        n_classes=n_classes,
        bound_upper=True,
        reuse_input=False
    )
    
    if return_adv_output:
        return lb, ub, adv_output
    return lb, ub

def get_propagation_region(model, data, target, subselection_ratio, step_size, n_steps, restarts, x_L=None, x_U=None, data_min=None, data_max=None, eps=None, early_stopping=True, decay_factor=.1, decay_checkpoints=(4, 7)):
    """
    Get the shrinked propagation region for the SABR method. This is done by performing a PGD attack on the model and taking the resulting adversarial examples as the center of a smaller propagation region.

    Parameters:
        model (torch.nn.Module): The neural network model.
        data (Tensor): The input data tensor.
        target (Tensor): The target labels tensor.
        subselection_ratio (float): The ratio for subselection.
        step_size (float): The step size for the attack.
        n_steps (int): The number of steps for the attack.
        restarts (int): The number of restarts for the attack.
        x_L (Tensor, optional): The lower bound of the input data. Default is None.
        x_U (Tensor, optional): The upper bound of the input data. Default is None.
        data_min (Tensor, optional): The minimum value of the input data. Default is None.
        data_max (Tensor, optional): The maximum value of the input data. Default is None.
        eps (float, optional): The epsilon value for perturbation. Default is None.
        early_stopping (bool, optional): Whether to use early stopping. Default is True.
        decay_factor (float, optional): The decay factor for the attack. Default is 0.1.
        decay_checkpoints (tuple, optional): The decay checkpoints for the attack. Default is (4, 7).

    Returns:
        (Tuple[Tensor, float, Tensor]): The propagation inputs, tau value, and adversarial examples.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    assert (x_L is None and x_U is None and eps is not None) or (x_L is not None and x_U is not None and eps is None), "Please only provide epsilon value OR upper and lower input bounds"
    tau = None
    if eps.all() and data is not None:
        x_L=torch.clamp(data - eps, data_min, data_max).to(device)
        x_U=torch.clamp(data + eps, data_min, data_max).to(device)
    else:
        # TODO: This might break TAPS/STAPS
        eps = torch.max((x_U - x_L))
    
    tau =  subselection_ratio * eps
    
    with torch.no_grad():
        x_adv = pgd_attack(
            model=model,
            data=data,
            target=target,
            x_L=x_L,
            x_U=x_U,
            n_steps=n_steps,
            step_size=step_size,
            restarts=restarts,
            early_stopping=early_stopping,
            device=device,
            decay_checkpoints=decay_checkpoints,
            decay_factor=decay_factor
        )
    
    propagation_inputs = torch.clamp(x_adv, x_L + tau, x_U - tau) # called midpoints in SABR code
    tau = torch.tensor(tau, device=device)
    return propagation_inputs, tau, x_adv