import numpy as np
import torch
import torch.nn.functional as F
from auto_LiRPA import BoundDataParallel
from auto_LiRPA.operators.relu import BoundRelu

from auto_LiRPA.perturbations import PerturbationLpNorm
from CTRAIN.bound import bound_ibp

def get_shi_regulariser(model, ptb, data, target, eps_scheduler, n_classes, device, tolerance=.5, verbose=False, included_regularisers=['relu', 'tightness'], regularisation_decay=True, loss_fusion=False):
    """
    Compute the Shi regularisation loss for a given model. See Shi et al. (2020) for more details.
    
    Args:
        model (auto_LiRPA.BoundedModule): The bounded model. IMPORTANT: Do not pass the original model, but the hardened model.
        ptb (autoLiRPA.PerturbationLpNorm): The perturbation applied to the input data.
        data (torch.Tensor): Input data tensor.
        target (torch.Tensor): Target labels tensor.
        eps_scheduler (BaseScheduler): Scheduler for epsilon values.
        n_classes (int): Number of classes in the classification task.
        device (torch.device): Device to perform computations on (e.g., 'cpu' or 'cuda').
        tolerance (float, optional): Tolerance value for regularisation. Default is 0.5.
        verbose (bool, optional): If True, prints detailed information during computation. Default is False.
        included_regularisers (list of str, optional): List of regularisers to include in the loss computation. Default is ['relu', 'tightness'].
        regularisation_decay (bool, optional): If True, applies decay to the regularisation loss. Default is True.
        loss_fusion (bool, optional): If True, uses loss fusion. Default is False.
    
    Returns:
        torch.Tensor: The computed SHI regulariser loss.
    """
    loss = torch.zeros(()).to(device)

    # Handle the non-feedforward case
    l0 = torch.zeros_like(loss)
    loss_tightness, loss_std, loss_relu, loss_ratio = (l0.clone() for i in range(4))

    if isinstance(model, BoundDataParallel):
        modules = list(model._modules.values())[0]._modules
    else:
        modules = model._modules
    # print(modules)
    # print(modules.keys())
    # print(model)
    node_inp = modules['/input-1']#modules['/input.1']
    if node_inp.upper is None:
        _, _ = bound_ibp(
                model=model,
                ptb=ptb,
                data=data,
                target=target,
                n_classes=n_classes,
                bound_upper=True,
                loss_fusion=loss_fusion
            )
    tightness_0 = ((node_inp.upper - node_inp.lower) / 2).mean()
    ratio_init = tightness_0 / ((node_inp.upper + node_inp.lower) / 2).std()
    cnt_layers = 0
    cnt = 0
    for m in model._modules.values():
        if isinstance(m, BoundRelu):
            lower, upper = m.inputs[0].lower, m.inputs[0].upper
            center = (upper + lower) / 2
            diff = ((upper - lower) / 2)
            tightness = diff.mean()
            mean_ = center.mean()
            std_ = center.std()            

            loss_tightness += F.relu(tolerance - tightness_0 / tightness.clamp(min=1e-12)) / tolerance
            loss_std += F.relu(tolerance - std_) / tolerance
            cnt += 1

            # L_{relu}
            mask_act, mask_inact = lower>0, upper<0
            mean_act = (center * mask_act).mean()
            mean_inact = (center * mask_inact).mean()
            delta = (center - mean_)**2
            var_act = (delta * mask_act).sum()# / center.numel()
            var_inact = (delta * mask_inact).sum()# / center.numel()                        

            mean_ratio = mean_act / -mean_inact
            var_ratio = var_act / var_inact
            mean_ratio = torch.min(mean_ratio, 1 / mean_ratio.clamp(min=1e-12))
            var_ratio = torch.min(var_ratio, 1 / var_ratio.clamp(min=1e-12))
            loss_relu_ = ((
                F.relu(tolerance - mean_ratio) + F.relu(tolerance - var_ratio)) 
                / tolerance)       
            if not torch.isnan(loss_relu_) and not torch.isinf(loss_relu_):
                loss_relu += loss_relu_ 

            if verbose:
                bn_mean = (lower.mean() + upper.mean()) / 2
                bn_var = ((upper**2 + lower**2) / 2).mean() - bn_mean**2
                print(m.name, m, 
                    'tightness {:.4f} gain {:.4f} std {:.4f}'.format(
                        tightness.item(), (tightness/tightness_0).item(), std_.item()),
                    'input', m.inputs[0], m.inputs[0].name,
                    'active {:.4f} inactive {:.4f}'.format(
                        (lower>0).float().sum()/lower.numel(),
                        (upper<0).float().sum()/lower.numel()),
                    'bnv2_mean {:.5f} bnv2_var {:.5f}'.format(bn_mean.item(), bn_var.item())
                )
                # pre-bn
                lower, upper = m.inputs[0].inputs[0].lower, m.inputs[0].inputs[0].upper
                bn_mean = (lower.mean() + upper.mean()) / 2
                bn_var = ((upper**2 + lower**2) / 2).mean() - bn_mean**2
                print('pre-bn',
                    'bnv2_mean {:.5f} bnv2_var {:.5f}'.format(bn_mean.item(), bn_var.item()))

    loss_tightness /= cnt
    loss_std /= cnt
    loss_relu /= cnt

    for item in ['tightness', 'relu', 'std']:
        loss_ = eval('loss_{}'.format(item))
        if item in included_regularisers:
            loss += loss_

    if regularisation_decay:
        loss = (1 - (eps_scheduler.get_cur_eps(normalise=False) / eps_scheduler.get_max_eps(normalise=False))) * loss
    
    return loss
