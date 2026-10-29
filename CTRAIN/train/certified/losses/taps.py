"""TAPS and STAPS objectives combining latent-adversarial and classifier-interval losses."""
import torch

from CTRAIN.bound.taps import GradExpander
from CTRAIN.bound import bound_taps
from CTRAIN.train.certified.losses import get_loss_from_bounds


def get_taps_loss(original_model, hardened_model, bounded_blocks, criterion, data, target, n_classes, ptb, device='cuda', pgd_steps=8, pgd_restarts=1, 
                  pgd_step_size=None, pgd_decay_factor=.2, pgd_decay_checkpoints=(5,7), gradient_link_thresh=.5,
                  gradient_link_tolerance=1e-05, gradient_expansion_alpha=5, propagation="IBP", sabr_args=None, return_bounds=False, return_stats=False, relu_upper_retention=1.0, pgd_eps_factor=1.0):
    
    """Combine latent-adversarial and interval losses for TAPS or STAPS.

    The objective multiplies the latent-margin loss by the classifier IBP loss.
    Gradient expansion scales the latent-loss gradient without changing its
    forward value. Returned error statistics measure the latent-margin proxy,
    not certified accuracy over the original input region.

    Args:
        original_model (torch.nn.Module): The original model.
        hardened_model (auto_LiRPA.BoundedModule): The bounded model to be trained.
        bounded_blocks (list): List of the LiRPA blocks needed for TAPS (feature extractor
            and classifier).
        criterion (callable): Loss function to be used.
        data (torch.Tensor): Input data.
        target (torch.Tensor): Target labels.
        n_classes (int): Number of classes in the classification task.
        ptb (auto_LiRPA.PerturbationLpNorm): The perturbation applied to the input data.
        device (str, optional): Device to run the computation on.
        pgd_steps (int, optional): Number of PGD steps.
        pgd_restarts (int, optional): Number of PGD restarts.
        pgd_step_size (float, optional): Latent-PGD step-size factor; None uses max(0.2, 2 /
            steps) for positive step counts.
        pgd_decay_factor (float, optional): Decay factor for PGD step size.
        pgd_decay_checkpoints (tuple, optional): Checkpoints for PGD decay.
        gradient_link_thresh (float, optional): Threshold for gradient linking.
        gradient_link_tolerance (float, optional): Tolerance for gradient linking.
        gradient_expansion_alpha (float, optional): Alpha value for gradient expansion.
        propagation (str, optional): Propagation method to be used (SABR or IBP).
        sabr_args (dict, optional): Additional arguments for SABR.
        return_bounds (bool, optional): Whether to return bounds.
        return_stats (bool, optional): Whether to return statistics.
        relu_upper_retention (float): Unstable ReLU upper bound fraction for this training
            bound; default 1. Values below 1 are not sound
            certificates.
        pgd_eps_factor (float): Positive latent PGD search-range factor; also scales SABR
            input searches in STAPS. Nominal IBP regions are unchanged.

    Returns:
        (tuple): A tuple containing the loss, and optionally the bounds and robust error statistics.
    """
    assert len(bounded_blocks) == 2, "Split not supported!"
    
    taps_bound, ibp_bound = bound_taps(
        original_model=original_model,
        hardened_model=hardened_model,
        bounded_blocks=bounded_blocks,
        data=data,
        target=target,
        n_classes=n_classes,
        ptb=ptb,
        device=device,
        pgd_steps=pgd_steps,
        pgd_restarts=pgd_restarts,
        pgd_step_size=pgd_step_size, 
        pgd_decay_factor=pgd_decay_factor,
        pgd_decay_checkpoints=pgd_decay_checkpoints,
        gradient_link_thresh=gradient_link_thresh,
        gradient_link_tolerance=gradient_link_tolerance,
        propagation=propagation,
        sabr_args=sabr_args,
        relu_upper_retention=relu_upper_retention,
        pgd_eps_factor=pgd_eps_factor
    )
        
    taps_loss = get_loss_from_bounds(taps_bound, criterion)
    ibp_loss = get_loss_from_bounds(ibp_bound, criterion)
    
    loss = GradExpander.apply(taps_loss, gradient_expansion_alpha) * ibp_loss
    
    return_tuple = (loss,)
    
    if return_bounds:
        return_tuple = return_tuple + (taps_bound, None)
    if return_stats:
        robust_err = torch.sum(~(taps_bound > 0).all(dim=1)).item() / data.size(0)
        return_tuple = return_tuple + (robust_err,)

    return return_tuple
