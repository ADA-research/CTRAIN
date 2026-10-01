import math
import torch
from auto_LiRPA import PerturbationLpNorm

from CTRAIN.bound import bound_ibp, bound_sabr
from CTRAIN.util import construct_c
from CTRAIN.bound.util import validate_hybrid_options
from CTRAIN.util.util import preserve_model_state


def bound_taps(original_model, hardened_model, bounded_blocks, data, target, n_classes, ptb, device='cuda', pgd_steps=20, pgd_restarts=1, pgd_step_size=.2,
               pgd_decay_factor=.2, pgd_decay_checkpoints=(5,7),
               gradient_link_thresh=.5, gradient_link_tolerance=1e-05, propagation="IBP", sabr_args=None, relu_upper_retention=1.0, pgd_eps_factor=1.0):
    """
    Compute the bounds of the model's output using the TAPS method.

    Parameters:
        original_model (torch.nn.Module): The original neural network model.
        hardened_model (autoLiRPA.BoundedModule): The auto_LiRPA model.
        bounded_blocks (list): List of bounded blocks of the model.
        data (Tensor): The input data tensor.
        target (Tensor): The target labels tensor.
        n_classes (int): The number of classes for classification.
        ptb (auto_LiRPA.PerturbationLpNorm): The perturbation object defining the perturbation set.
        device (str, optional): The device to run the computation on. Default is 'cuda'.
        pgd_steps (int, optional): The number of steps for the PGD attack. Default is 20.
        pgd_restarts (int, optional): The number of restarts for the PGD attack. Default is 1.
        pgd_step_size (float, optional): The step size for the PGD attack. Default is 0.2.
        pgd_decay_factor (float, optional): The decay factor for the PGD attack. Default is 0.2.
        pgd_decay_checkpoints (tuple, optional): The decay checkpoints for the PGD attack. Default is (5, 7).
        gradient_link_thresh (float, optional): The threshold for gradient linking. Default is 0.5.
        gradient_link_tolerance (float, optional): The tolerance for gradient linking. Default is 1e-05.
        propagation (str, optional): The propagation method to use ('IBP' or 'SABR'). Default is 'IBP'.
        sabr_args (dict, optional): The arguments for the SABR method. Default is None.

        relu_upper_retention (float): Unstable ReLU upper bound fraction for this training bound; default 1. Values below 1 are not sound certificates.
        pgd_eps_factor (float): Positive latent PGD search-range factor; also scales SABR input searches in STAPS. Nominal IBP regions are unchanged.

    Returns:
        taps_bound(Tuple[Tensor, Tensor]): The TAPS bounds of the model's output.
    """
    validate_hybrid_options(relu_upper_retention, pgd_eps_factor)
    with preserve_model_state(original_model, hardened_model, *bounded_blocks):
        assert len(bounded_blocks) == 2, "Split not supported!"

        if propagation == 'IBP':
            lb, ub = bound_ibp(
                model=bounded_blocks[0],
                ptb=ptb,
                data=data,
                target=None,
                n_classes=n_classes,
                bound_upper=True,
                relu_upper_retention=relu_upper_retention,
            )
        elif propagation == 'SABR':
            assert sabr_args is not None, "Need to Provide SABR arguments if you choose SABR for propagation"
            lb, ub = bound_sabr(
                # Intermediate Bound model instructs to return bounds after the first network block
                **{**sabr_args, "intermediate_bound_model": bounded_blocks[0], "return_adv_output": False,
                                  "relu_upper_retention": relu_upper_retention, "pgd_eps_factor": pgd_eps_factor},
            )

        else:
            raise ValueError("propagation must be IBP or SABR")

        c = construct_c(data, target, n_classes)
        # Attack in eval mode, then evaluate points in the caller's mode (De Palma et al.).
        with preserve_model_state(original_model, hardened_model, *bounded_blocks):
            for model in (original_model, hardened_model, *bounded_blocks):
                model.eval()
            with torch.no_grad():
                attack_lb, attack_ub = lb, ub
                if pgd_eps_factor != 1:
                    midpoint, radius = (lb + ub) / 2, (ub - lb) / 2
                    attack_lb, attack_ub = midpoint - radius * pgd_eps_factor, midpoint + radius * pgd_eps_factor
                adv_samples = _get_pivotal_points(bounded_blocks[1], attack_lb, attack_ub, pgd_steps, pgd_restarts, pgd_step_size, pgd_decay_factor, pgd_decay_checkpoints, n_classes, C=c)

        pts = adv_samples[0].detach()
        pts = torch.transpose(pts, 0, 1)
        pts = RectifiedLinearGradientLink.apply(lb.unsqueeze(0), ub.unsqueeze(0), pts, gradient_link_thresh, gradient_link_tolerance)
        pts = torch.transpose(pts, 0, 1)
        pgd_bounds = _get_bound_estimation_from_pts(bounded_blocks[1], pts, None, c)
        # NOTE: VERY IMPORTANT CHANGES TO TAPS BOUND TO BE COMPATIBLE WITH CTRAIN WORKFLOW
        pgd_bounds = pgd_bounds[:, 1:]
        pgd_bounds = -pgd_bounds


        ibp_lb, ibp_ub = bound_ibp(
            model=bounded_blocks[1],
            ptb=PerturbationLpNorm(x_L=lb, x_U=ub),
            data=(lb + ub) / 2,
            relu_upper_retention=relu_upper_retention,
            target=target,
            n_classes=n_classes,
        )

        return pgd_bounds, ibp_lb

# TODO: Adapted from TAPS code, should be checked if one wants to use multiestimator PGD loss alone
# TODO: Refactoring needed!!
def _get_pivotal_points(block, input_lb, input_ub, pgd_steps, pgd_restarts, pgd_step_size, pgd_decay_factor, pgd_decay_checkpoints, n_classes, C=None):
    """
    Estimate pivotal points for the classifier block using Projected Gradient Descent (PGD).

    Parameters:
        block (autoLiRPA.BoundedModule): The neural network block for which to estimate pivotal points.
        input_lb (torch.Tensor): Lower bound of the input to the network block.
        input_ub (torch.Tensor): Upper bound of the input to the network block.
        pgd_steps (int): Number of PGD steps to perform.
        pgd_restarts (int): Number of PGD restarts to perform.
        pgd_step_size (float): Step size for PGD.
        pgd_decay_factor (float): Decay factor for PGD step size.
        pgd_decay_checkpoints (list of int): Checkpoints at which to decay the PGD step size.
        n_classes (int): Number of classes in the classification task.
        C (torch.Tensor, optional): Matrix specifying the correct class for bound margin calculation. Must be provided.

    Returns:
        (list of torch.Tensor): List containing the concatenated pivotal points tensor.
    """
    assert C is not None # Should only estimate for the final block
    lb, ub = input_lb.clone().detach(), input_ub.clone().detach()

    pt_list = []
    # split into batches
    # TODO: Can we keep this fixed batch size?
    bs = 128
    lb_batches = [lb[i*bs:(i+1)*bs] for i in range(math.ceil(len(lb) / bs))]
    ub_batches = [ub[i*bs:(i+1)*bs] for i in range(math.ceil(len(ub) / bs))]
    C_batches = [C[i*bs:(i+1)*bs] for i in range(math.ceil(len(C) / bs))]
    for lb_one_batch, ub_one_batch, C_one_batch in zip(lb_batches, ub_batches, C_batches):
        pt_list.append(_get_pivotal_points_one_batch(block, lb_one_batch, ub_one_batch, pgd_steps, pgd_restarts, pgd_step_size, pgd_decay_factor, pgd_decay_checkpoints, n_classes=n_classes, C=C_one_batch))
    pts = torch.cat(pt_list, dim=0)
    return [pts, ]

def _get_pivotal_points_one_batch(block, lb, ub, pgd_steps, pgd_restarts, pgd_step_size, pgd_decay_factor, pgd_decay_checkpoints, C, n_classes, device=None):
    """
    Estimate pivotal points for a batch using Projected Gradient Descent (PGD).

    Args:
        block (autoLiRPA.BoundedModule): The neural network block for which to estimate pivotal points.
        lb (torch.Tensor): Lower bound of the input.
        ub (torch.Tensor): Upper bound of the input.
        pgd_steps (int): Number of PGD steps.
        pgd_restarts (int): Number of PGD restarts.
        pgd_step_size (float): Step size for PGD.
        pgd_decay_factor (float): Decay factor for learning rate.
        pgd_decay_checkpoints (list): Checkpoints for learning rate decay.
        C (torch.Tensor): Matrix specifying the correct class for bound margin calculation. Must be provided.
        n_classes (int): Number of classes.
        device (str, optional): Compatibility argument; allocation uses the bounds' device and dtype.

    Returns:
        (torch.Tensor): Adversarial examples per class for whole batch.
    """

    lb, ub = lb.detach(), ub.detach()
    num_pivotal = n_classes - 1 # only need to estimate n_class - 1 dim for the final output

    def init_pts(input_lb, input_ub):
        rand_init = input_lb.unsqueeze(1) + (input_ub-input_lb).unsqueeze(1)*torch.rand(input_lb.shape[0], num_pivotal, *input_lb.shape[1:], device=input_lb.device, dtype=input_lb.dtype)
        return rand_init

    lr_decay_milestones = pgd_decay_checkpoints
    lr_decay_factor = pgd_decay_factor
    if pgd_steps < 0 or pgd_restarts < 1:
        raise ValueError("PGD requires nonnegative steps and at least one restart")
    init_lr = max(0.2, 2 / max(pgd_steps, 1)) if pgd_step_size is None else pgd_step_size
    if init_lr <= 0:
        raise ValueError("PGD step size must be positive")
    pts = init_pts(lb, ub)
    variety = (ub - lb).unsqueeze(1).detach()
    best_estimation = pts.new_full(pts.shape[:2], -torch.inf)
    best_pts = torch.zeros_like(pts)
    with torch.enable_grad():
        for re in range(pgd_restarts):
            lr = init_lr
            pts = init_pts(lb, ub)
            for it in range(pgd_steps+1):
                pts.requires_grad = True
                estimated_pseudo_bound = _get_bound_estimation_from_pts(block, pts, None, C=C)
                improve_idx = estimated_pseudo_bound[:, 1:] > best_estimation
                best_estimation[improve_idx] = estimated_pseudo_bound[:, 1:][improve_idx].detach()
                best_pts[improve_idx] = pts[improve_idx].detach()
                # wants to maximize the estimated bound
                if it != pgd_steps:
                    loss = - estimated_pseudo_bound.sum()
                    grad = torch.autograd.grad(loss, pts)[0]
                    new_pts = pts - grad.sign() * lr * variety
                    pts = torch.max(torch.min(new_pts, ub.unsqueeze(1)), lb.unsqueeze(1)).detach()
                    if (it+1) in lr_decay_milestones:
                        lr *= lr_decay_factor
    return best_pts.detach()


def _get_bound_estimation_from_pts(block, pts, dim_to_estimate, C=None):
    """
    Estimate bounds for specified dimensions from given adversarial examples.

    Parameters:
        block (autoLiRPA.BoundedModule): The neural network block for which to estimate pivotal points.
        pts (torch.Tensor): Tensor of adversarial examples of shape (batch_size, num_pivotal, *shape_in[1:]).
        dim_to_estimate (torch.Tensor): Tensor indicating the dimensions to estimate, shape (batch_size, num_dims, dim_size).
        C (torch.Tensor): Matrix specifying the correct class for bound margin calculation. Must be provided.

    Returns:
        estimated_bounds(torch.Tensor): Estimated bounds tensor of shape (batch_size, num_pivotal) if C is None,
                    otherwise shape (batch_size, n_class).
    """

    if C is None:
        # pts shape (batch_size, num_pivotal, *shape_in[1:])
        out_pts = block(pts.reshape(-1, *pts.shape[2:]))
        out_pts = out_pts.reshape(*pts.shape[:2], -1)
        dim_to_estimate = dim_to_estimate.unsqueeze(1)
        dim_to_estimate = dim_to_estimate.expand(dim_to_estimate.shape[0], out_pts.shape[1], dim_to_estimate.shape[2])
        out_pts = torch.gather(out_pts, dim=2, index=dim_to_estimate) # shape: (batch_size, num_pivotal, num_pivotal)
        estimated_bounds = torch.diagonal(out_pts, dim1=1, dim2=2) # shape: (batch_size, num_pivotal)
    else:
        # # main idea: convert the 9 adv inputs into one batch to compute the bound at the same time; involve many reshaping
        batch_C = C.unsqueeze(1).expand(-1, pts.shape[1], -1, -1).reshape(-1, *(C.shape[1:])) # may need shape adjustment
        batch_pts = pts.reshape(-1, *(pts.shape[2:]))
        out_pts = block(batch_pts)
        out_pts = torch.bmm(batch_C, out_pts.unsqueeze(-1)).squeeze(-1)
        out_pts = out_pts.reshape(*(pts.shape[:2]), *(out_pts.shape[1:]))
        out_pts = - out_pts # the out is the lower bound of yt - yi, transform it to the upper bound of yi - yt
        # the out_pts should be in shape (batch_size, n_class - 1, n_class - 1)
        ub = torch.diagonal(out_pts, dim1=1, dim2=2) # shape: (batch_size, n_class - 1)
        estimated_bounds = torch.cat([torch.zeros(size=(ub.shape[0],1), dtype=ub.dtype, device=ub.device), ub], dim=1) # shape: (batch_size, n_class)

    return estimated_bounds

class RectifiedLinearGradientLink(torch.autograd.Function):
    """
    RectifiedLinearGradientLink is a custom autograd function that establishes a rectified linear gradient link
    between the IBP bounds of the feature extractor (lb, ub) and the
    PGD bounds (x_adv) of the classifier. This function is not a valid gradient with respect
    to the forward function.

    Attributes:
        c (float): A constant used to determine the slope.
        tol (float): A tolerance value to avoid division by zero.

    Methods:
        forward(ctx, lb, ub, x, c: float, tol: float)
        backward(ctx, grad_x):
    """
    @staticmethod
    def forward(ctx, lb, ub, x, c:float, tol:float):
        """
        Saves the input tensors and constants for backward computation.

        Args:
            ctx: Context object to save information for backward computation.
            lb: Lower bound tensor.
            ub: Upper bound tensor.
            x: Input tensor.
            c (float): A constant used to determine the slope.
            tol (float): A tolerance value to avoid division by zero.

        Returns:
            (Tensor): The input tensor x.
        """
        ctx.save_for_backward(lb, ub, x)
        ctx.c = c
        ctx.tol = tol
        return x
    @staticmethod
    def backward(ctx, grad_x):
        """
        Computes the gradient of the loss with respect to the input bounds (lb, ub).

        Args:
            ctx: Context object containing saved tensors and constants.
            grad_x: Gradient of the loss with respect to the output of the forward function.

        Returns:
            (Tuple[Tensor, Tensor, None, None, None]): Gradients with respect to lb, ub, and None for other inputs.
        """
        lb, ub, x = ctx.saved_tensors
        c, tol = ctx.c, ctx.tol
        slackness = c * (ub - lb)
        # handle grad w.r.t. ub
        thre = (ub - slackness)
        Rectifiedgrad_mask = (x >= thre)
        grad_ub = (Rectifiedgrad_mask * grad_x * (x - thre).clamp(min=0.5*tol) / slackness.clamp(min=tol)).sum(dim=0, keepdim=True)
        # handle grad w.r.t. lb
        thre = (lb + slackness)
        Rectifiedgrad_mask = (x <= thre)
        grad_lb = (Rectifiedgrad_mask * grad_x * (thre - x).clamp(min=0.5*tol) / slackness.clamp(min=tol)).sum(dim=0, keepdim=True)
        # we don't need grad w.r.t. x and param
        return grad_lb, grad_ub, None, None, None

class GradExpander(torch.autograd.Function):
    """
    A custom autograd function that scales the gradient during the backward pass.
    This function allows you to define a custom forward and backward pass for a
    PyTorch operation. The forward pass simply returns the input tensor, while
    the backward pass scales the gradient by a specified factor `alpha`.
    Methods:
        forward(ctx, x, alpha: float = 1):
        backward(ctx, grad_x):

    """

    @staticmethod
    def forward(ctx, x, alpha:float=1):
        """
        Forward pass for the custom operation.

        Args:
            ctx: The context object that can be used to stash information
                for backward computation.
            x: The input tensor.
            alpha (float, optional): A scaling factor. Defaults to 1.

        Returns:
            (torch.Tensor): The input tensor `x`.
        """
        ctx.alpha = alpha
        return x

    @staticmethod
    def backward(ctx, grad_x):
        """
        Performs the backward pass for the custom autograd function.

        Args:
            ctx: The context object that can be used to stash information for backward computation.
            grad_x: The gradient of the loss with respect to the output of the forward pass.

        Returns:
            (Tuple[Tensor, None]): A tuple containing the gradient of the loss with respect to the input of the forward pass and None (as there is no gradient with respect to the second input).
        """
        return ctx.alpha * grad_x, None
