"""Combine feature bounds, latent searches and gradient links for TAPS and STAPS training."""
import math
import torch
from torch import nn
from auto_LiRPA import PerturbationLpNorm
from auto_LiRPA.operators.normalization import BoundBatchNormalization

from CTRAIN.bound import bound_ibp, bound_sabr
from CTRAIN.util import construct_c
from CTRAIN.bound.util import validate_hybrid_options
from CTRAIN.util.util import preserve_model_state


class _BoundGraphBlock(nn.Module):
    """View a feature-extractor or classifier segment of an existing LiRPA graph with shared
    parameters and buffers.
    """

    def __init__(self, graph, end, start=None):
        super().__init__()
        object.__setattr__(self, 'graph', graph)
        object.__setattr__(self, 'start', start)
        object.__setattr__(self, 'end', end)
        nodes = []
        def visit(node):
            if node is start or node in nodes:
                return
            nodes.append(node)
            for parent in node.inputs:
                visit(parent)
        visit(end)
        self.operators = nn.ModuleList(nodes)
        self.train(graph.training)

    def forward(self, data):
        """Evaluate the graph segment on image or latent inputs.

        Classifier inputs seed the existing intermediate-node cache. During training,
        latent forwards use batch statistics without advancing BatchNorm running
        buffers; the original momentum values are restored afterwards.
        """
        if self.start is None:
            return self.graph(data, final_node_name=self.end.name)
        # LiRPA has no forward-from-intermediate API. Seed its existing forward
        # cache at the cut and invalidate only the suffix, including parameters.
        for node in self.operators:
            if hasattr(node, 'forward_value'):
                del node.forward_value
        self.start.forward_value = data
        # Latent point batches set current BN statistics for the loss, while
        # inference running buffers continue to follow clean/SABR forwards.
        batchnorms = [node for node in self.operators
                      if isinstance(node, BoundBatchNormalization) and node.training]
        momenta = [node.momentum for node in batchnorms]
        try:
            for node in batchnorms:
                node.momentum = 0
            return self.graph.get_forward_value(self.end)
        finally:
            for node, momentum in zip(batchnorms, momenta):
                node.momentum = momentum

    def compute_bounds(self, x, **kwargs):
        """Compute segment bounds, supplying explicit intermediate intervals for classifier inputs."""
        if self.start is not None:
            latent = x[0]
            if latent.ptb.x_L is None or latent.ptb.x_U is None:
                raise ValueError('Classifier bounds require explicit latent lower/upper bounds')
            kwargs['interm_bounds'] = {self.start.name: (latent.ptb.x_L, latent.ptb.x_U)}
            x = (self.graph[self.graph.input_name[0]].value,)
        return self.graph.compute_bounds(x=x, final_node_name=self.end.name, **kwargs)


def split_bound_graph(graph, block_sizes, layers):
    """Expose feature-extractor and classifier views of one converted network.

    The views share the original graph's operators, parameters and BatchNorm
    buffers. This avoids a second conversion when TAPS alternates between feature
    bounds and latent classifier searches.

    Args:
        graph (auto_LiRPA.BoundedModule): Converted sequential single-input network.
        block_sizes (tuple[int, int]): Positive layer counts covering every layer exactly
            once.
        layers (Sequence): Sequential layers corresponding to the graph operators.

    Returns:
        (list): Feature-extractor and classifier graph views.
    """
    if (len(block_sizes) != 2
            or any(type(size) is not int or size <= 0 for size in block_sizes)
            or sum(block_sizes) != len(layers)):
        raise ValueError('TAPS/STAPS require two positive block sizes covering every layer')
    if sum(index is not None for index in graph.input_index) != 1:
        raise ValueError('TAPS/STAPS require a sequential single-input bound graph')
    chain, node = [], graph[graph.output_name[0]]
    while node.name != graph.input_name[0]:
        if not node.inputs or any(parent.from_input for parent in node.inputs[1:]):
            raise ValueError('TAPS/STAPS require a sequential single-input bound graph')
        chain.append(node)
        node = node.inputs[0]
    chain.reverse()
    if len(chain) != len(layers):
        raise ValueError('TAPS/STAPS require one bound operator per sequential layer')
    cut = chain[block_sizes[0] - 1]
    return [_BoundGraphBlock(graph, cut), _BoundGraphBlock(graph, chain[-1], cut)]


def bound_taps(original_model, hardened_model, bounded_blocks, data, target, n_classes, ptb, device='cuda', pgd_steps=20, pgd_restarts=1, pgd_step_size=.2,
               pgd_decay_factor=.2, pgd_decay_checkpoints=(5,7),
               gradient_link_thresh=.5, gradient_link_tolerance=1e-05, propagation="IBP", sabr_args=None, relu_upper_retention=1.0, pgd_eps_factor=1.0):
    """Estimate classifier margins using interval propagation and latent PGD.

    TAPS propagates the input region through the feature extractor using IBP,
    then searches for pivotal points in its output interval. STAPS uses SABR
    for feature propagation. The returned latent margins are training proxies;
    use ordinary full-region bounds for sound certification.

    Args:
        original_model (torch.nn.Module): The original neural network model.
        hardened_model (auto_LiRPA.BoundedModule): The auto_LiRPA model.
        bounded_blocks (list): List of bounded blocks of the model.
        data (Tensor): The input data tensor.
        target (Tensor): The target labels tensor.
        n_classes (int): The number of classes for classification.
        ptb (auto_LiRPA.PerturbationLpNorm): The perturbation object defining the
            perturbation set.
        device (str, optional): The device to run the computation on.
        pgd_steps (int, optional): The number of steps for the PGD attack.
        pgd_restarts (int, optional): The number of restarts for the PGD attack.
        pgd_step_size (float, optional): Latent-PGD step-size factor; None uses max(0.2, 2 /
            steps) for positive step counts.
        pgd_decay_factor (float, optional): The decay factor for the PGD attack.
        pgd_decay_checkpoints (tuple, optional): The decay checkpoints for the PGD attack.
        gradient_link_thresh (float, optional): The threshold for gradient linking.
        gradient_link_tolerance (float, optional): The tolerance for gradient linking.
        propagation (str, optional): The propagation method to use ('IBP' or 'SABR').
        sabr_args (dict, optional): The arguments for the SABR method.
        relu_upper_retention (float): Unstable ReLU upper bound fraction for this training
            bound; default 1. Values below 1 are not sound
            certificates.
        pgd_eps_factor (float): Positive latent PGD search-range factor; also scales SABR
            input searches in STAPS. Nominal IBP regions are unchanged.

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
    """Estimate pivotal points for the classifier block using Projected Gradient Descent (PGD).

    Args:
        block (auto_LiRPA.BoundedModule): The neural network block for which to estimate
            pivotal points.
        input_lb (torch.Tensor): Lower bound of the input to the network block.
        input_ub (torch.Tensor): Upper bound of the input to the network block.
        pgd_steps (int): Number of PGD steps to perform.
        pgd_restarts (int): Number of PGD restarts to perform.
        pgd_step_size (float): Latent-PGD step-size factor; None uses max(0.2, 2 / steps)
            for positive step counts.
        pgd_decay_factor (float): Decay factor for PGD step size.
        pgd_decay_checkpoints (list of int): Checkpoints at which to decay the PGD step
            size.
        n_classes (int): Number of classes in the classification task.
        C (torch.Tensor, optional): Matrix specifying the correct class for bound margin
            calculation. Must be provided.

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
    """Estimate pivotal points for a batch using Projected Gradient Descent (PGD).

    Args:
        block (auto_LiRPA.BoundedModule): The neural network block for which to estimate
            pivotal points.
        lb (torch.Tensor): Lower bound of the input.
        ub (torch.Tensor): Upper bound of the input.
        pgd_steps (int): Number of PGD steps.
        pgd_restarts (int): Number of PGD restarts.
        pgd_step_size (float): Latent-PGD step-size factor; None uses max(0.2, 2 / steps)
            for positive step counts.
        pgd_decay_factor (float): Decay factor for learning rate.
        pgd_decay_checkpoints (list): Checkpoints for learning rate decay.
        C (torch.Tensor): Matrix specifying the correct class for bound margin calculation.
            Must be provided.
        n_classes (int): Number of classes.
        device (str, optional): Compatibility argument; allocation uses the bounds' device
            and dtype.

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
    """Estimate bounds for specified dimensions from given adversarial examples.

    Args:
        block (auto_LiRPA.BoundedModule): The neural network block for which to estimate
            pivotal points.
        pts (torch.Tensor): Tensor of adversarial examples of shape (batch_size,
            num_pivotal, *shape_in[1:]).
        dim_to_estimate (torch.Tensor): Tensor indicating the dimensions to estimate, shape
            (batch_size, num_dims, dim_size).
        C (torch.Tensor): Matrix specifying the correct class for bound margin calculation.
            Must be provided.

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
    """Link latent pivotal-point gradients to feature interval endpoints for TAPS.

    The forward computation returns the selected point. Backward assigns
    endpoint gradients according to its distance from each interval boundary.
    This is the method's surrogate gradient, not the derivative of the identity
    forward computation. The threshold c determines the linked boundary region,
    and tol limits division near zero-width intervals.
    """
    @staticmethod
    def forward(ctx, lb, ub, x, c:float, tol:float):
        """Return the pivotal point and retain the feature bounds for the surrogate backward pass.

        Args:
            ctx (torch.autograd.function.FunctionCtx): Context object to save information for backward computation.
            lb (torch.Tensor): Lower bound tensor.
            ub (torch.Tensor): Upper bound tensor.
            x (torch.Tensor): Input tensor.
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
        """Distribute pivotal-point gradients to nearby lower and upper feature bounds.

        Args:
            ctx (torch.autograd.function.FunctionCtx): Context object containing saved tensors and constants.
            grad_x (torch.Tensor): Gradient of the loss with respect to the output of the forward function.

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
    """Scale a TAPS loss gradient while preserving its forward value.

    This custom autograd operation lets the training objective weight the
    latent-loss derivative by alpha without multiplying the reported loss.
    """

    @staticmethod
    def forward(ctx, x, alpha:float=1):
        """Return x unchanged and retain the requested gradient multiplier.

        Args:
            ctx (torch.autograd.function.FunctionCtx): The context object that can be used to stash information
                for backward computation.
            x (torch.Tensor): The input tensor.
            alpha (float, optional): A scaling factor.

        Returns:
            (torch.Tensor): The input tensor `x`.
        """
        ctx.alpha = alpha
        return x

    @staticmethod
    def backward(ctx, grad_x):
        """Multiply the incoming gradient by alpha; return no gradient for the multiplier.

        Args:
            ctx (torch.autograd.function.FunctionCtx): The context object that can be used to stash information for backward
                computation.
            grad_x (torch.Tensor): The gradient of the loss with respect to the output of the forward pass.

        Returns:
            (Tuple[Tensor, None]): A tuple containing the gradient of the loss with respect to the input of the forward pass and None (as there is no gradient with respect to the second input).
        """
        return ctx.alpha * grad_x, None
