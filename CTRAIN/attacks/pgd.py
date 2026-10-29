"""Search for classification and general-property counterexamples using projected gradient descent."""
import math

import torch

def pgd_attack(model, data, target, x_L, x_U, restarts=1, step_size=.2,
               n_steps=200, early_stopping=True, device='cuda',
               decay_factor=.1, decay_checkpoints=(), *, loss_fn=None, is_success=None):
    """Search for counterexamples using Projected Gradient Descent (PGD).

    Each restart begins at a random point in the input box. Gradient-sign
    updates maximise the attack objective and are projected back into that box.
    The step size is scaled by the largest coordinate half-width. The caller
    chooses the model mode; the attack computes input gradients without
    accumulating parameter gradients. Optional callbacks extend the search
    from misclassification to general output-property violations.

    Args:
        model (torch.nn.Module): The neural network model to attack.
        data (torch.Tensor): The input data to perturb.
        target (torch.Tensor | None): Labels, or None with both property callbacks.
        x_L (torch.Tensor): The lower bound of the input data.
        x_U (torch.Tensor): The upper bound of the input data.
        restarts (int, optional): The number of random restarts.
        step_size (float, optional): Step-size factor multiplied by the largest coordinate
            half-width of the search box.
        n_steps (int, optional): The number of steps for the attack.
        early_stopping (bool, optional): Whether to stop early if adversarial examples are
            found.
        device (str, optional): The device to perform the attack on.
        decay_factor (float, optional): The factor by which to decay the step size at each
            checkpoint.
        decay_checkpoints (tuple, optional): The checkpoints at which to decay the step
            size.
        loss_fn (callable, optional): Per-example attack objective to maximise,
            called as loss_fn(output, target). Defaults to cross-entropy.
        is_success (callable, optional): Per-example Boolean violation test,
            called as is_success(output, target). Defaults to misclassification.
            General-property callbacks may use target=None.

    Returns:
        (torch.Tensor): The best successful candidate per input, or the largest
            attack-objective candidate if none succeeded. Checks nominal inputs,
            random initializations and every update, including with n_steps=0.
    """
    if type(restarts) is not int or restarts <= 0:
        raise ValueError('restarts must be a positive integer')
    if type(n_steps) is not int or n_steps < 0:
        raise ValueError('n_steps must be a nonnegative integer')
    if not math.isfinite(step_size) or step_size < 0:
        raise ValueError('step_size must be finite and nonnegative')
    if loss_fn is not None and not callable(loss_fn) or is_success is not None and not callable(is_success):
        raise ValueError('PGD callbacks must be callable')
    if target is None and (loss_fn is None or is_success is None):
        raise ValueError('PGD without labels requires loss and success callbacks')

    x_L, x_U = x_L.to(device).detach(), x_U.to(device).detach()
    if not torch.isfinite(x_L).all() or not torch.isfinite(x_U).all() or (x_L > x_U).any():
        raise ValueError('PGD requires finite, ordered input bounds')
    data = (x_L + x_U) / 2 if data is None else data.to(device).detach()
    target = None if target is None else target.to(device)
    lr_scale = torch.max((x_U - x_L) / 2)

    adversarial_examples = torch.clamp(data, x_L, x_U).clone()
    example_found = torch.zeros(len(data), dtype=torch.bool, device=device)
    best_loss = torch.full((len(data),), -torch.inf, dtype=data.dtype, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none") if loss_fn is None else loss_fn
    is_success = (lambda output, labels: output.argmax(1) != labels) if is_success is None else is_success

    def record(inputs, outputs, indices):
        labels = None if target is None else target[indices]
        losses = loss_fn(outputs, labels).detach()
        successful = is_success(outputs.detach(), labels)
        if losses.shape != (len(inputs),) or successful.shape != losses.shape or successful.dtype != torch.bool:
            raise ValueError('PGD callbacks must return per-example losses and Boolean violations')
        if not torch.isfinite(losses).all():
            raise FloatingPointError('Non-finite PGD losses')
        # A larger attack objective at a safe point must not erase a counterexample.
        improve = (successful & ~example_found[indices]) | (
            (successful == example_found[indices]) & (losses > best_loss[indices]))
        selected = indices[improve]
        adversarial_examples[selected] = inputs[improve].detach()
        best_loss[selected] = losses[improve]
        example_found[indices] |= successful

    with torch.no_grad():
        record(adversarial_examples, model(adversarial_examples), torch.arange(len(data), device=device))

    for _ in range(restarts):
        if early_stopping and example_found.all():
            break
        attack_input = x_L + torch.rand(data.shape, device=device, dtype=data.dtype) * (x_U - x_L)
        with torch.enable_grad():
            for step in range(n_steps + 1):
                indices = torch.where(~example_found)[0] if early_stopping else torch.arange(len(data), device=device)
                if not len(indices):
                    break
                inputs = attack_input[indices].detach().requires_grad_(True)
                outputs = model(inputs)
                record(inputs, outputs, indices)
                if step == n_steps or (early_stopping and example_found.all()):
                    break
                loss = loss_fn(outputs, None if target is None else target[indices]).sum()
                if not loss.requires_grad:
                    break
                grad = torch.autograd.grad(loss, inputs, allow_unused=True)[0]
                if grad is None:
                    break
                if not torch.isfinite(grad).all():
                    raise FloatingPointError('Non-finite PGD input gradients')
                decay = decay_factor ** sum(step >= checkpoint for checkpoint in decay_checkpoints)
                attack_input[indices] = torch.clamp(
                    inputs.detach() + step_size * lr_scale * decay * grad.sign(),
                    x_L[indices], x_U[indices]).detach()

    return adversarial_examples.detach()
