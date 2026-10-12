import math

import torch

def pgd_attack(model, data, target, x_L, x_U, restarts=1, step_size=.2,
               n_steps=200, early_stopping=True, device='cuda',
               decay_factor=.1, decay_checkpoints=()):
    """
    Performs the Projected Gradient Descent (PGD) attack on the given model and data.

    Args:
        model (torch.nn.Module): The neural network model to attack.
        data (torch.Tensor): The input data to perturb.
        target (torch.Tensor): The target labels for the input data.
        x_L (torch.Tensor): The lower bound of the input data.
        x_U (torch.Tensor): The upper bound of the input data.
        restarts (int, optional): The number of random restarts. Default is 1.
        step_size (float, optional): The step size for each gradient update. Default is 0.2.
        n_steps (int, optional): The number of steps for the attack. Default is 200.
        early_stopping (bool, optional): Whether to stop early if adversarial examples are found. Default is True.
        device (str, optional): The device to perform the attack on. Default is 'cuda'.
        decay_factor (float, optional): The factor by which to decay the step size at each checkpoint. Default is 0.1.
        decay_checkpoints (tuple, optional): The checkpoints at which to decay the step size. Default is ().

    Returns:
        torch.Tensor: The best successful candidate per input, or the largest
            cross-entropy candidate if none succeeded. Checks nominal inputs,
            random initializations and every update, including with n_steps=0.
    """
    if type(restarts) is not int or restarts <= 0:
        raise ValueError('restarts must be a positive integer')
    if type(n_steps) is not int or n_steps < 0:
        raise ValueError('n_steps must be a nonnegative integer')
    if not math.isfinite(step_size) or step_size < 0:
        raise ValueError('step_size must be finite and nonnegative')

    x_L, x_U = x_L.to(device).detach(), x_U.to(device).detach()
    if not torch.isfinite(x_L).all() or not torch.isfinite(x_U).all() or (x_L > x_U).any():
        raise ValueError('PGD requires finite, ordered input bounds')
    data = (x_L + x_U) / 2 if data is None else data.to(device).detach()
    target = target.to(device)
    lr_scale = torch.max((x_U - x_L) / 2)

    adversarial_examples = torch.clamp(data, x_L, x_U).clone()
    example_found = torch.zeros(len(data), dtype=torch.bool, device=device)
    best_loss = torch.full((len(data),), -torch.inf, dtype=data.dtype, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

    def record(inputs, outputs, indices):
        losses = loss_fn(outputs, target[indices]).detach()
        successful = outputs.detach().argmax(1) != target[indices]
        # A higher CE at a correctly classified point must not erase a counterexample.
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
                loss = loss_fn(outputs, target[indices]).sum()
                grad = torch.autograd.grad(loss, inputs)[0]
                decay = decay_factor ** sum(step >= checkpoint for checkpoint in decay_checkpoints)
                attack_input[indices] = torch.clamp(
                    inputs.detach() + step_size * lr_scale * decay * grad.sign(),
                    x_L[indices], x_U[indices]).detach()

    return adversarial_examples.detach()
