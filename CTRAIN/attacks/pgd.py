import torch
import torch.optim as optim
import numpy as np

def pgd_attack(model, data, target, x_L, x_U, restarts=1, step_size=.2, n_steps=200, early_stopping=True, device='cuda', decay_factor=.1, decay_checkpoints=()):
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
        torch.Tensor: The generated adversarial examples.
    """
    x_L, x_U = x_L.to(device).detach().clone(), x_U.to(device).detach().clone()
    if data is None:
        data = ((x_L + x_U) / 2).to(device)
    
    lr_scale = torch.max((x_U-x_L)/2)
    
    adversarial_examples = data.detach().clone()
    example_found = torch.zeros(data.shape[0], dtype=torch.bool, device=device)
    best_loss = torch.ones(data.shape[0], dtype=torch.float32, device=device)*(-np.inf)
    
    # TODO: Also support margin loss (although not used in TAPS/SABR/MTL-IBP)
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    for restart_idx in range(restarts):
        
        if early_stopping and example_found.all():
            break

        random_init = (x_L + torch.rand(data.shape, device=device) * (x_U - x_L)).to(device)
        attack_input = random_init.detach().clone()     
                        
        grad_cleaner = optim.SGD([attack_input], lr=1e-3)
        with torch.enable_grad():
            for step in range(n_steps):
                grad_cleaner.zero_grad()
                
                if early_stopping:
                    active_indices = torch.where(~example_found)[0]
                    current_attack_input = attack_input[active_indices]
                    target_active = target[active_indices]
                    x_L_active = x_L[active_indices]
                    x_U_active = x_U[active_indices]
                else:
                    active_indices = None
                    current_attack_input = attack_input
                    target_active = target
                    x_L_active = x_L
                    x_U_active = x_U
                
                current_attack_input.requires_grad = True

                model_out = model(current_attack_input)
                
                loss = loss_fn(model_out, target_active)
                grad = torch.autograd.grad(loss.sum(), current_attack_input)[0]

                if len(decay_checkpoints) > 0:
                    no_passed_checkpoints = len([checkpoint for checkpoint in decay_checkpoints if step >= checkpoint])
                    decay = decay_factor ** no_passed_checkpoints
                else:
                    decay = 1
                    
                step_input_change = step_size * lr_scale * decay * grad.sign()
                
                current_attack_input = torch.clamp(current_attack_input.detach() + step_input_change, x_L_active, x_U_active)
                if early_stopping:
                    attack_input[active_indices] = current_attack_input.detach()
                else:
                    attack_input = current_attack_input

                adv_out = model(current_attack_input)
                
                adv_loss = loss_fn(adv_out, target_active)
                
                if early_stopping:
                    improvement_idx = adv_loss > best_loss[active_indices]
                    improved_indices = active_indices[improvement_idx]
                    best_loss[improved_indices] = adv_loss[improvement_idx].detach()
                    adversarial_examples[improved_indices] = current_attack_input[improvement_idx].detach()
                    
                    misclassified = ~torch.argmax(adv_out.detach(), dim=1).eq(target_active)
                    example_found[active_indices[misclassified]] = True
                    
                else:
                    improvement_idx = adv_loss > best_loss
                    best_loss[improvement_idx] = adv_loss[improvement_idx].detach()
                    adversarial_examples[improvement_idx] = current_attack_input[improvement_idx].detach()
                    
                    example_found[~torch.argmax(adv_out.detach(), dim=1).eq(target)] = True
                    
                if early_stopping and example_found.all():
                    break
                
    grad_cleaner.zero_grad()
    return adversarial_examples.detach()
