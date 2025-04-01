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

        random_noise = (x_L + torch.rand(data.shape, device=device) * (x_U - x_L)).to(device)
        attack_input = data.detach().clone().to(device) + random_noise            
                        
        grad_cleaner = optim.SGD([attack_input], lr=1e-3)
        with torch.enable_grad():
            for step in range(n_steps):
                grad_cleaner.zero_grad()
                
                if early_stopping:
                    attack_input = attack_input[~example_found]
                
                attack_input.requires_grad = True

                model_out = model(attack_input)
                
                loss = loss_fn(model_out, target)
                
                loss.sum().backward(retain_graph=False)

                if len(decay_checkpoints) > 0:
                    no_passed_checkpoints = len([checkpoint for checkpoint in decay_checkpoints if step >= checkpoint])
                    decay = decay_factor ** no_passed_checkpoints
                else:
                    decay = 1
                    
                step_input_change = step_size * lr_scale * decay * attack_input.grad.data.sign()
                
                attack_input = torch.clamp(attack_input.detach() + step_input_change, x_L, x_U)
                adv_out = model(attack_input)
                
                adv_loss = loss_fn(adv_out, target)
                
                if early_stopping:
                    improvement_idx = adv_loss > best_loss[~example_found]
                    best_loss[~example_found & improvement_idx] = adv_loss[improvement_idx].detach()
                    adversarial_examples[~example_found & improvement_idx] = attack_input[improvement_idx].detach()
                    
                    example_found[~example_found][~torch.argmax(adv_out.detach(), dim=1).eq(target)] = True
                    
                else:
                    improvement_idx = adv_loss > best_loss
                    best_loss[improvement_idx] = adv_loss[improvement_idx].detach()
                    adversarial_examples[improvement_idx] = attack_input[improvement_idx].detach()
                    
                    example_found[~torch.argmax(adv_out.detach(), dim=1).eq(target)] = True
                    
                if early_stopping and example_found.all():
                    break
                
    return adversarial_examples.detach()
