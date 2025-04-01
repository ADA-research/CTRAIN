from CTRAIN.attacks import pgd_attack
import torch

def get_pgd_loss(hardened_model, ptb, data, target, n_classes, criterion, return_bounds=False, n_steps=200, step_size=0.2, restarts=1, early_stopping=True):
    assert not return_bounds, "Return Bounds not available PGD training!"
    
    with torch.no_grad():
        x_adv = pgd_attack(
            model=hardened_model,
            data=data,
            target=target,
            x_L=ptb.x_L,
            x_U=ptb.x_U,
            restarts=restarts,
            step_size=step_size,
            n_steps=n_steps,
            early_stopping=early_stopping
        )
    
    adv_pred = hardened_model(x_adv)
    
    return criterion(adv_pred, target).mean()