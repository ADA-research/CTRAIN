"""Adversarial training losses using the shared PGD implementation."""
from CTRAIN.attacks import pgd_attack
from CTRAIN.util.util import preserve_model_state

def get_pgd_loss(hardened_model, ptb, data, target, n_classes, criterion, return_bounds=False, n_steps=200, step_size=0.2, restarts=1, early_stopping=True, *, decay_factor=.1, decay_checkpoints=(), return_stats=False):
    """Compute the mean criterion on PGD-generated adversarial inputs.

    The attack runs in evaluation mode; loss evaluation uses the caller's model
    mode. return_bounds must be False. Returns the scalar loss, or (loss,
    adversarial_error) when return_stats=True.
    """
    assert not return_bounds, "Return Bounds not available PGD training!"
    
    with preserve_model_state(hardened_model):
        hardened_model.eval()
        x_adv = pgd_attack(
            model=hardened_model, data=data, target=target,
            x_L=ptb.x_L, x_U=ptb.x_U, restarts=restarts,
            step_size=step_size, n_steps=n_steps, early_stopping=early_stopping,
            device=data.device, decay_factor=decay_factor,
            decay_checkpoints=decay_checkpoints,
        )
    
    adv_pred = hardened_model(x_adv)
    
    loss = criterion(adv_pred, target).mean()
    if return_stats:
        return loss, (adv_pred.detach().argmax(1) != target).float().mean().item()
    return loss
