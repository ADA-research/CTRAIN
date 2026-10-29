"""Native PyTorch PGD training, including the CC-Dist teacher schedule."""
from CTRAIN.train.certified.util import (
    prepare_model_selection, update_model_selection, restore_model_selection,
    evaluate_validation_loss, validation_ptb, synchronise_bn,
)
import math
from pathlib import Path

import torch
from auto_LiRPA import PerturbationLpNorm

from CTRAIN.train.adversarial.losses import get_pgd_loss
from CTRAIN.train.certified.initialisation import ibp_init_shi
from CTRAIN.train.certified.regularisers import get_l1_reg
from CTRAIN.train.certified.progress import progress_bar, update_progress, log_epoch_summary
from CTRAIN.util import save_checkpoint


def validate_pgd_training_options(eps, steps, alpha, restarts, clip, l1):
    if not math.isfinite(eps) or eps < 0:
        raise ValueError('eps must be finite and nonnegative')
    if (type(steps) is not int or steps < 1 or type(restarts) is not int or restarts < 1
            or not math.isfinite(alpha) or alpha <= 0):
        raise ValueError('Invalid PGD steps, restarts or step size')
    if clip is not None and (not math.isfinite(clip) or clip <= 0):
        raise ValueError('gradient_clip must be finite and positive')
    if l1 is not None and (not math.isfinite(l1) or l1 < 0):
        raise ValueError('l1_reg_weight must be finite and nonnegative')


def make_pgd_lr_scheduler(optimizer, num_epochs, batches_per_epoch):
    """The paper's teacher LR triangle, stepping once per optimizer update."""
    steps = num_epochs * batches_per_epoch
    if steps < 2:
        raise ValueError('The triangular LR schedule needs at least two batches')
    return torch.optim.lr_scheduler.CyclicLR(
        optimizer, base_lr=0., max_lr=[group['lr'] for group in optimizer.param_groups],
        step_size_up=steps / 2, step_size_down=steps / 2, cycle_momentum=False)


def pgd_train_model(model, train_loader, optimizer, num_epochs, eps,
                    lr_scheduler=None, start_epoch=0, end_epoch=None,
                    pgd_steps=10, pgd_alpha=.25, pgd_restarts=1,
                    gradient_clip=10, l1_reg_weight=0., shi_init=True,
                    population_bn=False, results_path=None,
                    checkpoint_save_interval=10, device='cuda', n_classes=10,
                    val_loader=None, eval_eps=None, selection_state=None):
    """Train a native classifier without constructing a bound graph.

    Fixed input radius, evaluation-mode PGD, then training-mode adversarial CE.
    The supplied LR scheduler steps per batch; none gives the CC-Dist teacher
    triangle 0 -> optimizer's LR -> 0 over the full epoch budget. Final weights
    are returned. Checkpoints contain the native model, optimizer and scheduler;
    RNG/loader state is not saved. Partial calls retain the full planned budget.
    """
    selection = prepare_model_selection(selection_state, val_loader, eval_eps)
    end_epoch = num_epochs if end_epoch is None else end_epoch
    if (type(num_epochs) is not int or num_epochs < 1
            or type(start_epoch) is not int or type(end_epoch) is not int
            or not 0 <= start_epoch <= end_epoch <= num_epochs or not len(train_loader)):
        raise ValueError('Invalid epoch range or empty training loader')
    validate_pgd_training_options(eps, pgd_steps, pgd_alpha, pgd_restarts,
                                  gradient_clip, l1_reg_weight)
    if type(checkpoint_save_interval) is not int or checkpoint_save_interval < 1:
        raise ValueError('checkpoint_save_interval must be a positive integer')
    std = torch.as_tensor(train_loader.std)
    if train_loader.normalised and (not torch.isfinite(std).all() or (std <= 0).any()):
        raise ValueError('Loader standard deviation must be finite and positive')
    if lr_scheduler is None:
        if start_epoch:
            raise ValueError('Resuming PGD training requires the saved batch LR scheduler')
        lr_scheduler = make_pgd_lr_scheduler(optimizer, num_epochs, len(train_loader))
    if results_path is not None:
        Path(results_path).mkdir(parents=True, exist_ok=True)
    if start_epoch == 0 and shi_init:
        ibp_init_shi(model, model)
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    for epoch in range(start_epoch, end_epoch):
        model.train()
        loss_sum = error_sum = samples = 0
        progress = progress_bar(train_loader, epoch, num_epochs, eps, method='PGD')
        for data, target in progress:
            data, target = data.to(device), target.to(device)
            radius = torch.as_tensor(eps, device=data.device, dtype=data.dtype)
            if train_loader.normalised:
                radius = radius / std.to(data)
            radius = radius.reshape(-1, *([1] * (data.ndim - 2)))
            minimum, maximum = train_loader.min.to(data), train_loader.max.to(data)
            ptb = PerturbationLpNorm(norm=float('inf'), eps=radius,
                x_L=(data - radius).clamp(minimum, maximum),
                x_U=(data + radius).clamp(minimum, maximum))
            optimizer.zero_grad()
            loss, error = get_pgd_loss(model, ptb, data, target, n_classes, criterion,
                n_steps=pgd_steps, step_size=pgd_alpha, restarts=pgd_restarts,
                early_stopping=False, return_stats=True)
            if l1_reg_weight:
                loss = loss + l1_reg_weight * get_l1_reg(model, device)
            loss.backward()
            if gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip, error_if_nonfinite=True)
            optimizer.step()
            lr_scheduler.step()
            samples += len(data)
            loss_sum += loss.detach().item() * len(data)
            error_sum += error * len(data)
            update_progress(progress, loss_sum / samples, adv_acc=1 - error_sum / samples,
                            lr=optimizer.param_groups[0]['lr'])
        synchronise_bn(model, model, train_loader, device, population_bn)
        log_epoch_summary(epoch, num_epochs, loss_sum / samples, None,
                          adv_acc=1 - error_sum / samples)
        if val_loader is not None:
            def validation_loss(data, target):
                region = validation_ptb(data, val_loader, eps if eval_eps is None else eval_eps)
                return get_pgd_loss(model, region, data, target, n_classes, criterion,
                                    n_steps=pgd_steps, step_size=pgd_alpha,
                                    restarts=pgd_restarts, early_stopping=False)
            validation = evaluate_validation_loss((model,), val_loader, validation_loss, device)
            print(f"Validation robust_loss={validation:.6f}")
            update_model_selection(selection, model, -validation, epoch + 1)
        if results_path is not None and (epoch + 1) % checkpoint_save_interval == 0:
            save_checkpoint(model, optimizer, loss_sum / samples, epoch + 1,
                            results_path, lr_scheduler=lr_scheduler, selection_state=selection)
    if val_loader is not None:
        restore_model_selection(selection, model)
    return model
