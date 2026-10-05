"""IBP regularisation training from De Palma et al., arXiv:2206.14772."""
import math
import copy

import torch
from auto_LiRPA import PerturbationLpNorm

from CTRAIN.train.certified.losses.ibp_r import get_ibp_r_loss, validate_ibp_r_options
from CTRAIN.train.certified.regularisers import get_l1_reg, get_shi_regulariser
from CTRAIN.train.certified.initialisation import ibp_init_shi
from CTRAIN.train.certified.eps_scheduler import BaseScheduler
from CTRAIN.bound.ibp import bound_ibp
from CTRAIN.attacks import pgd_attack
from CTRAIN.train.certified.util import synchronise_bn
from CTRAIN.train.certified.progress import progress_bar, update_progress, log_epoch_summary
from CTRAIN.util import save_checkpoint
from CTRAIN.util.util import preserve_model_state


def ibp_r_train_model(original_model, hardened_model, train_loader,
                      start_epoch=0, end_epoch=None, num_epochs=800, eps=.3,
                      warm_up_epochs=0, ramp_up_epochs=600, optimizer=None,
                      lr_scheduler=None, gradient_clip=None,
                      l1_regularisation_weight=2e-5, rho=2e-4,
                      regularisation_mask=False, pgd_steps=8, pgd_alpha=.25,
                      pgd_restarts=1, pgd_eps_factor=1.0,
                      relu_upper_retention=1.0, population_bn=False,
                      results_path=None, checkpoint_save_interval=10, device="cuda",
                      shi_regularisation_weight=.5, shi_reg_decay=True, shi_init=True,
                      val_loader=None, eval_eps=None, n_classes=10,
                      lr_decay_schedule_unit="epoch", lr_decay_after_ramp_up=False,
                      pgd_early_stopping=False, pgd_decay_factor=.1,
                      pgd_decay_checkpoints=()):
    """Train with linear mixing; select validation adversarial accuracy if supplied.

    ``eps`` is the enlarged training radius, in unnormalised pixel units.
    The loader provides mean/std, min/max and normalised, as elsewhere in CTRAIN.
    Shi initialization and warm-up/ramp-up regularisation follow CTRAIN defaults.
    Linear mixing is specific to IBP-R. LR decay follows the other trainers;
    ``lr_decay_after_ramp_up=True`` reproduces IBP-R's delayed decay instead.
    Validation selects by adversarial accuracy at the fixed ``eval_eps`` radius.
    Epoch indices determine the schedule, including partial/resumed calls;
    the caller must restore optimizer/scheduler state separately when resuming.
    """
    validate_ibp_r_options(rho, pgd_steps, pgd_restarts, pgd_alpha,
                          relu_upper_retention, pgd_eps_factor,
                          pgd_decay_factor, pgd_decay_checkpoints)
    end_epoch = num_epochs if end_epoch is None else end_epoch
    if any(isinstance(value, bool) or not isinstance(value, int) for value in
           (start_epoch, end_epoch, num_epochs, warm_up_epochs, ramp_up_epochs,
            checkpoint_save_interval)):
        raise ValueError("epoch counts and checkpoint_save_interval must be integers")
    if not 0 <= start_epoch <= end_epoch <= num_epochs or num_epochs <= 0:
        raise ValueError("require 0 <= start_epoch <= end_epoch <= num_epochs")
    if warm_up_epochs < 0 or ramp_up_epochs < 0 or warm_up_epochs + ramp_up_epochs > num_epochs:
        raise ValueError("warm-up and ramp-up must be nonnegative and fit num_epochs")
    if not math.isfinite(eps) or eps < 0:
        raise ValueError("eps must be nonnegative and finite")
    if l1_regularisation_weight is not None and (
            not math.isfinite(l1_regularisation_weight) or l1_regularisation_weight < 0):
        raise ValueError("l1_regularisation_weight must be nonnegative and finite")
    if checkpoint_save_interval <= 0 or not len(train_loader):
        raise ValueError("checkpoint interval and loader length must be positive")
    if optimizer is None:
        raise ValueError("optimizer is required")
    if not math.isfinite(shi_regularisation_weight) or shi_regularisation_weight < 0:
        raise ValueError("shi_regularisation_weight must be nonnegative and finite")
    if lr_decay_schedule_unit not in ("epoch", "batch"):
        raise ValueError("lr_decay_schedule_unit must be 'epoch' or 'batch'")
    if eval_eps is not None and (not math.isfinite(eval_eps) or eval_eps < 0):
        raise ValueError("eval_eps must be nonnegative and finite")
    if val_loader is not None and not len(val_loader):
        raise ValueError("validation loader must not be empty")
    if start_epoch == 0 and end_epoch > 0 and shi_init:
        ibp_init_shi(original_model, hardened_model)
    # Reuse the common scheduler's normalisation and regularisation-decay API;
    # IBP-R's linear schedule is set from absolute progress below.
    eps_scheduler = BaseScheduler(num_epochs, eps, train_loader.mean,
                                  train_loader.std if train_loader.normalised else 1.,
                                  eps_schedule=(warm_up_epochs, ramp_up_epochs),
                                  eps_schedule_unit="epoch", batches_per_epoch=len(train_loader))
    best_accuracy, best_state = -1., None
    for epoch in range(start_epoch, end_epoch):
        original_model.train()
        hardened_model.train()
        loss_sum = nat_correct = adv_correct = total = 0
        epoch_mix = min(1., max(0., (epoch - warm_up_epochs) / ramp_up_epochs)) if ramp_up_epochs else float(epoch >= warm_up_epochs)
        progress = progress_bar(train_loader, epoch, num_epochs, eps * epoch_mix, method="IBP-R")
        for batch_idx, (data, target) in enumerate(progress):
            # Absolute progress avoids scheduler off-by-one/resume drift.
            elapsed = epoch + batch_idx / len(train_loader) - warm_up_epochs
            kappa = min(1., max(0., elapsed / ramp_up_epochs)) if ramp_up_epochs else float(elapsed >= 0)
            eps_scheduler.cur_eps = eps * kappa
            data, target = data.to(device), target.to(device)
            radius = data.new_tensor(eps * kappa)
            if train_loader.normalised:
                radius = radius / torch.as_tensor(train_loader.std, device=data.device, dtype=data.dtype)
            radius = radius.reshape(-1, 1, 1)
            data_min = torch.as_tensor(train_loader.min, device=data.device, dtype=data.dtype)
            data_max = torch.as_tensor(train_loader.max, device=data.device, dtype=data.dtype)
            ptb = PerturbationLpNorm(norm=float("inf"), eps=radius,
                                    x_L=torch.clamp(data - radius, data_min, data_max),
                                    x_U=torch.clamp(data + radius, data_min, data_max))
            optimizer.zero_grad()
            clean_output = hardened_model(data)
            nat_correct += (clean_output.detach().argmax(1) == target).sum().item()
            loss, adv_output = get_ibp_r_loss(
                hardened_model, original_model, ptb, data, target,
                kappa=kappa, rho=rho, regularisation_mask=regularisation_mask,
                pgd_steps=pgd_steps, pgd_alpha=pgd_alpha, pgd_restarts=pgd_restarts,
                pgd_eps_factor=pgd_eps_factor, relu_upper_retention=relu_upper_retention,
                data_min=data_min, data_max=data_max, eps=radius,
                pgd_early_stopping=pgd_early_stopping, pgd_decay_factor=pgd_decay_factor,
                pgd_decay_checkpoints=pgd_decay_checkpoints, clean_output=clean_output)
            if shi_regularisation_weight and kappa < 1 and eps > 0:
                # Refresh even during clean warm-up: cached intervals can belong
                # to the previous batch or an adversarial forward pass.
                bound_ibp(hardened_model, ptb, data, None, bound_upper=True,
                          relu_upper_retention=relu_upper_retention)
                loss = loss + shi_regularisation_weight * get_shi_regulariser(
                    model=hardened_model, ptb=ptb, data=data, target=target,
                    eps_scheduler=eps_scheduler, n_classes=n_classes, device=device,
                    included_regularisers=["relu", "tightness"],
                    regularisation_decay=shi_reg_decay,
                    relu_upper_retention=relu_upper_retention)
            if l1_regularisation_weight:
                loss = loss + l1_regularisation_weight * get_l1_reg(original_model, device=device)
            loss.backward()
            if gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(hardened_model.parameters(), gradient_clip)
            optimizer.step()
            if lr_scheduler is not None and lr_decay_schedule_unit == "batch" and (
                    not lr_decay_after_ramp_up or kappa == 1):
                lr_scheduler.step()
            adv_correct += (adv_output.detach().argmax(1) == target).sum().item()
            total += len(data)
            loss_sum += loss.item() * len(data)
            update_progress(progress, loss_sum / total, nat_acc=nat_correct / total,
                            adv_acc=adv_correct / total, lr=optimizer.param_groups[-1]["lr"])
        log_epoch_summary(epoch, num_epochs, loss_sum / total,
                          nat_correct / total, adv_acc=adv_correct / total)
        synchronise_bn(original_model, hardened_model, train_loader, device, population_bn)
        if val_loader is not None and epoch >= warm_up_epochs + ramp_up_epochs:
            correct = count = 0
            with preserve_model_state(original_model, hardened_model):
                original_model.eval()
                hardened_model.eval()
                for data, target in val_loader:
                    data, target = data.to(device), target.to(device)
                    radius = data.new_tensor(eps if eval_eps is None else eval_eps)
                    if val_loader.normalised:
                        radius = radius / torch.as_tensor(val_loader.std, device=data.device, dtype=data.dtype)
                    radius = radius.reshape(-1, 1, 1)
                    lower = torch.clamp(data - radius, val_loader.min.to(device), val_loader.max.to(device))
                    upper = torch.clamp(data + radius, val_loader.min.to(device), val_loader.max.to(device))
                    adversarial = pgd_attack(hardened_model, data, target, lower, upper,
                                             restarts=pgd_restarts, step_size=pgd_alpha,
                                             n_steps=pgd_steps, early_stopping=False,
                                             device=device, decay_factor=pgd_decay_factor,
                                             decay_checkpoints=pgd_decay_checkpoints)
                    with torch.no_grad():
                        correct += (hardened_model(adversarial).argmax(1) == target).sum().item()
                    count += len(data)
            accuracy = correct / count
            print(f"Validation adv_acc={accuracy:.4f}")
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_state = copy.deepcopy(hardened_model.state_dict())
        if lr_scheduler is not None and lr_decay_schedule_unit == "epoch" and (
                not lr_decay_after_ramp_up or epoch >= warm_up_epochs + ramp_up_epochs):
            lr_scheduler.step()
        if results_path is not None and (epoch + 1) % checkpoint_save_interval == 0:
            save_checkpoint(hardened_model, optimizer, loss_sum / total, epoch + 1,
                            results_path, lr_scheduler=lr_scheduler)
    if best_state is not None:
        hardened_model.load_state_dict(best_state)
        synchronise_bn(original_model, hardened_model, train_loader, device)
    return hardened_model
