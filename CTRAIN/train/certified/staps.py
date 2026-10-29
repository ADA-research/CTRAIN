"""Functional training procedure for STAPS, including schedules, validation and checkpoints."""
from CTRAIN.train.certified.util import (
    prepare_model_selection, update_model_selection, restore_model_selection,
    validate_training_run, checkpoint_due, EpochMetrics,
    evaluate_taps_proxy, synchronise_bn,
)
import torch
import torch.nn as nn
import numpy as np
from auto_LiRPA import PerturbationLpNorm
from CTRAIN.bound.util import validate_hybrid_options

from CTRAIN.train.certified.eps_scheduler import SmoothedScheduler
from CTRAIN.train.certified.losses import get_sabr_loss, get_taps_loss
from CTRAIN.train.certified.initialisation import ibp_init_shi
from CTRAIN.train.certified.regularisers import get_shi_regulariser
from CTRAIN.util import save_checkpoint
from CTRAIN.train.certified.regularisers import get_l1_reg
from CTRAIN.train.certified.progress import progress_bar, update_progress, log_epoch_summary


def staps_train_model(
    original_model,
    hardened_model,
    train_loader,
    val_loader=None,
    start_epoch=0,
    end_epoch=None,
    num_epochs=None,
    eps=0.3,
    eps_std=0.3,
    eps_schedule=(0, 20, 50),
    eps_schedule_unit="epoch",
    eps_scheduler_args=dict(),
    optimizer=None,
    subselection_ratio=0.4,
    lr_scheduler=None,
    lr_decay_schedule_unit="epoch",
    n_classes=10,
    gradient_clip=None,
    l1_regularisation_weight=0.00001,
    shi_regularisation_weight=0.5,
    shi_reg_decay=True,
    gradient_expansion_alpha=5.0,
    sabr_pgd_steps=8,
    sabr_pgd_step_size=0.5,
    sabr_pgd_restarts=1,
    sabr_pgd_early_stopping=True,
    sabr_pgd_decay_factor=0.1,
    sabr_pgd_decay_checkpoints=(4, 7),
    taps_pgd_steps=20,
    taps_pgd_step_size=None,
    taps_pgd_restarts=1,
    taps_pgd_decay_factor=0.2,
    taps_pgd_decay_checkpoints=(5, 7),
    taps_gradient_link_thresh=0.5,
    taps_gradient_link_tolerance=0.00001,
    results_path="./results",
    checkpoint_save_interval=10,
    device="cuda",
    population_bn=False,
    validation_eps=None,
    relu_upper_retention=1.0,
    pgd_eps_factor=1.0, selection_state=None, *, history=None):
    """Train a classifier using SABR feature propagation and latent TAPS searches.

    STAPS replaces the TAPS feature-bound computation with bounds over a
    contained SABR input subregion. Input and latent searches share the requested
    PGD range factor. Validation selects latent-margin proxy accuracy at the
    fixed evaluation radius after ramp-up, independently of the training radius.
    Without validation, the trainer returns the final model.

    Args:
        original_model (torch.nn.Module): The original model to be hardened.
        hardened_model (auto_LiRPA.BoundedModule): The bounded model to be trained.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training data.
        val_loader (torch.utils.data.DataLoader, optional): Optional validation data.
            Selects the best model by
            full-epsilon latent-margin
            accuracy after ramp-up;
            otherwise returns the final
            model.
        start_epoch (int, optional): Epoch to start training from.
        end_epoch (int, optional): Epoch to prematurely end training at.
        num_epochs (int, optional): Number of epochs to train the model.
        eps (float, optional): Epsilon value for perturbation.
        eps_std (float, optional): Legacy normalised-radius argument, retained for
            positional compatibility; the scheduler uses eps and
            loader metadata.
        eps_schedule (tuple, optional): Warm-up, ramp-up and remaining training durations in
            the selected scheduling unit.
        eps_schedule_unit (str, optional): Unit for epsilon schedule ('epoch' or 'batch').
        eps_scheduler_args (dict, optional): Additional arguments for the epsilon scheduler.
        optimizer (torch.optim.Optimizer, optional): Optimizer for training.
        subselection_ratio (float, optional): Ratio for subselection in SABR loss.
        lr_scheduler (torch.optim.lr_scheduler.LRScheduler, optional): Learning-rate scheduler associated with the supplied optimiser.
        lr_decay_schedule_unit (str, optional): Unit for learning rate decay schedule
            ('epoch' or 'batch').
        n_classes (int, optional): Number of classes in the dataset.
        gradient_clip (float, optional): Value for gradient clipping.
        l1_regularisation_weight (float, optional): Weight for L1 regularisation.
        shi_regularisation_weight (float, optional): Weight for SHI regularisation.
        shi_reg_decay (bool, optional): Whether to decay SHI regularisation.
        gradient_expansion_alpha (float, optional): Alpha value for gradient expansion.
        sabr_pgd_steps (int, optional): Number of PGD steps for SABR loss.
        sabr_pgd_step_size (float, optional): Step size for PGD in SABR loss.
        sabr_pgd_restarts (int, optional): Number of PGD restarts for SABR loss.
        sabr_pgd_early_stopping (bool, optional): Whether to use early stopping in PGD for
            SABR loss.
        sabr_pgd_decay_factor (float, optional): Decay factor for PGD in SABR loss.
        sabr_pgd_decay_checkpoints (tuple, optional): Checkpoints for PGD decay in SABR
            loss.
        taps_pgd_steps (int, optional): Number of PGD steps for TAPS loss.
        taps_pgd_step_size (float, optional): Step size for PGD in TAPS loss.
        taps_pgd_restarts (int, optional): Number of PGD restarts for TAPS loss.
        taps_pgd_decay_factor (float, optional): Decay factor for PGD in TAPS loss.
        taps_pgd_decay_checkpoints (tuple, optional): Checkpoints for PGD decay in TAPS
            loss.
        taps_gradient_link_thresh (float, optional): Threshold for gradient linking in TAPS
            loss.
        taps_gradient_link_tolerance (float, optional): Tolerance for gradient linking in
            TAPS loss.
        results_path (str, optional): Path to save training results.
        checkpoint_save_interval (int, optional): Interval for saving checkpoints.
        device (str, optional): Device to use for training ('cuda' or 'cpu').
        population_bn (bool, optional): Recalibrate BatchNorm on the training loader after
            each epoch.
        validation_eps (float, optional): Evaluation radius for checkpoint selection;
            defaults to training epsilon.
        relu_upper_retention (float): Retained unstable ReLU upper bound fraction for
            training; default 1.
        pgd_eps_factor (float): PGD search range factor; does not change nominal IBP bounds.
        selection_state (dict, optional): Mutable dictionary retaining the best validation score, inference
            weights and epoch across training calls.
        history (list, optional): Optional list to which the trainer appends per-epoch losses, radii and
            learning rates.

    Returns:
        (auto_LiRPA.BoundedModule): The trained bounded model.
    """
    selection = prepare_model_selection(selection_state, val_loader, validation_eps)
    validate_hybrid_options(relu_upper_retention, pgd_eps_factor)
    if num_epochs is None and eps_schedule_unit == "epoch":
        num_epochs = sum(eps_schedule)
    end_epoch = validate_training_run(train_loader, num_epochs, start_epoch, end_epoch,
                                      checkpoint_save_interval, lr_decay_schedule_unit, history=history)

    criterion = nn.CrossEntropyLoss(reduction="none")
    if start_epoch == 0 and end_epoch > start_epoch:
        ibp_init_shi(original_model, hardened_model)

    no_batches = 0
    cur_lr = optimizer.param_groups[-1]["lr"]

    eps_scheduler = SmoothedScheduler(
        num_epochs=num_epochs,
        eps=eps,
        mean=train_loader.mean,
        std=train_loader.std if train_loader.normalised else torch.ones_like(train_loader.std),
        eps_schedule_unit=eps_schedule_unit,
        eps_schedule=eps_schedule,
        batches_per_epoch=len(train_loader),
        start_epoch=start_epoch,
        **eps_scheduler_args,
    )

    cur_eps = eps_scheduler.get_cur_eps()

    for epoch in range(start_epoch, end_epoch):

        if start_epoch > epoch:
            continue

        epoch_nat_err = 0
        epoch_rob_err = 0

        original_model.train()
        hardened_model.train()
        for block in hardened_model.bounded_blocks:
            block.train()

        metrics = EpochMetrics()

        progress = progress_bar(
            train_loader,
            epoch=epoch,
            num_epochs=num_epochs,
            eps=eps_scheduler.get_cur_eps(normalise=False),
            method="STAPS",
        )
        for batch_idx, (data, target) in enumerate(progress):

            if start_epoch > epoch:
                eps_scheduler.batch_step()
                continue

            cur_eps = eps_scheduler.get_cur_eps().reshape(-1, 1, 1)
            data, target = data.to(device), target.to(device)
            cur_eps_device = torch.as_tensor(cur_eps, device=device)
            data_min, data_max = train_loader.min.to(device), train_loader.max.to(device)

            ptb = PerturbationLpNorm(
                eps=cur_eps_device,
                norm=np.inf,
                x_L=torch.clamp(data - cur_eps_device, data_min, data_max),
                x_U=torch.clamp(data + cur_eps_device, data_min, data_max),
            )

            optimizer.zero_grad()
            clean_output = hardened_model(data)
            regular_err = torch.sum(
                torch.argmax(clean_output, dim=1) != target
            ).item() / data.size(0)
            epoch_nat_err += regular_err
            clean_loss = criterion(clean_output, target).mean()

            if eps_scheduler.get_cur_eps(normalise=False) == 0.0:
                loss = clean_loss
                epoch_rob_err += regular_err
            elif eps_scheduler.get_cur_eps(normalise=False) != 0.0 and (
                eps_scheduler.get_cur_eps(normalise=False)
                != eps_scheduler.get_max_eps(normalise=False)
            ):
                reg_loss, robust_err, adv_err = get_sabr_loss(
                    relu_upper_retention=relu_upper_retention,
                    pgd_eps_factor=pgd_eps_factor,
                    hardened_model=hardened_model,
                    original_model=original_model,
                    data_max=data_max,
                    data_min=data_min,
                    data=data,
                    target=target,
                    eps=cur_eps_device,
                    subselection_ratio=subselection_ratio,
                    criterion=criterion,
                    device=device,
                    n_classes=n_classes,
                    pgd_steps=sabr_pgd_steps,
                    pgd_step_size=sabr_pgd_step_size,
                    pgd_restarts=sabr_pgd_restarts,
                    pgd_early_stopping=sabr_pgd_early_stopping,
                    pgd_decay_checkpoints=sabr_pgd_decay_checkpoints,
                    pgd_decay_factor=sabr_pgd_decay_factor,
                    return_stats=True,
                )

                loss_regularisers = get_shi_regulariser(
                    relu_upper_retention=relu_upper_retention,
                    model=hardened_model,
                    ptb=ptb,
                    data=data,
                    target=target,
                    eps_scheduler=eps_scheduler,
                    n_classes=n_classes,
                    device=device,
                    included_regularisers=["relu", "tightness"],
                    verbose=False,
                    regularisation_decay=shi_reg_decay,
                )
                epoch_rob_err += robust_err
                loss_regularisers = shi_regularisation_weight * loss_regularisers
                loss = reg_loss + loss_regularisers

            elif eps_scheduler.get_cur_eps(
                normalise=False
            ) == eps_scheduler.get_max_eps(normalise=False):
                sabr_args = dict(
                    hardened_model=hardened_model,
                    original_model=original_model,
                    data_max=data_max,
                    data_min=data_min,
                    data=data,
                    target=target,
                    eps=cur_eps_device,
                    subselection_ratio=subselection_ratio,
                    device=device,
                    n_classes=n_classes,
                    n_steps=sabr_pgd_steps,
                    step_size=sabr_pgd_step_size,
                    restarts=sabr_pgd_restarts,
                    early_stopping=sabr_pgd_early_stopping,
                    decay_checkpoints=sabr_pgd_decay_checkpoints,
                    decay_factor=sabr_pgd_decay_factor,
                )
                loss, robust_err = get_taps_loss(
                    relu_upper_retention=relu_upper_retention,
                    pgd_eps_factor=pgd_eps_factor,
                    original_model=original_model,
                    hardened_model=hardened_model,
                    bounded_blocks=hardened_model.bounded_blocks,
                    criterion=criterion,
                    data=data,
                    target=target,
                    n_classes=n_classes,
                    ptb=ptb,
                    device=device,
                    pgd_steps=taps_pgd_steps,
                    pgd_restarts=taps_pgd_restarts,
                    pgd_step_size=taps_pgd_step_size,
                    pgd_decay_checkpoints=taps_pgd_decay_checkpoints,
                    pgd_decay_factor=taps_pgd_decay_factor,
                    gradient_link_thresh=taps_gradient_link_thresh,
                    gradient_link_tolerance=taps_gradient_link_tolerance,
                    gradient_expansion_alpha=gradient_expansion_alpha,
                    propagation="SABR",
                    sabr_args=sabr_args,
                    return_stats=True,
                )

                epoch_rob_err += robust_err

            if l1_regularisation_weight is not None:
                l1_regularisation = l1_regularisation_weight * get_l1_reg(
                    model=original_model, device=device
                )
                loss += l1_regularisation

            loss.backward()

            if gradient_clip is not None:
                nn.utils.clip_grad_norm_(
                    original_model.parameters(), max_norm=gradient_clip
                )

            optimizer.step()

            metrics.update(len(data), loss=loss.detach())
            last_eps = float(eps_scheduler.get_cur_eps(normalise=False))
            completed_batches = batch_idx + 1
            update_progress(
                progress,
                loss=metrics.mean('loss'),
                nat_acc=1 - epoch_nat_err / completed_batches,
                robust_acc=1 - epoch_rob_err / completed_batches,
                lr=optimizer.param_groups[-1]["lr"],
            )
            eps_scheduler.batch_step()
            no_batches += 1

            if lr_scheduler is not None and lr_decay_schedule_unit == "batch":
                lr_scheduler.step()

        train_acc_nat = 1 - epoch_nat_err / len(train_loader)
        train_acc_cert = 1 - epoch_rob_err / len(train_loader)

        log_epoch_summary(
            epoch=epoch,
            num_epochs=num_epochs,
            loss=metrics.mean('loss'),
            nat_acc=train_acc_nat,
            robust_acc=train_acc_cert,
        )

        synchronise_bn(original_model, hardened_model, train_loader, device, population_bn)

        # Compare only at the fixed evaluation radius, after the ramp-up phase.
        if val_loader is not None and cur_eps.equal(eps_scheduler.get_max_eps().reshape(-1, 1, 1)):
            eval_eps = eps if validation_eps is None else validation_eps
            val_eps = eval_eps / val_loader.std if val_loader.normalised else torch.tensor(eval_eps)
            accuracy = evaluate_taps_proxy(
                original_model, hardened_model, val_loader, val_eps, n_classes, device,
                relu_upper_retention=relu_upper_retention, pgd_eps_factor=pgd_eps_factor,
                pgd_steps=taps_pgd_steps, pgd_restarts=taps_pgd_restarts,
                pgd_step_size=taps_pgd_step_size,
                pgd_decay_factor=taps_pgd_decay_factor,
                pgd_decay_checkpoints=taps_pgd_decay_checkpoints,
                gradient_link_thresh=taps_gradient_link_thresh,
                gradient_link_tolerance=taps_gradient_link_tolerance,
                propagation="SABR",
                sabr_args=dict(
                    subselection_ratio=subselection_ratio, n_steps=sabr_pgd_steps,
                    step_size=sabr_pgd_step_size, restarts=sabr_pgd_restarts,
                    early_stopping=sabr_pgd_early_stopping,
                    decay_factor=sabr_pgd_decay_factor,
                    decay_checkpoints=sabr_pgd_decay_checkpoints,
                ),
            )
            print(f"Validation robust_proxy_acc={accuracy:.4f}")
            update_model_selection(selection, hardened_model, accuracy, epoch + 1)

        if lr_scheduler is not None and lr_decay_schedule_unit == "epoch":
            lr_scheduler.step()

        if history is not None:
            history.append(dict(epoch=epoch + 1, loss=metrics.mean('loss'),
                                eps=last_eps,
                                lr=optimizer.param_groups[-1]['lr']))
        if results_path is not None and checkpoint_due(epoch + 1, end_epoch, checkpoint_save_interval):
            save_checkpoint(
                hardened_model, optimizer, metrics.mean('loss'), epoch + 1, results_path,
                lr_scheduler=lr_scheduler, selection_state=selection,
                extra_state=None if history is None else {'history': history},
            )

    if val_loader is not None and restore_model_selection(selection, hardened_model):
        synchronise_bn(original_model, hardened_model, train_loader, device)
    return hardened_model
