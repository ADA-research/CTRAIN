"""Functional training procedure for SABR, including schedules, validation and checkpoints."""
from CTRAIN.train.certified.util import (
    prepare_model_selection, update_model_selection, restore_model_selection,
    validate_training_run, checkpoint_due, EpochMetrics,
    evaluate_validation_loss, validation_ptb, synchronise_bn,
)
from CTRAIN.util.util import loader_limits
from concurrent.futures import ProcessPoolExecutor
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from auto_LiRPA import BoundedModule

from auto_LiRPA.perturbations import PerturbationLpNorm
from CTRAIN.eval import eval_acc, eval_certified, eval_epoch, eval_adversarial
from CTRAIN.bound import bound_ibp
from CTRAIN.train.certified.eps_scheduler import SmoothedScheduler
from CTRAIN.bound.util import validate_hybrid_options
from CTRAIN.train.certified.losses import get_sabr_loss
from CTRAIN.train.certified.initialisation import ibp_init_shi
from CTRAIN.train.certified.regularisers.shi import get_shi_regulariser
from CTRAIN.util import save_checkpoint
from CTRAIN.train.certified.regularisers import get_l1_reg
from CTRAIN.train.certified.progress import progress_bar, update_progress, log_epoch_summary


def sabr_train_model(
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
    shi_regularisation_weight=1,
    shi_reg_decay=True,
    pgd_steps=8,
    pgd_step_size=0.5,
    pgd_restarts=1,
    pgd_early_stopping=True,
    pgd_decay_factor=0.1,
    pgd_decay_checkpoints=(4, 7),
    pgd_eps_factor=1,
    results_path="./results",
    checkpoint_save_interval=10,
    device="cuda",
    relu_upper_retention=1.0,
    population_bn=False, validation_eps=None, selection_state=None, *, history=None):
    """Train a classifier on adversarially selected SABR subregions.

    PGD selects difficult inputs, and IBP bounds a contained smaller region
    around each selected point. Warm-up, perturbation ramp-up, regularisation
    and checkpointing are handled by this training loop. Training bounds on
    selected subregions are surrogates for full-region robustness.

    Args:
        original_model (torch.nn.Module): The original model to be hardened.
        hardened_model (torch.nn.Module): The model to be trained with SABR.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training data.
        val_loader (torch.utils.data.DataLoader, optional): DataLoader for the validation
            data.
        start_epoch (int, optional): Epoch to start training from.
        end_epoch (int, optional): Epoch to prematurely end training at.
        num_epochs (int, optional): Number of epochs to train the model.
        eps (float, optional): Initial epsilon value for adversarial perturbations.
        eps_std (float, optional): Legacy normalised-radius argument, retained for
            positional compatibility; the scheduler uses eps and
            loader metadata.
        eps_schedule (tuple, optional): Warm-up, ramp-up and remaining training durations in
            the selected scheduling unit.
        eps_schedule_unit (str, optional): Unit for epsilon scheduling ('epoch' or 'batch').
        eps_scheduler_args (dict, optional): Additional arguments for the epsilon scheduler.
        optimizer (torch.optim.Optimizer, optional): Optimizer for training the model.
        subselection_ratio (float, optional): Ratio for subselection in SABR.
        lr_scheduler (torch.optim.lr_scheduler.LRScheduler, optional): Learning-rate scheduler associated with the supplied optimiser.
        lr_decay_schedule_unit (str, optional): Unit for learning rate decay scheduling
            ('epoch' or 'batch').
        n_classes (int, optional): Number of classes in the dataset.
        gradient_clip (float, optional): Value for gradient clipping.
        l1_regularisation_weight (float, optional): Weight for L1 regularisation.
        shi_regularisation_weight (float, optional): Weight for Shi regularisation.
        shi_reg_decay (bool, optional): Whether to decay Shi regularisation.
        pgd_steps (int, optional): Number of steps for PGD (Projected Gradient Descent).
        pgd_step_size (float, optional): Step size for PGD.
        pgd_restarts (int, optional): Number of restarts for PGD.
        pgd_early_stopping (bool, optional): Whether to use early stopping for PGD.
        pgd_decay_factor (float, optional): Decay factor for PGD.
        pgd_decay_checkpoints (tuple, optional): Checkpoints for PGD decay.
        pgd_eps_factor (float, optional): Factor for PGD epsilon.
        results_path (str, optional): Path to save the training results.
        checkpoint_save_interval (int, optional): Interval for saving checkpoints.
        device (str, optional): Device to use for training ('cuda' or 'cpu').
        relu_upper_retention (float): Retained unstable ReLU upper bound fraction for
            training; default 1.
        population_bn (bool): Recalibrate BatchNorm after each epoch; default False.
        validation_eps (float or torch.Tensor, optional): Fixed raw-input radius used for validation; None uses the training
            target.
        selection_state (dict, optional): Mutable dictionary retaining the best validation score, inference
            weights and epoch across training calls.
        history (list, optional): Optional list to which the trainer appends per-epoch losses, radii and
            learning rates.

    Returns:
        (auto_LiRPA.BoundedModule): The trained hardened model.
    """

    selection = prepare_model_selection(selection_state, val_loader, validation_eps)
    validate_hybrid_options(relu_upper_retention, pgd_eps_factor)
    if num_epochs is None and eps_schedule_unit == "epoch":
        num_epochs = sum(eps_schedule)
    end_epoch = validate_training_run(train_loader, num_epochs, start_epoch, end_epoch,
                                      checkpoint_save_interval, lr_decay_schedule_unit, history=history)

    criterion = nn.CrossEntropyLoss(reduction="none")

    if start_epoch == 0 and end_epoch > start_epoch:
        # Important Change to Vanilla IBP: Initialise Weights to normal distribution with sigma_i = sqrt(2*pi)/n_i, for layer i and fan in n_i
        ibp_init_shi(original_model, hardened_model)

    no_batches = 0
    cur_lr = optimizer.param_groups[-1]["lr"]

    # Important Change to Vanilla IBP: Schedule Eps smoothly
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
    # Training loop
    for epoch in range(start_epoch, end_epoch):

        epoch_adv_err = 0
        epoch_rob_err = 0
        epoch_nat_err = 0

        hardened_model.train()
        original_model.train()
        metrics = EpochMetrics()

        progress = progress_bar(
            train_loader,
            epoch=epoch,
            num_epochs=num_epochs,
            eps=eps_scheduler.get_cur_eps(normalise=False),
            method="SABR",
        )
        for batch_idx, (data, target) in enumerate(progress):

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
            if eps_scheduler.get_cur_eps(normalise=False) != 0.0:
                
                sabr_loss, robust_err, adv_err = get_sabr_loss(
                    relu_upper_retention=relu_upper_retention,
                    pgd_eps_factor=pgd_eps_factor,
                    hardened_model=hardened_model,
                    original_model=original_model,
                    train_loader=train_loader,
                    data=data,
                    data_min=data_min,
                    data_max=data_max,
                    target=target,
                    eps=cur_eps_device,
                    subselection_ratio=subselection_ratio,
                    criterion=criterion,
                    device=device,
                    n_classes=n_classes,
                    pgd_steps=pgd_steps,
                    pgd_step_size=pgd_step_size,
                    pgd_restarts=pgd_restarts,
                    pgd_early_stopping=pgd_early_stopping,
                    pgd_decay_factor=pgd_decay_factor,
                    pgd_decay_checkpoints=pgd_decay_checkpoints,
                    return_stats=True,
                )

                epoch_adv_err += adv_err
                epoch_rob_err += robust_err

                loss = sabr_loss

            else:
                loss = criterion(clean_output, target).mean()

            if eps_scheduler.get_cur_eps(normalise=False) != eps_scheduler.get_max_eps(
                normalise=False
            ):
                # Use small boxes for the calculation of the Shi regulariser (in line with Müller et al. 2021)
                # usually, this should be done since the regularisation function uses previously computed bounds
                # however, sometimes these bounds may not be available
                reg_ptb = PerturbationLpNorm(
                    eps=cur_eps_device * subselection_ratio,
                    norm=np.inf,
                    x_L=torch.clamp(data - cur_eps_device, data_min, data_max),
                    x_U=torch.clamp(data + cur_eps_device, data_min, data_max),
                )
                # SABR also uses Shi regularisation during warm up/ramp-up
                loss_regularisers = get_shi_regulariser(
                    relu_upper_retention=relu_upper_retention,
                    model=hardened_model,
                    ptb=reg_ptb,
                    data=data,
                    target=target,
                    eps_scheduler=eps_scheduler,
                    n_classes=n_classes,
                    device=device,
                    included_regularisers=["relu", "tightness"],
                    verbose=False,
                    regularisation_decay=shi_reg_decay,
                )
                loss = loss + shi_regularisation_weight * loss_regularisers

            if l1_regularisation_weight is not None:
                l1_regularisation = l1_regularisation_weight * get_l1_reg(
                    model=original_model, device=device
                )
                loss = loss + l1_regularisation

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
                adv_acc=1 - epoch_adv_err / completed_batches,
                robust_acc=1 - epoch_rob_err / completed_batches,
                lr=optimizer.param_groups[-1]["lr"],
            )
            eps_scheduler.batch_step()
            no_batches += 1

            if lr_scheduler is not None and lr_decay_schedule_unit == "batch":
                lr_scheduler.step()

        train_acc_nat = 1 - epoch_nat_err / len(train_loader)
        train_acc_adv = 1 - epoch_adv_err / len(train_loader)
        train_acc_cert = 1 - epoch_rob_err / len(train_loader)

        log_epoch_summary(
            epoch=epoch,
            num_epochs=num_epochs,
            loss=metrics.mean('loss'),
            nat_acc=train_acc_nat,
            adv_acc=train_acc_adv,
            robust_acc=train_acc_cert,
        )

        synchronise_bn(original_model, hardened_model, train_loader, device, population_bn)

        if lr_scheduler is not None and lr_decay_schedule_unit == "epoch":
            lr_scheduler.step()

        if val_loader is not None and cur_eps.equal(eps_scheduler.get_max_eps().reshape(-1, 1, 1)):
            def validation_loss(data, target):
                region = validation_ptb(data, val_loader, eps if validation_eps is None else validation_eps)
                radius = region.eps
                minimum, maximum = loader_limits(val_loader, data)
                robust_loss = get_sabr_loss(
                    relu_upper_retention=relu_upper_retention,
                    pgd_eps_factor=pgd_eps_factor,
                    hardened_model=hardened_model,
                    original_model=original_model,
                    train_loader=val_loader,
                    data=data,
                    data_min=minimum,
                    data_max=maximum,
                    target=target,
                    eps=radius,
                    subselection_ratio=subselection_ratio,
                    criterion=criterion,
                    device=device,
                    n_classes=n_classes,
                    pgd_steps=pgd_steps,
                    pgd_step_size=pgd_step_size,
                    pgd_restarts=pgd_restarts,
                    pgd_early_stopping=pgd_early_stopping,
                    pgd_decay_factor=pgd_decay_factor,
                    pgd_decay_checkpoints=pgd_decay_checkpoints,
                    return_stats=False
                )[0]
                return robust_loss
            validation = evaluate_validation_loss((original_model, hardened_model), val_loader, validation_loss, device)
            print(f"Validation robust_loss={validation:.6f}")
            update_model_selection(selection, hardened_model, -validation, epoch + 1)

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
