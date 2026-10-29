"""Functional training procedure for CROWN-IBP, including schedules, validation and checkpoints."""
from CTRAIN.train.certified.util import (
    prepare_model_selection, update_model_selection, restore_model_selection,
    validate_training_run, checkpoint_due, EpochMetrics,
    evaluate_validation_loss, validation_ptb, synchronise_bn,
)
import torch
import torch.nn as nn
import numpy as np
from auto_LiRPA import PerturbationLpNorm

from CTRAIN.train.certified.eps_scheduler import SmoothedScheduler
from CTRAIN.train.certified.losses import get_crown_ibp_loss
from CTRAIN.util import save_checkpoint
from CTRAIN.train.certified.losses.crown_ibp import get_exp_module
from CTRAIN.train.certified.regularisers import get_l1_reg, get_shi_regulariser
from CTRAIN.train.certified.initialisation import ibp_init_shi
from CTRAIN.train.certified.progress import progress_bar, update_progress, log_epoch_summary


def crown_ibp_train_model(
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
    lr_scheduler=None,
    lr_decay_schedule_unit="epoch",
    n_classes=10,
    loss_fusion=False,
    loss_fusion_model=None,
    gradient_clip=None,
    l1_regularisation_weight=0.00001,
    shi_regularisation_weight=1,
    shi_reg_decay=1,
    results_path="./results",
    checkpoint_save_interval=10,
    device="cuda",
    population_bn=False, validation_eps=None, selection_state=None, *, history=None):
    """Train a classifier using mixed CROWN-IBP and IBP margin bounds.

    The perturbation and mixing schedules introduce the certified objective
    after warm-up. Optional loss fusion bounds the classifier and loss together.
    The trainer owns optimiser updates, validation, BatchNorm synchronisation
    and trajectory checkpoints.

    Args:
        original_model (torch.nn.Module): The original model to be trained.
        hardened_model (auto_LiRPA.BoundedModule): The bounded model to be trained.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training data.
        val_loader (torch.utils.data.DataLoader, optional): DataLoader for the validation
            data.
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
        eps_scheduler_args (dict, optional): Additional arguments for epsilon scheduler.
        optimizer (torch.optim.Optimizer, optional): Optimizer for training.
        lr_scheduler (torch.optim.lr_scheduler.LRScheduler, optional): Learning-rate scheduler associated with the supplied optimiser.
        lr_decay_schedule_unit (str, optional): Unit for learning rate decay schedule
            ('epoch' or 'batch').
        n_classes (int, optional): Number of classes in the dataset.
        loss_fusion (bool, optional): Whether to use loss fusion.
        loss_fusion_model (auto_LiRPA.BoundedModule, optional): Model to use for loss
            fusion.
        gradient_clip (float, optional): Value for gradient clipping.
        l1_regularisation_weight (float, optional): Weight for L1 regularisation.
        shi_regularisation_weight (float, optional): Weight for SHI regularisation.
        shi_reg_decay (float, optional): Decay factor for SHI regularisation.
        results_path (str, optional): Path to save the training results.
        checkpoint_save_interval (int, optional): Interval for saving checkpoints.
        device (str, optional): Device to use for training ('cuda' or 'cpu').
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
    if num_epochs is None and eps_schedule_unit == "epoch":
        num_epochs = sum(eps_schedule)
    end_epoch = validate_training_run(train_loader, num_epochs, start_epoch, end_epoch,
                                      checkpoint_save_interval, lr_decay_schedule_unit, history=history)

    criterion = nn.CrossEntropyLoss(reduction='none')
    
    no_batches = 0
    cur_lr = optimizer.param_groups[-1]["lr"]

    # Not done in original paper - however it is generally beneficial
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

    if start_epoch == 0 and end_epoch > start_epoch:
        # Not done in original paper - however it is SotA and generally beneficial
        ibp_init_shi(original_model, hardened_model)

    cur_eps, kappa = eps_scheduler.get_cur_eps(), eps_scheduler.get_cur_kappa()

    for epoch in range(start_epoch, end_epoch):

        epoch_rob_err = 0
        epoch_nat_err = 0

        hardened_model.train()
        original_model.train()
        loss_fusion_model.train() if loss_fusion else None

        metrics = EpochMetrics()

        progress = progress_bar(
            train_loader,
            epoch=epoch,
            num_epochs=num_epochs,
            eps=eps_scheduler.get_cur_eps(normalise=False),
            method="CROWN-IBP",
        )
        for batch_idx, (data, target) in enumerate(progress):

            cur_eps = eps_scheduler.get_cur_eps().reshape(-1, 1, 1)
            kappa = eps_scheduler.get_cur_kappa()
            beta = eps_scheduler.get_cur_beta()
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
            
            if loss_fusion:
                clean_output = loss_fusion_model(data, target)
                clean_loss = torch.mean(
                    torch.log(clean_output) + get_exp_module(loss_fusion_model).max_input
                )
                with torch.no_grad():
                    regular_err = torch.sum(
                        torch.argmax(hardened_model(data), dim=1) != target
                    ).item() / data.size(0)
                    epoch_nat_err += regular_err
            else:
                clean_output = hardened_model(data)
                clean_loss = criterion(clean_output, target).mean()
                regular_err = torch.sum(
                    torch.argmax(clean_output, dim=1) != target
                ).item() / data.size(0)
                epoch_nat_err += regular_err

            if eps_scheduler.get_cur_eps(normalise=False) != 0.0:
                certified_loss, robust_err = get_crown_ibp_loss(
                    hardened_model=hardened_model if not loss_fusion else loss_fusion_model,
                    ptb=ptb,
                    data=data,
                    target=target,
                    n_classes=n_classes,
                    criterion=criterion,
                    beta=beta,
                    loss_fusion=loss_fusion,
                    return_bounds=False,
                    return_stats=True,
                )

                epoch_rob_err += robust_err

                loss = kappa * clean_loss + (1 - kappa) * certified_loss
            else:
                loss = clean_loss

            if eps_scheduler.get_cur_eps(normalise=False) != eps_scheduler.get_max_eps(
                normalise=False
            ):
                # Not done in original paper - however it is SotA and generally beneficial
                loss_regularisers = get_shi_regulariser(
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
                    loss_fusion=loss_fusion,
                )

                loss_regularisers = shi_regularisation_weight * loss_regularisers
                loss = loss + loss_regularisers

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
                cert_acc=1 - epoch_rob_err / completed_batches,
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
            cert_acc=train_acc_cert,
        )

        synchronise_bn(original_model, hardened_model, train_loader, device, population_bn, loss_fusion_model=loss_fusion_model if loss_fusion else None)

        if lr_scheduler is not None and lr_decay_schedule_unit == "epoch":
            lr_scheduler.step()

        if val_loader is not None and cur_eps.equal(eps_scheduler.get_max_eps().reshape(-1, 1, 1)):
            def validation_loss(data, target):
                region = validation_ptb(data, val_loader, eps if validation_eps is None else validation_eps)
                if loss_fusion:
                    clean = loss_fusion_model(data, target)
                    clean_loss = (torch.log(clean) + get_exp_module(loss_fusion_model).max_input).mean()
                else:
                    clean_loss = criterion(hardened_model(data), target).mean()
                robust_loss = get_crown_ibp_loss(
                    hardened_model=hardened_model if not loss_fusion else loss_fusion_model,
                    ptb=region,
                    data=data,
                    target=target,
                    n_classes=n_classes,
                    criterion=criterion,
                    beta=beta,
                    loss_fusion=loss_fusion,
                    return_bounds=False,
                    return_stats=False
                )[0]
                return kappa * clean_loss + (1 - kappa) * robust_loss
            models = (original_model, hardened_model) + ((loss_fusion_model,) if loss_fusion else ())
            validation = evaluate_validation_loss(models, val_loader, validation_loss, device)
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
        synchronise_bn(original_model, hardened_model, train_loader, device, loss_fusion_model=loss_fusion_model if loss_fusion else None)
    return hardened_model
