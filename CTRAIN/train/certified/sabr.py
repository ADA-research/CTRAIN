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
from CTRAIN.train.certified.losses import get_sabr_loss
from CTRAIN.train.certified.initialisation import ibp_init_shi
from CTRAIN.train.certified.regularisers.shi import get_shi_regulariser
from CTRAIN.util import save_checkpoint
from CTRAIN.train.certified.regularisers import get_l1_reg


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
    lr_decay_schedule=(15, 25),
    lr_decay_factor=0.2,
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
):
    """
    Trains a model using the SABR method.

    Args:
        original_model (torch.nn.Module): The original model to be hardened.
        hardened_model (torch.nn.Module): The model to be trained with SABR.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training data.
        val_loader (torch.utils.data.DataLoader, optional): DataLoader for the validation data. Defaults to None.
        start_epoch (int, optional): Epoch to start training from. Defaults to 0.
        end_epoch (int, optional): Epoch to prematurely end training at. Defaults to None.
        num_epochs (int, optional): Number of epochs to train the model. Defaults to None.
        eps (float, optional): Initial epsilon value for adversarial perturbations. Defaults to 0.3.
        eps_std (float, optional): Standardised epsilon value. Defaults to 0.3.
        eps_schedule (tuple, optional): Schedule for epsilon values. Defaults to (0, 20, 50).
        eps_schedule_unit (str, optional): Unit for epsilon scheduling ('epoch' or 'batch'). Defaults to 'epoch'.
        eps_scheduler_args (dict, optional): Additional arguments for the epsilon scheduler. Defaults to an empty dictionary.
        optimizer (torch.optim.Optimizer, optional): Optimizer for training the model. Defaults to None.
        subselection_ratio (float, optional): Ratio for subselection in SABR. Defaults to 0.4.
        lr_decay_schedule (tuple, optional): Schedule for learning rate decay. Defaults to (15, 25).
        lr_decay_factor (float, optional): Factor by which to decay the learning rate. Defaults to .2.
        lr_decay_schedule_unit (str, optional): Unit for learning rate decay scheduling ('epoch' or 'batch'). Defaults to 'epoch'.
        n_classes (int, optional): Number of classes in the dataset. Defaults to 10.
        gradient_clip (float, optional): Value for gradient clipping. Defaults to None.
        l1_regularisation_weight (float, optional): Weight for L1 regularization. Defaults to 0.00001.
        shi_regularisation_weight (float, optional): Weight for Shi regularization. Defaults to 1.
        shi_reg_decay (bool, optional): Whether to decay Shi regularization. Defaults to True.
        pgd_steps (int, optional): Number of steps for PGD (Projected Gradient Descent). Defaults to 8.
        pgd_step_size (float, optional): Step size for PGD. Defaults to 0.5.
        pgd_restarts (int, optional): Number of restarts for PGD. Defaults to 1.
        pgd_early_stopping (bool, optional): Whether to use early stopping for PGD. Defaults to True.
        pgd_decay_factor (float, optional): Decay factor for PGD. Defaults to 0.1.
        pgd_decay_checkpoints (tuple, optional): Checkpoints for PGD decay. Defaults to (4, 7).
        pgd_eps_factor (float, optional): Factor for PGD epsilon. Defaults to 1.
        results_path (str, optional): Path to save the training results. Defaults to "./results".
        checkpoint_save_interval (int, optional): Interval for saving checkpoints. Defaults to 10.
        device (str, optional): Device to use for training ('cuda' or 'cpu'). Defaults to 'cuda'.

    Returns:
        (auto_LiRPA.BoundedModule): The trained hardened model.
    """

    if end_epoch is None:
        end_epoch = num_epochs

    criterion = nn.CrossEntropyLoss(reduction="none")

    if start_epoch == 0:
        # Important Change to Vanilla IBP: Initialise Weights to normal distribution with sigma_i = sqrt(2*pi)/n_i, for layer i and fan in n_i
        ibp_init_shi(original_model, hardened_model)

    no_batches = 0
    cur_lr = optimizer.param_groups[-1]["lr"]

    # Important Change to Vanilla IBP: Schedule Eps smoothly
    eps_scheduler = SmoothedScheduler(
        num_epochs=num_epochs,
        eps=eps,
        mean=train_loader.mean,
        std=train_loader.std,
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

        if lr_decay_schedule_unit == "epoch":
            if epoch + 1 in lr_decay_schedule:
                print("LEARNING RATE DECAYED!")
                cur_lr = cur_lr * lr_decay_factor
                for g in optimizer.param_groups:
                    g["lr"] = cur_lr

        print(
            f"[{epoch + 1}/{num_epochs}]: eps {eps_scheduler.get_cur_eps(normalise=False):.4f}"
        )
        hardened_model.train()
        original_model.train()
        running_loss = 0.0

        for batch_idx, (data, target) in enumerate(train_loader):

            cur_eps = eps_scheduler.get_cur_eps().reshape(-1, 1, 1)

            ptb = PerturbationLpNorm(
                eps=cur_eps,
                norm=np.inf,
                x_L=torch.clamp(data - cur_eps, train_loader.min, train_loader.max).to(
                    device
                ),
                x_U=torch.clamp(data + cur_eps, train_loader.min, train_loader.max).to(
                    device
                ),
            )

            if lr_decay_schedule_unit == "batch":
                if no_batches + 1 in lr_decay_schedule:
                    print("LEARNING RATE DECAYED!")
                    cur_lr = cur_lr * lr_decay_factor
                    for g in optimizer.param_groups:
                        g["lr"] = cur_lr

            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            clean_output = hardened_model(data)
            regular_err = torch.sum(
                torch.argmax(clean_output, dim=1) != target
            ).item() / data.size(0)
            epoch_nat_err += regular_err
            if eps_scheduler.get_cur_eps(normalise=False) != 0.0:
                
                if pgd_eps_factor == 1:
                    pgd_ptb = ptb
                else:
                    pgd_eps = (cur_eps * pgd_eps_factor).to(device)
                    data_min, data_max = train_loader.min.to(
                        device
                    ), train_loader.max.to(device)
                    pgd_ptb = PerturbationLpNorm(
                        eps=pgd_eps,
                        norm=np.inf,
                        x_L=torch.clamp(data - pgd_eps, data_min, data_max).to(device),
                        x_U=torch.clamp(data + pgd_eps, data_min, data_max).to(device),
                    )

                sabr_loss, robust_err, adv_err = get_sabr_loss(
                    hardened_model=hardened_model,
                    original_model=original_model,
                    train_loader=train_loader,
                    data=data,
                    data_min=train_loader.min.to(device),
                    data_max=train_loader.max.to(device),
                    target=target,
                    eps=torch.tensor(cur_eps, device=device),
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
                    pgd_ptb=pgd_ptb,
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
                # SABR also uses Shi regularisation during warm up/ramp-up
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

            running_loss += loss.item()
            eps_scheduler.batch_step()
            no_batches += 1

        train_acc_nat = 1 - epoch_nat_err / len(train_loader)
        train_acc_adv = 1 - epoch_adv_err / len(train_loader)
        train_acc_cert = 1 - epoch_rob_err / len(train_loader)

        print(
            f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {running_loss/len(train_loader):.4f}"
        )
        print(f"\t Natural Acc. Train: {train_acc_nat:.4f}")
        print(f"\t Adv. Acc. Train: {train_acc_adv:.4f}")
        print(f"\t Certified Acc. Train: {train_acc_cert:.4f}")

        if results_path is not None and (epoch + 1) % checkpoint_save_interval == 0:
            save_checkpoint(
                hardened_model, optimizer, running_loss, epoch + 1, results_path
            )

    return hardened_model
