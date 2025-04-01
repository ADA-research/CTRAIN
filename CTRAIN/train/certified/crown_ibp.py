import torch
import torch.nn as nn
import numpy as np
from auto_LiRPA import PerturbationLpNorm

from CTRAIN.train.certified.eps_scheduler import SmoothedScheduler
from CTRAIN.train.certified.losses import get_crown_ibp_loss
from CTRAIN.util import save_checkpoint
from CTRAIN.train.certified.regularisers import get_l1_reg, get_shi_regulariser
from CTRAIN.train.certified.initialisation import ibp_init_shi


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
    lr_decay_schedule=(15, 25),
    lr_decay_factor=0.2,
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
):
    """
    Train a model using the CROWN-IBP method.

    Args:
        original_model (torch.nn.Module): The original model to be trained.
        hardened_model (auto_LiRPA.BoundedModule): The bounded model to be trained.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training data.
        val_loader (torch.utils.data.DataLoader, optional): DataLoader for the validation data. Defaults to None.
        start_epoch (int, optional): Epoch to start training from. Defaults to 0.
        end_epoch (int, optional): Epoch to prematurely end training at. Defaults to None.
        num_epochs (int, optional): Number of epochs to train the model. Defaults to None.
        eps (float, optional): Epsilon value for perturbation. Defaults to 0.3.
        eps_std (float, optional): Standardised epsilon value. Defaults to 0.3.
        eps_schedule (tuple, optional): Schedule for epsilon values. Defaults to (0, 20, 50).
        eps_schedule_unit (str, optional): Unit for epsilon schedule ('epoch' or 'batch'). Defaults to 'epoch'.
        eps_scheduler_args (dict, optional): Additional arguments for epsilon scheduler. Defaults to dict().
        optimizer (torch.optim.Optimizer, optional): Optimizer for training. Defaults to None.
        lr_decay_schedule (tuple, optional): Schedule for learning rate decay. Defaults to (15, 25).
        lr_decay_factor (float, optional): Factor by which to decay the learning rate. Defaults to .2.
        lr_decay_schedule_unit (str, optional): Unit for learning rate decay schedule ('epoch' or 'batch'). Defaults to 'epoch'.
        n_classes (int, optional): Number of classes in the dataset. Defaults to 10.
        loss_fusion (bool, optional): Whether to use loss fusion. Defaults to True.
        loss_fusion_model (auto_LiRPA.BoundedModule, optional): Model to use for loss fusion. Defaults to None.
        gradient_clip (float, optional): Value for gradient clipping. Defaults to None.
        l1_regularisation_weight (float, optional): Weight for L1 regularization. Defaults to 0.00001.
        shi_regularisation_weight (float, optional): Weight for SHI regularization. Defaults to 1.
        shi_reg_decay (float, optional): Decay factor for SHI regularization. Defaults to 1.
        results_path (str, optional): Path to save the training results. Defaults to "./results".
        checkpoint_save_interval (int, optional): Interval for saving checkpoints. Defaults to 10.
        device (str, optional): Device to use for training ('cuda' or 'cpu'). Defaults to 'cuda'.

    Returns:
        (auto_LiRPA.BoundedModule): The trained hardened model.
    """

    if end_epoch is None:
        end_epoch = num_epochs

    criterion = nn.CrossEntropyLoss(reduction='none')
    
    no_batches = 0
    cur_lr = optimizer.param_groups[-1]["lr"]

    # Not done in original paper - however it is generally beneficial
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

    if start_epoch == 0:
        # Not done in original paper - however it is SotA and generally beneficial
        ibp_init_shi(original_model, hardened_model)

    cur_eps, kappa = eps_scheduler.get_cur_eps(), eps_scheduler.get_cur_kappa()

    for epoch in range(start_epoch, end_epoch):

        epoch_rob_err = 0
        epoch_nat_err = 0

        if lr_decay_schedule_unit == "epoch":
            if epoch + 1 in lr_decay_schedule:
                print("LEARNING RATE DECAYED!")
                cur_lr = cur_lr * lr_decay_factor
                for g in optimizer.param_groups:
                    g["lr"] = cur_lr

        print(
            f"[{epoch + 1}/{num_epochs}]: eps {eps_scheduler.get_cur_eps(normalise=False):.4f}, kappa {kappa:.2f} "
        )
        hardened_model.train()
        original_model.train()
        running_loss = 0.0

        for batch_idx, (data, target) in enumerate(train_loader):

            cur_eps = eps_scheduler.get_cur_eps().reshape(-1, 1, 1)
            kappa = eps_scheduler.get_cur_kappa()
            beta = eps_scheduler.get_cur_beta()

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
            
            if loss_fusion:
                clean_output = loss_fusion_model(data, target)
                clean_loss = torch.mean(torch.log(clean_output))
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
                    model=hardened_model if not loss_fusion else loss_fusion_model,
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

            running_loss += loss.item()
            eps_scheduler.batch_step()
            no_batches += 1

        train_acc_nat = 1 - epoch_nat_err / len(train_loader)
        train_acc_cert = 1 - epoch_rob_err / len(train_loader)

        print(
            f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {running_loss/len(train_loader):.4f}"
        )
        print(f"\t Natural Acc. Train: {train_acc_nat:.4f}")
        print(f"\t Adv. Acc. Train: N/A")
        print(f"\t Certified Acc. Train: {train_acc_cert:.4f}")

        if results_path is not None and (epoch + 1) % checkpoint_save_interval == 0:
            save_checkpoint(
                hardened_model, optimizer, running_loss, epoch + 1, results_path
            )

    return hardened_model

