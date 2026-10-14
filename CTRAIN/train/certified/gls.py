"""Gaussian loss smoothing (Balauca et al., arXiv:2403.07095).

The released PGPE implementation uses preconditioned score gradients: d*(L+-L-)
for the mean and (d²-s²)/s*(mean(L+,L-)-baseline) for the deviation.
RGS averages ordinary gradients at independent Gaussian parameter samples.
"""
import copy
import math
import torch
from auto_LiRPA import PerturbationLpNorm
from CTRAIN.bound import bound_ibp, bound_crown, bound_crown_ibp
from CTRAIN.train.certified.losses.util import get_loss_from_bounds
from CTRAIN.train.certified.initialisation import ibp_init_shi
from CTRAIN.train.certified.regularisers import get_shi_regulariser, get_l1_reg
from CTRAIN.train.certified.eps_scheduler import BaseScheduler
from CTRAIN.train.certified.util import synchronise_bn
from CTRAIN.util import save_checkpoint


class GLSOptimizer(torch.optim.Optimizer):
    """Closure-based GLS adapter; noise and both optimizers are resumable.

    ``samples`` counts loss evaluations (PGPE requires an even count). Buffers
    are restored between samples. Gradients refer to the centre, never to a
    sampled network. PGPE uses the unperturbed loss as its baseline, following
    the authors' released implementation. Its scale changes by at most 2% per
    step; RGS uses a fixed scale with optional epoch-based decay in the trainer.
    """
    def __init__(self, model, optimizer_func=torch.optim.Adam, lr=5e-5,
                 estimator='rgs', samples=2, sigma=1e-3, sigma_lr=1e-3, gradient_clip=None):
        if estimator not in ('rgs', 'pgpe'):
            raise ValueError("estimator must be 'rgs' or 'pgpe'")
        if type(samples) is not int or samples < 1 or (estimator == 'pgpe' and samples % 2):
            raise ValueError('samples must be positive and even for PGPE')
        if not math.isfinite(sigma) or sigma <= 0 or not math.isfinite(sigma_lr) or sigma_lr < 0:
            raise ValueError('sigma must be positive finite and sigma_lr nonnegative finite')
        self.gradient_clip = gradient_clip
        self.model = model
        self.estimator, self.samples = estimator, samples
        params = [p for p in model.parameters() if p.requires_grad]
        super().__init__(params, dict(lr=lr))
        self.center_optimizer = optimizer_func(params, lr=lr)
        self.param_groups = self.center_optimizer.param_groups
        self.scales = [torch.nn.Parameter(torch.full_like(p, sigma)) for p in params]
        self.scale_optimizer = optimizer_func(self.scales, lr=sigma_lr)

    def state_dict(self):
        return dict(center=self.center_optimizer.state_dict(),
                    scale=self.scale_optimizer.state_dict(),
                    scales=[s.detach().clone() for s in self.scales],
                    estimator=self.estimator, samples=self.samples)

    def load_state_dict(self, state):
        if state['estimator'] != self.estimator or state['samples'] != self.samples:
            raise ValueError('GLS checkpoint estimator/sample configuration differs')
        self.center_optimizer.load_state_dict(state['center'])
        self.param_groups = self.center_optimizer.param_groups
        self.scale_optimizer.load_state_dict(state['scale'])
        with torch.no_grad():
            for scale, value in zip(self.scales, state['scales']):
                scale.copy_(value)

    def step(self, closure):
        params = [p for g in self.param_groups for p in g['params']]
        centers = [p.detach().clone() for p in params]
        buffers = [(b, b.clone()) for b in self.model.buffers()]
        old_grads = [None if p.grad is None else p.grad.clone() for p in params]
        grads = [torch.zeros_like(p) for p in params]
        sigma_grads = [torch.zeros_like(s) for s in self.scales]
        total = 0.

        def restore():
            with torch.no_grad():
                for p, center in zip(params, centers):
                    p.copy_(center)
                for buf, saved in buffers:
                    buf.copy_(saved)

        def evaluate(noise, sign, backward):
            restore()
            with torch.no_grad():
                for p, center, d in zip(params, centers, noise):
                    p.copy_(center + sign * d)
            self.zero_grad(set_to_none=True)
            with torch.enable_grad() if backward else torch.no_grad():
                loss = closure()
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite GLS loss')
                if backward:
                    loss.backward()
                    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in params):
                        raise FloatingPointError('Non-finite GLS gradients')
                    # The released RGS implementation clips each sample's
                    # gradient before averaging, not the averaged gradient.
                    if self.gradient_clip is not None:
                        torch.nn.utils.clip_grad_norm_(params, self.gradient_clip)
            return loss.detach()

        try:
            if self.estimator == 'pgpe':
                restore()
                with torch.no_grad():
                    baseline = closure().detach()
                    if not torch.isfinite(baseline):
                        raise FloatingPointError('Non-finite GLS baseline loss')
            count = self.samples if self.estimator == 'rgs' else self.samples // 2
            for _ in range(count):
                noise = [torch.randn_like(s) * s.detach() for s in self.scales]
                plus = evaluate(noise, 1, self.estimator == 'rgs')
                if self.estimator == 'rgs':
                    for grad, p in zip(grads, params):
                        if p.grad is not None:
                            grad.add_(p.grad / count)
                    total += plus / count
                else:
                    minus = evaluate(noise, -1, False)
                    for grad, sg, d, s in zip(grads, sigma_grads, noise, self.scales):
                        grad.add_(d * (plus - minus) / count)
                        sg.add_((d.square() - s.detach().square()) / s.detach()
                                * ((plus + minus) / 2 - baseline) / count)
                    total += (plus + minus) / (2 * count)
        except BaseException:
            for p, grad in zip(params, old_grads):
                p.grad = grad
            raise
        finally:
            restore()
        for p, grad in zip(params, grads):
            p.grad = grad
        if self.gradient_clip is not None and self.estimator == 'pgpe':
            torch.nn.utils.clip_grad_norm_(params, self.gradient_clip)
        self.center_optimizer.step()
        if self.estimator == 'pgpe':
            old_scales = [s.detach().clone() for s in self.scales]
            for s, grad in zip(self.scales, sigma_grads):
                s.grad = grad
            self.scale_optimizer.step()
            with torch.no_grad():
                for s, old in zip(self.scales, old_scales):
                    s.copy_(torch.maximum(old * .98, torch.minimum(old * 1.02, s)))
        return total


def gls_train_model(original_model, hardened_model, train_loader, optimizer,
                    lr_scheduler, num_epochs, eps, n_classes, start_epoch=0,
                    end_epoch=None, warm_up_epochs=1, ramp_up_epochs=80,
                    bound_method='CROWN', shi_init=True, shi_reg_weight=0.,
                    shi_reg_decay=True, l1_reg_weight=0., population_bn=False,
                    sigma_decay_milestones=(), sigma_decay_factor=.4,
                    results_path=None, checkpoint_save_interval=10, device='cuda',
                    val_loader=None, eval_eps=None):
    """Train the Gaussian-smoothed certified loss with a linear epsilon ramp.

    No clean/robust loss mixing is used after warm-up. Without validation, the
    final model is returned. Validation selects by sound centre-model bounds
    after ramp-up; optimizer checkpoints retain the training trajectory.
    """
    bounds = {'IBP': bound_ibp, 'CROWN': bound_crown, 'CROWN-IBP': bound_crown_ibp}
    if bound_method not in bounds:
        raise ValueError('bound_method must be IBP, CROWN, or CROWN-IBP')
    final_epoch = num_epochs if end_epoch is None else end_epoch
    if not (0 <= start_epoch <= final_epoch <= num_epochs):
        raise ValueError('invalid epoch range')
    if warm_up_epochs < 0 or ramp_up_epochs < 0 or warm_up_epochs + ramp_up_epochs > num_epochs:
        raise ValueError('invalid warm-up/ramp-up schedule')
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = BaseScheduler(num_epochs, eps, train_loader.mean,
                              train_loader.std if train_loader.normalised else torch.ones_like(train_loader.std),
                              eps_schedule=(warm_up_epochs, ramp_up_epochs),
                              eps_schedule_unit='epoch', batches_per_epoch=len(train_loader))
    if start_epoch == 0 and shi_init:
        ibp_init_shi(original_model, hardened_model)
    best_score, best_weights = -1., None
    for epoch in range(start_epoch, final_epoch):
        hardened_model.train()
        original_model.train()
        total = 0.
        for index, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            progress = epoch + index / len(train_loader) - warm_up_epochs
            fraction = min(1., max(0., progress / ramp_up_epochs)) if ramp_up_epochs else float(progress >= 0)
            scheduler.cur_eps = eps * fraction
            radius = scheduler.get_cur_eps().to(data).reshape(-1, 1, 1)
            ptb = PerturbationLpNorm(norm=float('inf'), eps=radius,
                    x_L=torch.maximum(data - radius, train_loader.min.to(data)),
                    x_U=torch.minimum(data + radius, train_loader.max.to(data)))

            def closure():
                output = hardened_model(data)
                if fraction:
                    lb, _ = bounds[bound_method](hardened_model, ptb, data, target, n_classes)
                    loss = get_loss_from_bounds(lb, criterion)
                else:
                    loss = criterion(output, target)
                if shi_reg_weight and fraction < 1:
                    bound_ibp(hardened_model, ptb, data, target, n_classes)
                    loss = loss + shi_reg_weight * get_shi_regulariser(
                        hardened_model, ptb, data, target, scheduler,
                        n_classes=n_classes, device=device, regularisation_decay=shi_reg_decay)
                if l1_reg_weight:
                    loss = loss + l1_reg_weight * get_l1_reg(original_model, device=device)
                return loss

            total += float(optimizer.step(closure))
            # Update BN once at the centre, rather than once per noisy sample.
            with torch.no_grad():
                hardened_model(data)
        synchronise_bn(original_model, hardened_model, train_loader, device, population_bn)
        lr_scheduler.step()
        if optimizer.estimator == 'rgs' and epoch + 1 in sigma_decay_milestones:
            with torch.no_grad():
                for scale in optimizer.scales:
                    scale.mul_(sigma_decay_factor)
        if val_loader is not None and fraction == 1.:
            from CTRAIN.eval import eval_certified
            eval_radius = torch.as_tensor(eps if eval_eps is None else eval_eps, device=device)
            if val_loader.normalised:
                eval_radius = eval_radius / val_loader.std.to(device)
            score, _ = eval_certified(hardened_model, val_loader, eval_radius.reshape(-1, 1, 1),
                                      n_classes=n_classes, method=bound_method, device=device)
            if score > best_score:
                best_score, best_weights = score, copy.deepcopy(hardened_model.state_dict())
        print(f'GLS-{optimizer.estimator.upper()} epoch {epoch + 1}: loss={total / len(train_loader):.6f}')
        if results_path is not None and (epoch + 1) % checkpoint_save_interval == 0:
            save_checkpoint(hardened_model, optimizer, total, epoch + 1, results_path,
                            lr_scheduler=lr_scheduler)
    if best_weights is not None:
        hardened_model.load_state_dict(best_weights)
        synchronise_bn(original_model, hardened_model, train_loader, device, False)
    return hardened_model
