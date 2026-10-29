"""CC-IBP wrapper using the common expressive IBP training and HPO infrastructure."""
import torch

from CTRAIN.model_wrappers.certified.expressive_ibp_model_wrapper import ExpressiveIBPModelWrapper
from CTRAIN.train.certified.losses import get_cc_ibp_loss


class CCIBPModelWrapper(ExpressiveIBPModelWrapper):
    """Wrap a classifier for CC-IBP training using cross-entropy applied to mixed adversarial
    and IBP margins.

    The shared expressive-loss implementation handles attacks, bound propagation
    and schedules, while this subclass selects the loss combination. Alpha
    weights the IBP component. Evaluation and hyperparameter optimisation use
    the common wrapper interface.

    See De Palma et al. (2024): https://arxiv.org/abs/2305.13991.
    """
    _loss_type = 'cc_ibp'
    _alpha_name = 'cc_ibp_alpha'
    _loss_func = staticmethod(get_cc_ibp_loss)

    def __init__(self, model, input_shape, eps, num_epochs, train_eps_factor=1,
                 optimizer_func=torch.optim.Adam, lr=0.0005, warm_up_epochs=1, ramp_up_epochs=70,
                 lr_scheduler_func=torch.optim.lr_scheduler.MultiStepLR,
                 lr_decay_kwargs=dict(milestones=(80, 90)), gradient_clip=10, l1_reg_weight=0.000001,
                 shi_reg_weight=.5, shi_reg_decay=True, pgd_steps=1, pgd_alpha=10, pgd_restarts=1,
                 pgd_early_stopping=False, pgd_alpha_decay_factor=.1, pgd_decay_milestones=(),
                 pgd_eps_factor=1, cc_ibp_alpha=.5, checkpoint_save_path=None, checkpoint_save_interval=10,
                 bound_opts=dict(conv_mode='patches', relu='adaptive'), device=torch.device('cuda'),
                 population_bn=False, relu_upper_retention=1.0):
        """Configure the CC-IBP margin mixture and shared expressive training options.

        Args:
            model (torch.nn.Module): PyTorch model to train or evaluate.
            input_shape (tuple[int, ...]): Shape of a model input, excluding the batch dimension unless supplied
                explicitly.
            eps (float or torch.Tensor): Target perturbation radius in raw input units, before loader normalisation.
            num_epochs (int): Total training budget, including warm-up and ramp-up.
            train_eps_factor (float): Factor applied to the training radius; the nominal evaluation
                radius remains eps.
            optimizer_func (callable): Optimiser constructor accepting the model parameters and a learning
                rate.
            lr (float): Learning rate passed to the optimiser.
            warm_up_epochs (int): Initial epochs using only the natural task loss.
            ramp_up_epochs (int): Epochs over which the perturbation radius increases to its training
                target.
            lr_scheduler_func (callable): Constructor for the learning-rate scheduler.
            lr_decay_kwargs (dict): Scheduler arguments, such as milestones and gamma for MultiStepLR.
            gradient_clip (float): Maximum parameter-gradient norm. None disables clipping where
                supported.
            l1_reg_weight (float): Weight of the L1 parameter regulariser.
            shi_reg_weight (float): Weight of the interval-stability regulariser proposed by Shi et al.
                (2021).
            shi_reg_decay (bool): Whether to reduce the Shi regulariser as the perturbation radius
                increases.
            pgd_steps (int): Number of projected gradient updates in the training attack.
            pgd_alpha (float): PGD step-size factor relative to the search-region radius.
            pgd_restarts (int): Number of random starts for the training attack.
            pgd_early_stopping (bool): Whether to stop searching an input once a counterexample has
                been found.
            pgd_alpha_decay_factor (float): Multiplier applied to the PGD step size at each decay
                milestone.
            pgd_decay_milestones (Sequence[int]): Attack iterations at which to reduce the PGD step size.
            pgd_eps_factor (float): Positive factor scaling the PGD search range, independently of the
                nominal bound region.
            cc_ibp_alpha (float): Weight of the IBP margins in the CC-IBP margin mixture.
            checkpoint_save_path (str or pathlib.Path, optional): Optional directory for epoch checkpoints containing the
                training trajectory.
            checkpoint_save_interval (int): Save an epoch checkpoint at this interval and at the end
                of the training call.
            bound_opts (dict): Bound-propagation options passed to auto_LiRPA.
            device (str or torch.device): Device on which to run the model and tensor computations.
            population_bn (bool): Whether to recalibrate BatchNorm on training data after each epoch,
                before model selection and saving.
            relu_upper_retention (float): Fraction of unstable ReLU upper bounds retained during
                training, in (0, 1]. Values below one give an unsound training
                surrogate.
        """
        super().__init__(
            model, input_shape, eps, num_epochs, train_eps_factor, optimizer_func, lr,
            warm_up_epochs, ramp_up_epochs, lr_scheduler_func, lr_decay_kwargs, gradient_clip,
            l1_reg_weight, shi_reg_weight, shi_reg_decay, pgd_steps, pgd_alpha, pgd_restarts,
            pgd_early_stopping, pgd_alpha_decay_factor, pgd_decay_milestones, pgd_eps_factor,
            cc_ibp_alpha, checkpoint_save_path, checkpoint_save_interval, bound_opts, device,
            population_bn, relu_upper_retention)
