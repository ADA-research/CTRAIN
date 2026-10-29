"""Model-wrapper interface for STAPS training and robustness evaluation."""
import math
import torch
import copy

from smac.utils.configspace import get_config_hash

from CTRAIN.model_wrappers.certified.model_wrapper import CTRAINWrapper
from CTRAIN.train.certified import staps_train_model
from CTRAIN.bound.taps import split_bound_graph
from CTRAIN.util import seed_ctrain
from CTRAIN.bound.util import validate_hybrid_options


class STAPSModelWrapper(CTRAINWrapper):
    """Wrap a PyTorch classifier for STAPS certified training.

    This wrapper configures STAPS training, using SABR subregions for TAPS feature propagation as proposed by Mao et al. (2023).
    Training is invoked through train_model; evaluation and hyperparameter
    optimisation use the common CTRAINWrapper interface. The training loop,
    loss computation and bound propagation are implemented in the corresponding
    library functions.

    See https://proceedings.neurips.cc/paper_files/paper/2023/file/e8b0c97b34fdaf58b2f48f8cca85e76a-Paper-Conference.pdf.
    """

    def __init__(self, model, input_shape, eps, num_epochs, train_eps_factor=1, optimizer_func=torch.optim.Adam, lr=0.0005, warm_up_epochs=1, ramp_up_epochs=70,
                 lr_scheduler_func=torch.optim.lr_scheduler.MultiStepLR, lr_decay_kwargs=dict(milestones=(80, 90), gamma=0.2), gradient_clip=10, l1_reg_weight=0.000001,
                 shi_reg_weight=.5, shi_reg_decay=True, pgd_steps=8,
                 pgd_alpha=0.5, pgd_restarts=1, pgd_early_stopping=False, pgd_alpha_decay_factor=.1,
                 pgd_decay_steps=(4,7), sabr_pgd_steps=8, sabr_pgd_alpha=0.5, sabr_pgd_restarts=1,
                 sabr_pgd_early_stopping=False, sabr_pgd_alpha_decay_factor=.1, sabr_pgd_decay_milestones=(4,7),
                 sabr_subselection_ratio=0.8, block_sizes=None, gradient_expansion_alpha=5,
                 gradient_link_thresh=.5, gradient_link_tol=0.00001,
                 checkpoint_save_path=None, checkpoint_save_interval=10,
                 bound_opts=dict(conv_mode='patches', relu='adaptive'), device=torch.device('cuda'),
                 relu_upper_retention=1.0, population_bn=False, pgd_eps_factor=1.0):
        """Configure the model, training schedule and STAPS hyperparameters.

        Args:
            model (torch.nn.Module): The model to be trained.
            input_shape (tuple): Shape of the input data.
            eps (float): Epsilon value describing the perturbation the network should be
                certifiably robust against.
            num_epochs (int): Number of epochs for training.
            train_eps_factor (float): Factor for training epsilon.
            optimizer_func (torch.optim.Optimizer): Optimizer function.
            lr (float): Learning rate.
            warm_up_epochs (int): Number of warm-up epochs, i.e. epochs where the model is
                trained on clean loss.
            ramp_up_epochs (int): Number of ramp-up epochs, i.e. epochs where the epsilon is
                gradually increased to the target train epsilon.
            lr_scheduler_func (callable): Constructor for the learning-rate scheduler.
            lr_decay_kwargs (dict): Scheduler arguments, such as milestones and gamma for MultiStepLR.
            gradient_clip (float): Gradient clipping value.
            l1_reg_weight (float): L1 regularisation weight.
            shi_reg_weight (float): SHI regularisation weight.
            shi_reg_decay (bool): Whether to decay SHI regularisation during the ramp up phase.
            pgd_steps (int): Number of PGD steps for TAPS loss calculation.
            pgd_alpha (float): PGD step size for TAPS loss calculation.
            pgd_restarts (int): Number of PGD restarts ffor TAPS loss calculation.
            pgd_early_stopping (bool): Whether to use early stopping in PGD for TAPS loss
                calculation.
            pgd_alpha_decay_factor (float): PGD alpha decay factor.
            pgd_decay_steps (tuple): Milestones for PGD alpha decay.
            sabr_pgd_steps (int): Number of PGD steps for SABR.
            sabr_pgd_alpha (float): PGD step size for SABR.
            sabr_pgd_restarts (int): Number of PGD restarts for SABR.
            sabr_pgd_early_stopping (bool): Whether to use early stopping in PGD for SABR.
            sabr_pgd_alpha_decay_factor (float): PGD alpha decay factor for SABR.
            sabr_pgd_decay_milestones (tuple): Milestones for PGD alpha decay for SABR.
            sabr_subselection_ratio (float): Subselection ratio (lambda) for SABR.
            block_sizes (list): Sizes of blocks for STAPS. This is used to split up the network
                into feature extractor and classifier. These must sum up to the
                number of layers in the network.
            gradient_expansion_alpha (float): Alpha value for gradient expansion, i.e. the
                factor the STAPS gradient is multiplied by.
            gradient_link_thresh (float): Threshold for gradient link.
            gradient_link_tol (float): Tolerance for gradient link.
            checkpoint_save_path (str): Path to save checkpoints.
            checkpoint_save_interval (int): Interval for saving checkpoints.
            bound_opts (dict): Options for bounding according to the auto_LiRPA documentation.
            device (torch.device): Device to run the training on.
            relu_upper_retention (float): Fraction of unstable ReLU upper bounds retained during
                training and selection; (0, 1], default 1.
            population_bn (bool): Recalibrate BatchNorm on training data after each epoch;
                default False.
            pgd_eps_factor (float): PGD search range factor; nominal IBP bounds are unchanged.
        """
        validate_hybrid_options(relu_upper_retention, pgd_eps_factor)
        super().__init__(model, eps, input_shape, train_eps_factor, lr, optimizer_func, lr_scheduler_func, lr_decay_kwargs, bound_opts, device, checkpoint_save_path=checkpoint_save_path, checkpoint_save_interval=checkpoint_save_interval, population_bn=population_bn)
        self.pgd_eps_factor = pgd_eps_factor
        self.relu_upper_retention = relu_upper_retention
        self.cert_train_method = 'staps'
        self.num_epochs = num_epochs
        self.lr = lr
        self.warm_up_epochs = warm_up_epochs
        self.ramp_up_epochs = ramp_up_epochs
        self.gradient_clip = gradient_clip
        self.l1_reg_weight = l1_reg_weight
        self.shi_reg_weight = shi_reg_weight
        self.shi_reg_decay = shi_reg_decay
        self.optimizer_func = optimizer_func
        self.pgd_steps = pgd_steps
        self.pgd_alpha = pgd_alpha
        self.pgd_restarts = pgd_restarts
        self.pgd_early_stopping = pgd_early_stopping
        self.pgd_alpha_decay_factor = pgd_alpha_decay_factor
        self.pgd_decay_steps = pgd_decay_steps
        self.block_sizes = block_sizes
        self.gradient_expansion_alpha = gradient_expansion_alpha
        self.gradient_link_thresh = gradient_link_thresh
        self.gradient_link_tol = gradient_link_tol
        self.sabr_pgd_steps = sabr_pgd_steps
        self.sabr_pgd_alpha = sabr_pgd_alpha
        self.sabr_pgd_restarts = sabr_pgd_restarts
        self.sabr_pgd_early_stopping = sabr_pgd_early_stopping
        self.sabr_pgd_alpha_decay_factor = sabr_pgd_alpha_decay_factor
        self.sabr_pgd_decay_milestones = sabr_pgd_decay_milestones
        self.sabr_subselection_ratio = sabr_subselection_ratio

        ori_train = self.original_model.training
        self.original_model.eval()
        self.bounded_model.eval()

        assert block_sizes is not None, "For TAPS training we require a network split!"
        self.bounded_model.bounded_blocks = split_bound_graph(
            self.bounded_model, block_sizes, self.original_model.layers)

        if ori_train:
            self.original_model.train()
            self.bounded_model.train()



    def train_model(self, train_loader, val_loader=None, start_epoch=0, end_epoch=None):
        """Train the wrapped model using the configured method and schedule.

        The wrapper forwards its configuration to the corresponding library trainer.
        start_epoch and end_epoch select a segment of the full schedule, retaining
        its original warm-up, ramp-up and learning-rate milestones.

        Args:
            train_loader (torch.utils.data.DataLoader): DataLoader for training data.
            val_loader (torch.utils.data.DataLoader, optional): DataLoader for validation data.
            start_epoch (int, optional): Epoch to start training from. Initialises learning rate
                and epsilon schedulers accordingly.
            end_epoch (int, optional): Epoch to prematurely end training at.

        Returns:
            (auto_LiRPA.BoundedModule): Trained model.
        """
        eps_std = self.train_eps / train_loader.std if train_loader.normalised else torch.tensor(self.train_eps)

        eps_std = torch.reshape(eps_std, (*eps_std.shape, 1, 1))
        trained_model = staps_train_model(
            original_model=self.original_model,
            hardened_model=self.bounded_model,
            train_loader=train_loader,
            val_loader=val_loader,
            start_epoch=start_epoch,
            end_epoch=end_epoch,
            num_epochs=self.num_epochs,
            eps=self.train_eps,
            eps_std=eps_std,
            eps_schedule=(self.warm_up_epochs, self.ramp_up_epochs),
            eps_scheduler_args={},
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            n_classes=self.n_classes,
            gradient_clip=self.gradient_clip,
            l1_regularisation_weight=self.l1_reg_weight,
            shi_regularisation_weight=self.shi_reg_weight,
            shi_reg_decay=self.shi_reg_decay,
            gradient_expansion_alpha=self.gradient_expansion_alpha,
            taps_gradient_link_thresh=self.gradient_link_thresh,
            taps_gradient_link_tolerance=self.gradient_link_tol,
            taps_pgd_restarts=self.pgd_restarts,
            taps_pgd_step_size=self.pgd_alpha,
            taps_pgd_steps=self.pgd_steps,
            taps_pgd_decay_checkpoints=self.pgd_decay_steps,
            taps_pgd_decay_factor=self.pgd_alpha_decay_factor,
            sabr_pgd_steps=self.sabr_pgd_steps,
            sabr_pgd_restarts=self.sabr_pgd_restarts,
            sabr_pgd_step_size=self.sabr_pgd_alpha,
            sabr_pgd_early_stopping=self.sabr_pgd_early_stopping,
            sabr_pgd_decay_checkpoints=self.sabr_pgd_decay_milestones,
            sabr_pgd_decay_factor=self.sabr_pgd_alpha_decay_factor,
            subselection_ratio=self.sabr_subselection_ratio,
            results_path=self.checkpoint_path,
            checkpoint_save_interval=self.checkpoint_save_interval,
            population_bn=self.population_bn,
            relu_upper_retention=self.relu_upper_retention,
            validation_eps=self.eps,
            pgd_eps_factor=self.pgd_eps_factor,
            device=self.device, selection_state=self.selection_state, history=self.history)

        self.epoch = self.num_epochs if end_epoch is None else end_epoch
        return trained_model

    def _hpo_runner(self, config, seed, epochs, train_loader, val_loader, output_dir, cert_eval_samples=1000, nat_loss_weight=1, adv_loss_weight=1, cert_loss_weight=1, complete_verify=False):
        """Train and evaluate one hyperparameter configuration.

        Construct a method-specific wrapper, train it and save its inference weights.
        Return the negative weighted accuracy and the natural, certified and
        adversarial validation accuracies for the optimisation procedure.

        Args:
            config (dict): Configuration of hyperparameters.
            seed (int): Seed used.
            epochs (int): Number of epochs for training.
            train_loader (torch.utils.data.DataLoader): DataLoader for training data.
            val_loader (torch.utils.data.DataLoader): DataLoader for validation data.
            output_dir (str): Directory to save output.
            cert_eval_samples (int, optional): Number of samples for certification evaluation.
            nat_loss_weight (float, optional): Weight for the natural accuracy in the loss
                function.
            adv_loss_weight (float, optional): Weight for the adversarial accuracy in the loss
                function.
            cert_loss_weight (float, optional): Weight for the certified accuracy in the loss
                function.
            complete_verify (bool): Whether to evaluate the trial through complete verification.

        Returns:
            (tuple): Loss and dictionary of accuracies that is saved as information to the run by SMAC3.
        """
        config_hash = get_config_hash(config, 32)
        seed_ctrain(seed)

        optimizer_func = self._optimizer_from_config(config)

        no_layers = len(self.original_model.layers)
        feature_extractor_size = min(no_layers - 1, max(1,
            math.ceil(config['taps:block_split_point'] * no_layers)))
        classifier_size = no_layers - feature_extractor_size
        block_sizes = (feature_extractor_size, classifier_size)

        model_wrapper = STAPSModelWrapper(
            model=copy.deepcopy(self.original_model),
            input_shape=self.input_shape,
            eps=self.eps,
            num_epochs=epochs,
            bound_opts=self.bound_opts,
            relu_upper_retention=self.relu_upper_retention,
            population_bn=self.population_bn,
            pgd_eps_factor=config['sabr:pgd_eps_factor'],
            checkpoint_save_path=None,
            device=self.device,
            train_eps_factor=config['train_eps_factor'],
            optimizer_func=optimizer_func,
            lr=config['learning_rate'],
            lr_scheduler_func=torch.optim.lr_scheduler.MultiStepLR,
            warm_up_epochs=config['warm_up_epochs'],
            ramp_up_epochs=config['ramp_up_epochs'],
            gradient_clip=self.gradient_clip,
            lr_decay_kwargs=self._lr_decay_kwargs_from_config(config, epochs),
            l1_reg_weight=config['l1_reg_weight'],
            shi_reg_weight=config['shi_reg_weight'],
            shi_reg_decay=config['shi_reg_decay'],
            sabr_subselection_ratio=config['sabr:subselection_ratio'],
            sabr_pgd_alpha=config['sabr:pgd_alpha'],
            sabr_pgd_early_stopping=False,
            sabr_pgd_restarts=config['sabr:pgd_restarts'],
            sabr_pgd_steps=config['sabr:pgd_steps'],
            sabr_pgd_decay_milestones=(),
            pgd_alpha=config['taps:pgd_alpha'],
            pgd_restarts=config['taps:pgd_restarts'],
            pgd_steps=config['taps:pgd_steps'],
            gradient_expansion_alpha=config['taps:gradient_expansion_alpha'],
            pgd_early_stopping=False,
            pgd_decay_steps=(),
            block_sizes=block_sizes
        )

        model_wrapper.train_model(train_loader=train_loader)
        torch.save(model_wrapper.state_dict(), f'{output_dir}/nets/{config_hash}.pt')
        model_wrapper.eval()
        std_acc, cert_acc, adv_acc = self._evaluate_hpo_model(
            model_wrapper, val_loader, cert_eval_samples, output_dir, config_hash, complete_verify
        )

        loss = 0
        loss -= nat_loss_weight * std_acc
        loss -= adv_loss_weight * adv_acc
        loss -= cert_loss_weight * cert_acc

        return loss, {'nat_acc': std_acc, 'adv_acc': adv_acc, 'cert_acc': cert_acc}
