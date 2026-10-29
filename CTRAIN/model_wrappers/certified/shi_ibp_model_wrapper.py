"""Model-wrapper interface for Shi-IBP training and robustness evaluation."""
import copy
import torch

from smac.utils.configspace import get_config_hash

from CTRAIN.model_wrappers.certified.model_wrapper import CTRAINWrapper
from CTRAIN.train.certified import shi_train_model
from CTRAIN.util import seed_ctrain

class ShiIBPModelWrapper(CTRAINWrapper):
    """Wrap a PyTorch classifier for Shi-IBP certified training.

    This wrapper configures IBP training with the initialisation and interval-stability regularisers proposed by Shi et al. (2021).
    Training is invoked through train_model; evaluation and hyperparameter
    optimisation use the common CTRAINWrapper interface. The training loop,
    loss computation and bound propagation are implemented in the corresponding
    library functions.

    See https://arxiv.org/abs/2102.06700.
    """

    def __init__(self, model, input_shape, eps, num_epochs, train_eps_factor=1,
                 optimizer_func=torch.optim.Adam, lr=0.0005, warm_up_epochs=1,
                 ramp_up_epochs=70, lr_scheduler_func=torch.optim.lr_scheduler.MultiStepLR,
                 lr_decay_kwargs=dict(milestones=(80, 90), gamma=0.2), gradient_clip=10,
                 l1_reg_weight=0.000001, shi_reg_weight=.5, shi_reg_decay=True,
                 checkpoint_save_path=None, checkpoint_save_interval=10,
                 bound_opts=dict(conv_mode='patches', relu='adaptive'),
                 device=torch.device('cuda'), population_bn=False,
                 start_kappa=0, end_kappa=0):
        """Configure the model, training schedule and Shi-IBP hyperparameters.

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
            lr_decay_kwargs (dict): Arguments for the learning-rate scheduler,
                e.g. milestones and gamma for MultiStepLR.
            gradient_clip (float): Gradient clipping value.
            l1_reg_weight (float): L1 regularisation weight.
            shi_reg_weight (float): SHI regularisation weight.
            shi_reg_decay (bool): Whether to decay SHI regularisation during the ramp up phase.
            checkpoint_save_path (str): Path to save checkpoints.
            checkpoint_save_interval (int): Interval for saving checkpoints.
            bound_opts (dict): Options for bounding according to the auto_LiRPA documentation.
            device (torch.device): Device to run the training on.
            population_bn (bool): Recalibrate BatchNorm on training data after every epoch.
            start_kappa (float): Initial natural-loss mixture weight.
            end_kappa (float): Final natural-loss mixture weight.
        """
        if not (0 <= start_kappa <= 1 and 0 <= end_kappa <= 1):
            raise ValueError('start_kappa and end_kappa must be finite values in [0, 1]')

        super().__init__(
            model, eps, input_shape, train_eps_factor, lr, optimizer_func,
            lr_scheduler_func, lr_decay_kwargs, bound_opts, device,
            checkpoint_save_path=checkpoint_save_path,
            checkpoint_save_interval=checkpoint_save_interval, population_bn=population_bn,
        )
        self.cert_train_method = 'shi'
        self.num_epochs = num_epochs
        self.lr = lr
        self.warm_up_epochs = warm_up_epochs
        self.ramp_up_epochs = ramp_up_epochs
        self.gradient_clip = gradient_clip
        self.l1_reg_weight = l1_reg_weight
        self.shi_reg_weight = shi_reg_weight
        self.shi_reg_decay = shi_reg_decay
        self.start_kappa = start_kappa
        self.end_kappa = end_kappa
        self.optimizer_func = optimizer_func

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
        trained_model = shi_train_model(
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
            eps_scheduler_args={'start_kappa': self.start_kappa, 'end_kappa': self.end_kappa},
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            n_classes=self.n_classes,
            gradient_clip=self.gradient_clip,
            l1_regularisation_weight=self.l1_reg_weight,
            shi_regularisation_weight=self.shi_reg_weight,
            shi_reg_decay=self.shi_reg_decay,
            results_path=self.checkpoint_path,
            checkpoint_save_interval=self.checkpoint_save_interval,
            population_bn=self.population_bn,
            device=self.device, selection_state=self.selection_state, history=self.history, validation_eps=self.eps)

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

        model_wrapper = ShiIBPModelWrapper(
            model=copy.deepcopy(self.original_model),
            input_shape=self.input_shape,
            eps=self.eps,
            num_epochs=epochs,
            bound_opts=self.bound_opts,
            population_bn=self.population_bn,
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
            start_kappa=config['shi:start_kappa'],
            end_kappa=config['shi:end_kappa'] * config['shi:start_kappa'],
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
