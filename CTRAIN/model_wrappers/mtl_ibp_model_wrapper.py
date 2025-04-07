import copy
import torch
from smac.utils.configspace import get_config_hash

from CTRAIN.model_wrappers.model_wrapper import CTRAINWrapper
from CTRAIN.train.certified import mtl_ibp_train_model
from CTRAIN.util import seed_ctrain

class MTLIBPModelWrapper(CTRAINWrapper):
    """
    Wrapper class for training models using MTL-IBP method. For details, see De Palma et al. (2024) Expressive Losses for Verified Robustness via Convex Combinations. https://arxiv.org/pdf/2305.13991
    """
    
    def __init__(self, model, input_shape, eps, num_epochs, train_eps_factor=1, optimizer_func=torch.optim.Adam, lr=0.0005, warm_up_epochs=1, ramp_up_epochs=70,
                 lr_decay_factor=.2, lr_decay_milestones=(80, 90), gradient_clip=10, l1_reg_weight=0.000001,
                 shi_reg_weight=.5, shi_reg_decay=True, pgd_steps=1, 
                 pgd_alpha=10, pgd_restarts=1, pgd_early_stopping=False, pgd_alpha_decay_factor=.1,
                 pgd_decay_milestones=(), pgd_eps_factor=1, mtl_ibp_alpha=0.5, checkpoint_save_path=None, checkpoint_save_interval=10,
                 bound_opts=dict(conv_mode='patches', relu='adaptive'), device=torch.device('cuda')):
        """
        Initializes the MTLIBPModelWrapper.

        Args:
            model (torch.nn.Module): The model to be trained.
            input_shape (tuple): Shape of the input data.
            eps (float): Epsilon value describing the perturbation the network should be certifiably robust against.
            num_epochs (int): Number of epochs for training.
            train_eps_factor (float): Factor for training epsilon.
            optimizer_func (torch.optim.Optimizer): Optimizer function.
            lr (float): Learning rate.
            warm_up_epochs (int): Number of warm-up epochs, i.e. epochs where the model is trained on clean loss.
            ramp_up_epochs (int): Number of ramp-up epochs, i.e. epochs where the epsilon is gradually increased to the target train epsilon.
            lr_decay_factor (float): Learning rate decay factor.
            lr_decay_milestones (tuple): Milestones for learning rate decay.
            gradient_clip (float): Gradient clipping value.
            l1_reg_weight (float): L1 regularization weight.
            shi_reg_weight (float): Shi regularization weight.
            shi_reg_decay (bool): Whether to decay Shi regularization during the ramp up phase.
            pgd_steps (int): Number of PGD steps for adversrial loss computation.
            pgd_alpha (float): PGD step size for adversarial loss calculation.
            pgd_restarts (int): Number of PGD restarts for adversarial loss calculation.
            pgd_early_stopping (bool): Whether to use early stopping in PGD during adversarial loss calculation.
            pgd_alpha_decay_factor (float): PGD alpha decay factor.
            pgd_decay_milestones (tuple): Milestones for PGD alpha decay.
            pgd_eps_factor (float): Factor for PGD epsilon.
            mtl_ibp_alpha (float): Alpha value for MTL-IBP, i.e. the trade-off between certified and adversarial loss.
            checkpoint_save_path (str): Path to save checkpoints.
            checkpoint_save_interval (int): Interval for saving checkpoints.
            bound_opts (dict): Options for bounding according to the auto_LiRPA documentation.
            device (torch.device): Device to run the training on.
        """
        super().__init__(model, eps, input_shape, train_eps_factor, lr, optimizer_func, bound_opts, device, checkpoint_save_path=checkpoint_save_path, checkpoint_save_interval=checkpoint_save_interval)
        self.cert_train_method = 'mtl_ibp'
        self.num_epochs = num_epochs
        self.lr = lr
        self.warm_up_epochs = warm_up_epochs
        self.ramp_up_epochs = ramp_up_epochs
        self.lr_decay_factor = lr_decay_factor
        self.lr_decay_milestones = lr_decay_milestones
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
        self.pgd_decay_milestones = pgd_decay_milestones
        self.pgd_eps_factor = pgd_eps_factor
        self.mtl_ibp_alpha = mtl_ibp_alpha
        

    def train_model(self, train_loader, val_loader=None, start_epoch=0, end_epoch=None):
        """
        Trains the model using the MTL-IBP method.

        Args:
            train_loader (torch.utils.data.DataLoader): DataLoader for training data.
            val_loader (torch.utils.data.DataLoader, optional): DataLoader for validation data.
            start_epoch (int, optional): Epoch to start training from. Initialises learning rate and epsilon schedulers accordingly. Defaults to 0.
            end_epoch (int, optional): Epoch to prematurely end training at. Defaults to None.

        Returns:
            (auto_LiRPA.BoundedModule): Trained model.
        """
        eps_std = self.train_eps / train_loader.std if train_loader.normalised else torch.tensor(self.train_eps)
        eps_std = torch.reshape(eps_std, (*eps_std.shape, 1, 1))

        trained_model = mtl_ibp_train_model(
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
            optimizer=self.optimizer,
            lr_decay_schedule=self.lr_decay_milestones,
            lr_decay_factor=self.lr_decay_factor,
            n_classes=self.n_classes,
            gradient_clip=self.gradient_clip,
            l1_regularisation_weight=self.l1_reg_weight,
            shi_regularisation_weight=self.shi_reg_weight,
            shi_reg_decay=self.shi_reg_decay,
            alpha=self.mtl_ibp_alpha,
            pgd_n_steps=self.pgd_steps,
            pgd_step_size=self.pgd_alpha,
            pgd_restarts=self.pgd_restarts,
            pgd_eps_factor=self.pgd_eps_factor,
            pgd_early_stopping=self.pgd_early_stopping,
            pgd_decay_factor=self.pgd_alpha_decay_factor,
            pgd_decay_checkpoints=self.pgd_decay_milestones,
            results_path=self.checkpoint_path,
            checkpoint_save_interval=self.checkpoint_save_interval,
            device=self.device
        )
        
        return trained_model

    def _hpo_runner(self, config, seed, epochs, train_loader, val_loader, output_dir, cert_eval_samples=1000, nat_loss_weight=1, adv_loss_weight=1, cert_loss_weight=1):
        """
        Function called during hyperparameter optimization (HPO) using SMAC3, returns the loss.

        Args:
            config (dict): Configuration of hyperparameters.
            seed (int): Seed used.
            epochs (int): Number of epochs for training.
            train_loader (torch.utils.data.DataLoader): DataLoader for training data.
            val_loader (torch.utils.data.DataLoader): DataLoader for validation data.
            output_dir (str): Directory to save output.
            cert_eval_samples (int, optional): Number of samples for certification evaluation.
            nat_loss_weight (float, optional): Weight for the natural accuracy in the loss function.
            adv_loss_weight (float, optional): Weight for the adversarial accuracy in the loss function.
            cert_loss_weight (float, optional): Weight for the certified accuracy in the loss function.
            
        Returns:
            tuple: Loss and dictionary of accuracies that is saved as information to the run by SMAC3.
        """
        config_hash = get_config_hash(config, 32)
        seed_ctrain(seed)
        
        if config['optimizer_func'] == 'adam':
            optimizer_func = torch.optim.Adam
        elif config['optimizer_func'] == 'radam':
            optimizer_func = torch.optim.RAdam
        if config['optimizer_func'] == 'adamw':
            optimizer_func = torch.optim.AdamW
        
        lr_decay_milestones = [
            config['warm_up_epochs'] + config['ramp_up_epochs'] + config['lr_decay_epoch_1'],
            config['warm_up_epochs'] + config['ramp_up_epochs'] + config['lr_decay_epoch_1'] + config['lr_decay_epoch_2']
        ]

        model_wrapper = MTLIBPModelWrapper(
            model=copy.deepcopy(self.original_model), 
            input_shape=self.input_shape,
            eps=self.eps,
            num_epochs=epochs, 
            bound_opts=self.bound_opts,
            checkpoint_save_path=None,
            device=self.device,
            train_eps_factor=config['train_eps_factor'],
            optimizer_func=optimizer_func,
            lr=config['learning_rate'],
            warm_up_epochs=config['warm_up_epochs'],
            ramp_up_epochs=config['ramp_up_epochs'],
            gradient_clip=10,
            lr_decay_factor=config['lr_decay_factor'],
            lr_decay_milestones=[epoch for epoch in lr_decay_milestones if epoch <= epochs],
            l1_reg_weight=config['l1_reg_weight'],
            shi_reg_weight=config['shi_reg_weight'],
            shi_reg_decay=config['shi_reg_decay'],
            mtl_ibp_alpha=config['mtl_ibp:mtl_ibp_alpha'],
            pgd_alpha=config['mtl_ibp:pgd_alpha'],
            pgd_early_stopping=False,
            pgd_restarts=config['mtl_ibp:pgd_restarts'],
            pgd_steps=config['mtl_ibp:pgd_steps'],
            pgd_eps_factor=config['mtl_ibp:mtl_ibp_eps_factor'],
            pgd_decay_milestones=()
        )

        model_wrapper.train_model(train_loader=train_loader)
        torch.save(model_wrapper.state_dict(), f'{output_dir}/nets/{config_hash}.pt')
        model_wrapper.eval()
        std_acc, cert_acc, adv_acc = model_wrapper.evaluate(test_loader=val_loader, test_samples=cert_eval_samples)

        loss = 0
        loss -= nat_loss_weight * std_acc
        loss -= adv_loss_weight * adv_acc
        loss -= cert_loss_weight * cert_acc

        return loss, {'nat_acc': std_acc, 'adv_acc': adv_acc, 'cert_acc': cert_acc}
