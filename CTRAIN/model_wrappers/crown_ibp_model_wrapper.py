import copy
import torch

from smac.utils.configspace import get_config_hash
from auto_LiRPA import BoundedModule, CrossEntropyWrapper

from CTRAIN.model_wrappers.model_wrapper import CTRAINWrapper
from CTRAIN.train.certified import crown_ibp_train_model
from CTRAIN.util.util import seed_ctrain

class CrownIBPModelWrapper(CTRAINWrapper):
    """
    Wrapper class for training models using CROWN-IBP method. For details, see Zhang et al. (2020) "Towards Stable and Efficient Training of Verifiably Robust Neural Networks". https://arxiv.org/pdf/1906.06316
    """
    
    def __init__(self, model, input_shape, eps, num_epochs, train_eps_factor=1, optimizer_func=torch.optim.Adam, lr=0.0005, warm_up_epochs=1, ramp_up_epochs=70,
                 lr_decay_factor=.2, lr_decay_milestones=(80, 90), gradient_clip=10, l1_reg_weight=0.000001,
                 shi_reg_weight=.5, shi_reg_decay=True, start_kappa=1, end_kappa=0, start_beta=1, end_beta=0,
                 loss_fusion=True, checkpoint_save_path=None, checkpoint_save_interval=10,
                 bound_opts=dict(conv_mode='patches', relu='adaptive'), device=torch.device('cuda')):
        """
        Initializes the CrownIBPModelWrapper.

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
            start_kappa (float): Starting value of kappa that trades-off IBP and clean loss.
            end_kappa (float): Ending value of kappa.
            start_beta (float): Starting value of beta that trades off IBP and CROWN-IBP loss.
            end_beta (float): Ending value of beta.
            loss_fusion (bool): Whether to use loss fusion in loss calculation (saves memory).
            checkpoint_save_path (str): Path to save checkpoints.
            checkpoint_save_interval (int): Interval for saving checkpoints.
            bound_opts (dict): Options for bounding according to the auto_LiRPA documentation.
            device (torch.device): Device to run the training on.
        """
        super().__init__(model, eps, input_shape, train_eps_factor, lr, optimizer_func, bound_opts, device, checkpoint_save_path=checkpoint_save_path, checkpoint_save_interval=checkpoint_save_interval)
        self.cert_train_method = 'crown_ibp'
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
        self.start_kappa = start_kappa
        self.end_kappa = end_kappa
        self.start_beta = start_beta
        self.end_beta = end_beta
        self.optimizer_func = optimizer_func
        self.loss_fusion = loss_fusion
        
        if loss_fusion:
            original_train = self.original_model.training
            self.original_model.eval()
            self.bounded_model.eval()
            example_input = torch.ones(self.input_shape, device=device)
            self.bound_opts['loss_fusion'] = True
            self.loss_fusion_model = BoundedModule(model=CrossEntropyWrapper(self.original_model), global_input=(example_input, torch.zeros(1, dtype=torch.long)), bound_opts=self.bound_opts, device=device)
            self.loss_fusion_optimizer = optimizer_func(self.loss_fusion_model.parameters(), lr=lr)

            if original_train:
                self.original_model.train()
                self.bounded_model.train()
                self.loss_fusion_model.train()
    
    def train(self):
        """
        Sets wrapper into training mode.

        This method calls the `train` method on both the `original_model` and 
        the `bounded_model` to set them into training mode
        """
        self.original_model.train()
        self.bounded_model.train()
        if self.loss_fusion:
            self.loss_fusion_model.train()
    
    def eval(self):
        """
        Sets the model to evaluation mode.

        This method sets both the original model and the bounded model to evaluation mode.
        In evaluation mode, certain layers like dropout and batch normalization behave differently
        compared to training mode, typically affecting the model's performance and predictions.
        """
        self.original_model.eval()
        self.bounded_model.eval()
        if self.loss_fusion:
            self.loss_fusion_model.eval()
            
    def train_model(self, train_loader, val_loader=None, start_epoch=0, end_epoch=None):
        """
        Trains the model using the CROWN-IBP method.

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
        trained_model = crown_ibp_train_model(
            original_model=self.original_model,
            hardened_model=self.bounded_model,
            loss_fusion_model=self.loss_fusion_model if self.loss_fusion else None,
            train_loader=train_loader,
            val_loader=val_loader,
            start_epoch=start_epoch,
            end_epoch=end_epoch,
            num_epochs=self.num_epochs,
            eps=self.train_eps,
            eps_std=eps_std,
            eps_schedule=(self.warm_up_epochs, self.ramp_up_epochs),
            eps_scheduler_args={'start_kappa': self.start_kappa, 'end_kappa': self.end_kappa, 'start_beta': self.start_beta, 'end_beta': self.end_beta},
            optimizer=self.optimizer if not self.loss_fusion else self.loss_fusion_optimizer,
            lr_decay_schedule=self.lr_decay_milestones,
            lr_decay_factor=self.lr_decay_factor,
            n_classes=self.n_classes,
            loss_fusion=self.loss_fusion,
            gradient_clip=self.gradient_clip,
            l1_regularisation_weight=self.l1_reg_weight,
            shi_regularisation_weight=self.shi_reg_weight,
            shi_reg_decay=self.shi_reg_decay,
            results_path=self.checkpoint_path,
            checkpoint_save_interval=self.checkpoint_save_interval,
            device=self.device
        )
        
        return trained_model
    
    def _hpo_runner(self, config, seed, epochs, train_loader, val_loader, output_dir, cert_eval_samples=1000, nat_loss_weight=1., adv_loss_weight=1., cert_loss_weight=1.):
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

        model_wrapper = CrownIBPModelWrapper(
            model=copy.deepcopy(self.original_model), 
            input_shape=self.input_shape,
            eps=self.eps,
            num_epochs=epochs, 
            bound_opts=self.bound_opts,
            checkpoint_save_path=None,
            device=self.device,
            loss_fusion=self.loss_fusion,
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
            start_kappa=config['crown_ibp:start_kappa'],
            end_kappa=config['crown_ibp:end_kappa'] * config['crown_ibp:start_kappa'],
            start_beta=config['crown_ibp:start_beta'],
            end_beta=config['crown_ibp:end_beta'] * config['crown_ibp:start_beta'],
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