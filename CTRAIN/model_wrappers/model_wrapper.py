from functools import partial
import os
import torch
import torch.nn as nn
import numpy as np
from auto_LiRPA.bound_general import BoundedModule

from smac import Scenario, HyperparameterOptimizationFacade
from smac.utils.configspace import get_config_hash

from CTRAIN.eval.eval import eval_acc, eval_model, eval_complete_abcrown
from CTRAIN.model_wrappers.configs import get_config_space


class CTRAINWrapper(nn.Module):
    """
    Wrapper base class for certifiably training models.
    """
    def __init__(self, model: nn.Module, eps:float, input_shape: tuple, train_eps_factor=1, lr=0.0005, optimizer_func=torch.optim.Adam, bound_opts=dict(conv_mode='patches', relu='adaptive'), device='cuda', checkpoint_save_path=None, checkpoint_save_interval=10):
        """
        Initialize the CTRAINWrapper Base Class.
        
        Args:
            model (nn.Module): The neural network model to be wrapped.
            eps (float): The epsilon value for training.
            input_shape (tuple): The shape of the input tensor.
            train_eps_factor (float, optional): Factor to scale epsilon during training. Default is 1.
            lr (float, optional): Learning rate for the optimizer. Default is 0.0005.
            optimizer_func (torch.optim.Optimizer, optional): The optimizer function to use. Default is torch.optim.Adam.
            bound_opts (dict, optional): Options for bounding the model. Default is {'conv_mode': 'patches', 'relu': 'adaptive'}.
            device (str or torch.device, optional): The device to run the model on. Default is 'cuda'.
            checkpoint_save_path (str, optional): Path to save checkpoints. Default is None.
            checkpoint_save_interval (int, optional): Interval to save checkpoints. Default is 10.
        
        Attributes:
            original_model (nn.Module): The original neural network model.
            eps (float): The epsilon value for training.
            train_eps (float): The scaled epsilon value for training.
            device (torch.device): The device to run the model on.
            n_classes (int): The number of classes in the model's output.
            bound_opts (dict): Options for bounding the model.
            bounded_model (BoundedModule): The bounded version of the original model.
            input_shape (tuple): The shape of the input tensor.
            optimizer_func (torch.optim.Optimizer): The optimizer function.
            optimizer (torch.optim.Optimizer): The optimizer instance.
            epoch (int): The current epoch number.
            checkpoint_path (str): Path to save checkpoints.
        """
        super(CTRAINWrapper, self).__init__()
        model = model.to(device)
        
        original_train = model.training
        self.original_model = model
        self.eps = eps
        self.train_eps = eps * train_eps_factor
        if isinstance(device, torch.device):
            self.device = device
        else:
            if device in ['cuda', 'cpu', 'mps']:
                self.device = torch.device(device)
            else:
                print("Unknown device - falling back to device CPU!")
                self.device = torch.device('cpu')
        
        if len(input_shape) < 4:
            input_shape = [1, *input_shape]
        model.eval()
        example_input = torch.ones(input_shape, device=device)
        self.n_classes = len(model(example_input)[0])
        self.bound_opts = bound_opts
        self.bounded_model = BoundedModule(model=self.original_model, global_input=example_input, bound_opts=bound_opts, device=device)
        self.input_shape = input_shape
        
        self.optimizer_func = optimizer_func
        self.optimizer = optimizer_func(self.bounded_model.parameters(), lr=lr)
        
        self.epoch = 0
        
        if original_train:
            self.original_model.train()
            self.bounded_model.train()
        
        self.checkpoint_path = checkpoint_save_path
        if checkpoint_save_path is not None:
            os.makedirs(self.checkpoint_path, exist_ok=True)
        
        self.checkpoint_save_interval = checkpoint_save_interval
    
    def train(self):
        """
        Sets wrapper into training mode.

        This method calls the `train` method on both the `original_model` and 
        the `bounded_model` to set them into training mode
        """
        self.original_model.train()
        self.bounded_model.train()
    
    def eval(self):
        """
        Sets the model to evaluation mode.

        This method sets both the original model and the bounded model to evaluation mode.
        In evaluation mode, certain layers like dropout and batch normalization behave differently
        compared to training mode, typically affecting the model's performance and predictions.
        """
        self.original_model.eval()
        self.bounded_model.eval()
    
    def forward(self, x):
        """
        Perform a forward pass through the LiRPA model.

        Args:
            x (torch.Tensor): Input tensor to be passed through the model.

        Returns:
            torch.Tensor: Output tensor after passing through the bounded model.
        """
        return self.bounded_model(x)
    
    def evaluate(self, test_loader, test_samples=np.inf, eval_method='ADAPTIVE'):
        """
        Evaluate the model using the provided test data loader.

        Args:
            test_loader (DataLoader): DataLoader containing the test dataset.
            test_samples (int, optional): Number of test samples to evaluate. Defaults to np.inf.
            eval_method (str or list, optional): The certification method to use. Options are 'IBP', 'CROWN', 'CROWN-IBP', 'ADAPTIVE', or a list of methods (which results in an ADAPTIVE evaluation using these methods). Default is 'ADAPTIVE'.

        Returns:
            (Tuple): Evaluation results in terms of std_acc, cert_acc and adv_acc.
        """
        eps_std = self.eps / test_loader.std if test_loader.normalised else torch.tensor(self.eps)
        eps_std = torch.reshape(eps_std, (*eps_std.shape, 1, 1))
        return eval_model(self.bounded_model, test_loader, n_classes=self.n_classes, eps=eps_std, test_samples=test_samples, method=eval_method, device=self.device)

    def evaluate_complete(self, test_loader, test_samples=np.inf, timeout=1000, no_cores=4, abcrown_batch_size=512, abcrown_config_dict=dict()):
        """
        Evaluate the model using the complete verification tool abCROWN.

        Args:
            test_loader (DataLoader): DataLoader for the test set.
            test_samples (int, optional): Number of test samples to evaluate. Defaults to np.inf.
            timeout (int, optional): Per-instance timeout for the verification process in seconds. Defaults to 1000.
            no_cores (int, optional): Number of CPU cores to use for verification. Only relevant, if MIP refinement is used in abCROWN. Defaults to 4.
            abcrown_batch_size (int, optional): Batch size for abCROWN. Defaults to 512. Decrease, if you run out of memory.
            abcrown_config_dict (dict, optional): Configuration dictionary for abCROWN according to the tools documentation. Defaults to an empty dictionary.

        Returns:
            (tuple): A tuple containing: std_acc (float): Standard accuracy of the model on the test set, certified_acc (float): Certified accuracy of the model on the test set and adv_acc (float): Adversarial accuracy of the model on the test set.
        """
        eps_std = self.eps / test_loader.std if test_loader.normalised else self.eps
        eps_std = torch.reshape(eps_std, (*eps_std.shape, 1, 1))
        std_acc = eval_acc(self.bounded_model, test_loader=test_loader, test_samples=test_samples)
        certified_acc, adv_acc = eval_complete_abcrown(
            model=self.bounded_model,
            eps_std=eps_std,
            data_loader=test_loader,
            n_classes=self.n_classes,
            input_shape=self.input_shape,
            test_samples=test_samples,
            timeout=timeout,
            no_cores=no_cores,
            abcrown_batch_size=abcrown_batch_size,
            abcrown_config_dict=abcrown_config_dict,
            device=self.device
        )
        return std_acc, certified_acc, adv_acc
    
    def state_dict(self):
        """
        Returns the state dictionary of the LiRPA model.

        The state dictionary contains the model parameters and persistent buffers.

        Returns:
            dict: A dictionary containing the model's state.
        """
        return self.bounded_model.state_dict()
    
    def load_state_dict(self, state_dict, strict = True):
        """
        Load the state dictionary into the bounded LiRPA model.

        Args:
            state_dict (dict): A dictionary containing model state parameters.
            strict (bool, optional): Whether to strictly enforce that the keys 
                                     in `state_dict` match the keys returned by 
                                     the model's `state_dict()` function. 
                                     Defaults to True.

        Returns:
            (NamedTuple): A named tuple with fields `missing_keys` and `unexpected_keys`.
                        `missing_keys` is a list of str containing the missing keys.
                        `unexpected_keys` is a list of str containing the unexpected keys.
        """
        return self.bounded_model.load_state_dict(state_dict, strict)
    
    def parameters(self, recurse=True):
        return self.bounded_model.parameters(recurse=recurse)
    # TODO: Add onnx export/loading

    def resume_from_checkpoint(self, checkpoint_path:str, train_loader, val_loader=None, end_epoch=None):
        """
        Resume training from a given checkpoint.

        Args:
            checkpoint_path (str): Path to the checkpoint file.
            train_loader (DataLoader): DataLoader for the training dataset.
            val_loader (DataLoader, optional): DataLoader for the validation dataset. Defaults to None.
            end_epoch (int, optional): Epoch to prematurely end training at. Defaults to None.

        Loads the model and optimizer state from the checkpoint, sets the starting epoch, 
        and resumes training from that epoch.
        """
        checkpoint = torch.load(checkpoint_path)
        model_state_dict = checkpoint['model_state_dict']
        self.load_state_dict(model_state_dict)
        self.epoch = checkpoint['epoch']
        optimizer_state_dict = checkpoint['optimizer_state_dict']
        self.optimizer.load_state_dict(optimizer_state_dict)

        self.train_model(train_loader, val_loader, start_epoch=self.epoch, end_epoch=end_epoch)
    
    def hpo(self, train_loader, val_loader, budget=5*24*60*60, defaults=dict(), eval_samples=1000, output_dir='./smac_hpo', deterministic=False, seed=42, nat_loss_weight=1., adv_loss_weight=1., cert_loss_weight=1.):
        """
        Perform hyperparameter optimization (HPO) using SMAC3 for the model. After the method returns, the model will have loaded the best hyperparameters found during the optimization and the according trained weights.
        
        Args:
            train_loader (DataLoader): DataLoader for the training dataset.
            val_loader (DataLoader): DataLoader for the validation dataset.
            budget (int, optional): Time budget for the HPO process in seconds. Default is 5 days (5*24*60*60).
            defaults (dict, optional): Default hyperparameter values. Default is an empty dictionary.
            eval_samples (int, optional): Number of samples to use for loss computation. Default is 1000.
            output_dir (str, optional): Directory to store HPO results. Default is './smac_hpo'.
            deterministic (bool, optional): Whether SMAC3 should treat the objective function as deterministic. Speeds up the optimisation. Default is False.
            seed (int, optional): Random seed for reproducibility of the HPO. Default is 42.
            nat_loss_weight (float, optional): Weight for the natural accuracy in the loss function.
            adv_loss_weight (float, optional): Weight for the adversarial accuracy in the loss function.
            cert_loss_weight (float, optional): Weight for the certified accuracy in the loss function.
            
        Returns:
            Configuration: The best hyperparameter configuration found during the optimization.
        """
        os.makedirs(output_dir, exist_ok=True)
        if os.listdir(output_dir):
            assert False, 'Output directory for HPO is not empty!'
        
        os.makedirs(f'{output_dir}/nets', exist_ok=True)
        os.makedirs(f'{output_dir}/smac/', exist_ok=True)

        eps_std = self.eps / train_loader.std
        scenario = Scenario(
            configspace=get_config_space(self, self.num_epochs, eps_std, defaults=defaults),
            deterministic=deterministic,
            walltime_limit=budget,
            n_trials=np.inf,
            output_directory=f'{output_dir}/smac/',
            use_default_config=True if len(defaults.values()) > 0 else False
        )
        initial_design = HyperparameterOptimizationFacade.get_initial_design(scenario, n_configs_per_hyperparamter=1)
        smac = HyperparameterOptimizationFacade(
            scenario,
            partial(self._hpo_runner, epochs=self.num_epochs, train_loader=train_loader, val_loader=val_loader, cert_eval_samples=eval_samples, output_dir=output_dir, nat_loss_weight=nat_loss_weight, adv_loss_weight=adv_loss_weight, cert_loss_weight=cert_loss_weight),
            initial_design=initial_design,
            overwrite=True,
            seed=seed,
        )

        inc = smac.optimize()
        
        config_hash = get_config_hash(inc, 32)
        self.load_state_dict(torch.load(f'{output_dir}/nets/{config_hash}.pt'))

        return inc

    def _hpo_runner(self, config, seed, epochs, train_loader, val_loader, output_dir, cert_eval_samples=1000):
        raise NotImplementedError('HPO can only be run on the concrete Wrappers!')


