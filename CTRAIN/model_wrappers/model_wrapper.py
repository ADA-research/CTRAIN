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


def _constant_value(hyperparameter):
    if hasattr(hyperparameter, "value"):
        return hyperparameter.value
    return hyperparameter.default_value


class CTRAINWrapper(nn.Module):
    """
    Wrapper base class for certifiably training models.
    """
    def __init__(self, model: nn.Module, eps:float, input_shape: tuple, train_eps_factor=1, lr=0.0005, optimizer_func=torch.optim.Adam,
                 lr_scheduler_func=torch.optim.lr_scheduler.MultiStepLR, lr_decay_kwargs=dict(milestones=(80, 90), gamma=0.2), bound_opts=dict(conv_mode='patches', relu='adaptive'), device='cuda', checkpoint_save_path=None, checkpoint_save_interval=10):
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

        self.lr_scheduler_func = lr_scheduler_func
        self.lr_scheduler = self.lr_scheduler_func(self.optimizer, **lr_decay_kwargs)

        self.epoch = 0

        if original_train:
            self.original_model.train()
            self.bounded_model.train()

        self.checkpoint_path = checkpoint_save_path
        if checkpoint_save_path is not None:
            os.makedirs(self.checkpoint_path, exist_ok=True)

        self.checkpoint_save_interval = checkpoint_save_interval

    def train(self, mode=True):
        """
        Sets wrapper into training mode.

        This method calls the `train` method on both the `original_model` and
        the `bounded_model` to set them into training mode
        """
        self.original_model.train(mode=mode)
        self.bounded_model.train(mode=mode)

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

    def evaluate_complete(
        self,
        test_loader,
        test_samples=np.inf,
        timeout=1000,
        no_cores=4,
        abcrown_batch_size=512,
        abcrown_config_dict=None,
        results_path='./abcrown_results',
        warm_start=False,
        start_idx=0,
        end_idx=None,
        results_filename="results.json",
    ):
        """
        Evaluate the model using the complete verification tool abCROWN.

        Args:
            test_loader (DataLoader): DataLoader for the test set.
            test_samples (int, optional): Number of test samples to evaluate. Defaults to np.inf.
            timeout (int, optional): Per-instance timeout for the verification process in seconds. Defaults to 1000.
            no_cores (int, optional): Number of CPU cores to use for verification. Only relevant, if MIP refinement is used in abCROWN. Defaults to 4.
            abcrown_batch_size (int, optional): Batch size for abCROWN. Defaults to 512. Decrease, if you run out of memory.
            abcrown_config_dict (dict, optional): Configuration dictionary for abCROWN according to the tools documentation. Defaults to an empty dictionary.
            results_path (str, optional): Path to save abCROWN results and logs. Defaults to './abcrown_results'.
            warm_start (bool, optional): Reuse existing results from the results file. Defaults to False.
            start_idx (int, optional): First dataset index to verify. Defaults to 0.
            end_idx (int, optional): Exclusive end dataset index to verify. Defaults to None.
            results_filename (str, optional): JSON file name under results_path. Defaults to 'results.json'.

        Returns:
            (tuple): A tuple containing: std_acc (float): Standard accuracy of the model on the test set, certified_acc (float): Certified accuracy of the model on the test set and adv_acc (float): Adversarial accuracy of the model on the test set.
        """
        eps_std = self.eps / test_loader.std if test_loader.normalised else torch.tensor(self.eps)
        eps_std = torch.reshape(eps_std, (*eps_std.shape, 1, 1)).to(self.device)
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
            device=self.device,
            results_path=results_path,
            warm_start=warm_start,
            start_idx=start_idx,
            end_idx=end_idx,
            results_filename=results_filename,
        )
        return std_acc, certified_acc, adv_acc

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """
        Returns the state dictionary of the LiRPA model.

        The state dictionary contains the model parameters and persistent buffers.

        Returns:
            dict: A dictionary containing the model's state.
        """
        return self.bounded_model.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)

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

    def hpo_smac(self, train_loader, val_loader, budget=5*24*60*60, defaults=dict(), eval_samples=1000, output_dir='./smac_hpo', deterministic=False, seed=42, nat_loss_weight=1., adv_loss_weight=1., cert_loss_weight=1.):
        """
        Perform single-objective hyperparameter optimization using SMAC3.

        After the method returns, the model will have loaded the best
        hyperparameters found during the optimization and the corresponding
        trained weights. New code should prefer :meth:`hpo` for multi-objective
        Optuna HPO.

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

    def hpo(
        self,
        train_loader,
        val_loader,
        budget_time=5 * 24 * 60 * 60,
        budget_trials=np.inf,
        defaults=None,
        eval_samples=np.inf,
        output_dir="./optuna_hpo",
        min_nat_acc=0.0,
        min_cert_acc=0.0,
        seed=0,
        sampler="botorch",
        complete_verify=False,
        study_name="moctrain",
    ):
        """
        Perform multi-objective HPO with Optuna and return the Pareto front.

        The objectives are natural and certified validation accuracy. The full
        Pareto front is stored in the Optuna study database under ``output_dir``.
        Trial checkpoints are written to ``output_dir/nets``. This method does
        not load one checkpoint implicitly, because selecting a final model is a
        downstream decision on the returned Pareto front.

        Args:
            train_loader (DataLoader): DataLoader for the training dataset.
            val_loader (DataLoader): DataLoader for validation.
            budget_time (int): Wall-clock budget in seconds.
            budget_trials (int or float): Maximum number of trials. ``np.inf`` disables this limit.
            defaults (dict): Default hyperparameter values for the ConfigSpace.
            eval_samples (int): Number of validation samples used per trial.
            output_dir (str): Directory for Optuna DBs and trial checkpoints.
            min_nat_acc (float): Feasibility threshold for natural accuracy.
            min_cert_acc (float): Feasibility threshold for certified accuracy.
            seed (int): Random seed.
            sampler (str or optuna.samplers.BaseSampler): ``"botorch"``, ``"nsgaii"``, or a sampler instance.
            complete_verify (bool): Use complete verification for the certified objective.
            study_name (str): Name of the persisted Optuna study.

        Returns:
            list[dict]: Pareto-optimal trials with their configs, metrics,
                feasibility status, and checkpoint paths.
        """
        defaults = defaults or {}
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/nets", exist_ok=True)

        try:
            import optuna
        except ImportError as exc:
            raise ImportError(
                "Optuna is required for hpo. Install CTRAIN with the HPO "
                "dependencies or install optuna and optuna-integration."
            ) from exc

        eps_std = self.eps / train_loader.std if train_loader.normalised else torch.tensor(self.eps)
        config_space = get_config_space(self, self.num_epochs, eps_std, defaults=defaults)

        def constraints(trial):
            return trial.user_attrs.get("constraints", (0.0, 0.0))

        def objective(trial):
            config = self._sample_optuna_config(trial, config_space)
            trial.set_user_attr("config_hash", get_config_hash(config, 32))
            _, metrics = self._hpo_runner(
                config=config,
                seed=seed,
                epochs=self.num_epochs,
                train_loader=train_loader,
                val_loader=val_loader,
                output_dir=output_dir,
                cert_eval_samples=eval_samples,
                complete_verify=complete_verify,
            )
            trial.set_user_attr(
                "constraints",
                (
                    min_nat_acc - metrics["nat_acc"],
                    min_cert_acc - metrics["cert_acc"],
                ),
            )
            trial.set_user_attr("adv_acc", metrics.get("adv_acc"))
            trial.set_user_attr("metrics", metrics)
            return metrics["nat_acc"], metrics["cert_acc"]

        optuna_sampler = self._get_optuna_sampler(optuna, sampler, constraints, seed)
        study = optuna.create_study(
            directions=["maximize", "maximize"],
            study_name=study_name,
            storage=f"sqlite:///{output_dir}/optuna_study.db",
            load_if_exists=True,
            sampler=optuna_sampler,
        )

        n_trials = None if budget_trials == np.inf else max(int(budget_trials) - len(study.trials), 0)
        if n_trials is None or n_trials > 0:
            study.optimize(
                objective,
                n_trials=n_trials,
                timeout=budget_time,
                show_progress_bar=True,
            )

        pareto_trials = self._constrained_pareto_trials(study.trials)
        if not pareto_trials:
            raise RuntimeError("Optuna did not produce any Pareto-optimal trials.")

        return [
            self._optuna_trial_result(trial, config_space, output_dir)
            for trial in pareto_trials
        ]

    def hpo_single_objective(
        self,
        train_loader,
        val_loader,
        budget_time=5 * 24 * 60 * 60,
        budget_trials=np.inf,
        defaults=None,
        eval_samples=np.inf,
        output_dir="./optuna_hpo_single_objective",
        nat_acc_weight=1.0,
        adv_acc_weight=0.0,
        cert_acc_weight=1.0,
        seed=0,
        sampler="botorch",
        complete_verify=False,
        study_name="ctrain_single_objective",
        load_best=True,
    ):
        """
        Perform scalar Optuna HPO and optionally load the best checkpoint.

        By default, the optimized objective is ``nat_acc + cert_acc``. The
        objective can be changed with ``nat_acc_weight``, ``adv_acc_weight``,
        and ``cert_acc_weight``.

        Returns:
            dict: Best trial with its scalar objective value, config, metrics,
                and checkpoint path.
        """
        defaults = defaults or {}
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/nets", exist_ok=True)

        try:
            import optuna
        except ImportError as exc:
            raise ImportError(
                "Optuna is required for hpo_single_objective. Install CTRAIN "
                "with the HPO dependencies or install optuna."
            ) from exc

        eps_std = self.eps / train_loader.std if train_loader.normalised else torch.tensor(self.eps)
        config_space = get_config_space(self, self.num_epochs, eps_std, defaults=defaults)

        def objective(trial):
            config = self._sample_optuna_config(trial, config_space)
            trial.set_user_attr("config_hash", get_config_hash(config, 32))
            _, metrics = self._hpo_runner(
                config=config,
                seed=seed,
                epochs=self.num_epochs,
                train_loader=train_loader,
                val_loader=val_loader,
                output_dir=output_dir,
                cert_eval_samples=eval_samples,
                complete_verify=complete_verify,
            )
            trial.set_user_attr("metrics", metrics)
            trial.set_user_attr("adv_acc", metrics.get("adv_acc"))
            return (
                nat_acc_weight * metrics["nat_acc"]
                + adv_acc_weight * (metrics.get("adv_acc") or 0.0)
                + cert_acc_weight * metrics["cert_acc"]
            )

        optuna_sampler = self._get_optuna_single_objective_sampler(optuna, sampler, seed)
        study = optuna.create_study(
            direction="maximize",
            study_name=study_name,
            storage=f"sqlite:///{output_dir}/optuna_study.db",
            load_if_exists=True,
            sampler=optuna_sampler,
        )

        n_trials = None if budget_trials == np.inf else max(int(budget_trials) - len(study.trials), 0)
        if n_trials is None or n_trials > 0:
            study.optimize(
                objective,
                n_trials=n_trials,
                timeout=budget_time,
                show_progress_bar=True,
            )

        try:
            best_trial = study.best_trial
        except ValueError as exc:
            raise RuntimeError("Optuna did not produce a completed trial.") from exc
        if best_trial is None:
            raise RuntimeError("Optuna did not produce a completed trial.")

        best = self._optuna_trial_result(best_trial, config_space, output_dir)
        if load_best:
            self.load_state_dict(torch.load(best["checkpoint_path"], map_location=self.device))
        return best

    def _sample_optuna_config(self, trial, config_space):
        config = {}
        for hp_name in config_space:
            hp = config_space[hp_name]
            if hasattr(hp, "choices"):
                config[hp_name] = trial.suggest_categorical(hp_name, list(hp.choices))
            elif hp.__class__.__name__.endswith("IntegerHyperparameter"):
                config[hp_name] = trial.suggest_int(hp_name, hp.lower, hp.upper, log=getattr(hp, "log", False))
            elif hp.__class__.__name__.endswith("FloatHyperparameter"):
                config[hp_name] = trial.suggest_float(hp_name, hp.lower, hp.upper, log=getattr(hp, "log", False))
            elif hp.__class__.__name__ == "Constant":
                config[hp_name] = _constant_value(hp)
            else:
                raise ValueError(f"Unsupported hyperparameter type for {hp_name}: {type(hp)}")
        return config

    def _config_from_optuna_trial(self, trial, config_space):
        config = {}
        for hp_name in config_space:
            hp = config_space[hp_name]
            if hp.__class__.__name__ == "Constant":
                config[hp_name] = _constant_value(hp)
            else:
                config[hp_name] = trial.params[hp_name]
        return config

    def _get_optuna_sampler(self, optuna, sampler, constraints_func, seed):
        if not isinstance(sampler, str):
            return sampler
        if sampler == "nsgaii":
            return optuna.samplers.NSGAIISampler(constraints_func=constraints_func, seed=seed)
        if sampler != "botorch":
            raise ValueError("sampler must be 'botorch', 'nsgaii', or an Optuna sampler instance")
        try:
            from optuna.integration import BoTorchSampler
        except (ImportError, ModuleNotFoundError) as exc:
            raise ImportError(
                "The BoTorch sampler requires optuna-integration with BoTorch support. "
                "Install optuna-integration[botorch] or use sampler='nsgaii'."
            ) from exc
        return BoTorchSampler(constraints_func=constraints_func, seed=seed, device=str(self.device))

    def _get_optuna_single_objective_sampler(self, optuna, sampler, seed):
        if not isinstance(sampler, str):
            return sampler
        if sampler == "tpe":
            return optuna.samplers.TPESampler(seed=seed)
        if sampler == "random":
            return optuna.samplers.RandomSampler(seed=seed)
        if sampler == "nsgaii":
            return optuna.samplers.NSGAIISampler(seed=seed)
        raise ValueError("sampler must be 'tpe', 'random', 'nsgaii', or an Optuna sampler instance")

    def _constrained_pareto_trials(self, trials):
        completed_trials = [
            trial for trial in trials
            if trial.values is not None and len(trial.values) == 2
        ]
        feasible_trials = [
            trial for trial in completed_trials
            if all(constraint <= 0 for constraint in trial.user_attrs.get("constraints", (0.0, 0.0)))
        ]
        candidates = feasible_trials if feasible_trials else completed_trials
        return [
            trial for trial in candidates
            if not any(self._dominates(other.values, trial.values) for other in candidates if other.number != trial.number)
        ]

    def _dominates(self, values, other_values):
        return all(value >= other for value, other in zip(values, other_values)) and any(
            value > other for value, other in zip(values, other_values)
        )

    def _optuna_trial_result(self, trial, config_space, output_dir):
        config = self._config_from_optuna_trial(trial, config_space)
        config_hash = trial.user_attrs.get("config_hash", get_config_hash(config, 32))
        constraints = trial.user_attrs.get("constraints")
        metrics = trial.user_attrs.get("metrics", {})
        if trial.values and len(trial.values) >= 2:
            metrics = {
                "nat_acc": trial.values[0],
                "cert_acc": trial.values[1],
                "adv_acc": trial.user_attrs.get("adv_acc"),
                **metrics,
            }
        return {
            "trial_number": trial.number,
            "values": tuple(trial.values) if trial.values is not None else None,
            "objective_value": trial.values[0] if trial.values is not None and len(trial.values) == 1 else None,
            "metrics": metrics,
            "config": config,
            "config_hash": config_hash,
            "checkpoint_path": f"{output_dir}/nets/{config_hash}.pt",
            "constraints": constraints,
            "feasible": None if constraints is None else all(constraint <= 0 for constraint in constraints),
        }

    def _optimizer_from_config(self, config):
        optimizer_name = config["optimizer_func"]
        if optimizer_name == "adam":
            return torch.optim.Adam
        if optimizer_name == "adamw":
            return torch.optim.AdamW
        if optimizer_name == "radam":
            return torch.optim.RAdam
        raise ValueError(f"Unknown optimizer_func: {optimizer_name}")

    def _lr_decay_kwargs_from_config(self, config, epochs):
        lr_decay_milestones = [
            config["warm_up_epochs"] + config["ramp_up_epochs"] + config["lr_decay_epoch_1"],
            config["warm_up_epochs"] + config["ramp_up_epochs"] + config["lr_decay_epoch_1"] + config["lr_decay_epoch_2"],
        ]
        return {
            "milestones": [epoch for epoch in lr_decay_milestones if epoch <= epochs],
            "gamma": config["lr_decay_factor"],
        }

    def _evaluate_hpo_model(self, model_wrapper, val_loader, cert_eval_samples, output_dir, config_hash, complete_verify):
        if complete_verify:
            return model_wrapper.evaluate_complete(
                test_loader=val_loader,
                test_samples=cert_eval_samples,
                timeout=60,
                results_path=f"{output_dir}/complete_verify/{config_hash}",
            )
        return model_wrapper.evaluate(test_loader=val_loader, test_samples=cert_eval_samples)

    def _hpo_runner(self, config, seed, epochs, train_loader, val_loader, output_dir, cert_eval_samples=1000, complete_verify=False):
        raise NotImplementedError('HPO can only be run on the concrete Wrappers!')
