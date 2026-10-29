"""Shared PyTorch interface for certified training, evaluation and hyperparameter optimisation."""
from functools import partial
import copy
import os
import torch
import torch.nn as nn
import numpy as np
from auto_LiRPA.bound_general import BoundedModule

from smac import Scenario, HyperparameterOptimizationFacade
from smac.utils.configspace import get_config_hash

from CTRAIN.complete_verification.neuralsat import validate_neuralsat_command, validate_neuralsat_options
from CTRAIN.eval.eval import eval_acc, eval_model, eval_complete
from CTRAIN.model_wrappers.configs import get_config_space
from CTRAIN.util.util import loader_radius


def _constant_value(hyperparameter):
    """Read a fixed ConfigSpace hyperparameter value across supported ConfigSpace versions."""
    if hasattr(hyperparameter, "value"):
        return hyperparameter.value
    return hyperparameter.default_value


class CTRAINWrapper(nn.Module):
    """Provide the common interface for CTRAIN's certified training wrappers.

    The wrapper encapsulates a PyTorch network and converts it to a bounded
    LiRPA graph for bound propagation. Method-specific subclasses configure the
    training procedures, while this base class provides evaluation, checkpoint
    resumption and hyperparameter optimisation. Since the wrapper inherits from
    nn.Module, it can be used in existing PyTorch workflows.
    """
    def __init__(self, model: nn.Module, eps:float, input_shape: tuple, train_eps_factor=1, lr=0.0005, optimizer_func=torch.optim.Adam,
                 lr_scheduler_func=torch.optim.lr_scheduler.MultiStepLR, lr_decay_kwargs=dict(milestones=(80, 90), gamma=0.2), bound_opts=dict(conv_mode='patches', relu='adaptive'), device='cuda', checkpoint_save_path=None, checkpoint_save_interval=10, population_bn=False):
        """Convert the network to a bounded graph and configure shared training options.

        Args:
            model (nn.Module): The neural network model to be wrapped.
            eps (float): The epsilon value for training.
            input_shape (tuple): The shape of the input tensor.
            train_eps_factor (float, optional): Factor to scale epsilon during training.
            lr (float, optional): Learning rate for the optimiser.
            optimizer_func (torch.optim.Optimizer, optional): The optimiser function to use.
            lr_scheduler_func (callable): Constructor for the learning-rate scheduler.
            lr_decay_kwargs (dict): Scheduler arguments, such as milestones and gamma for MultiStepLR.
            bound_opts (dict, optional): Options for bounding the model.
            device (str or torch.device, optional): The device to run the model on.
            checkpoint_save_path (str, optional): Path to save checkpoints.
            checkpoint_save_interval (int, optional): Interval to save checkpoints.
            population_bn (bool, optional): Recalibrate BatchNorm after each training epoch.

        Attributes:
            original_model (nn.Module): The original neural network model.
            eps (float): The epsilon value for training.
            train_eps (float): The scaled epsilon value for training.
            device (torch.device): The device to run the model on.
            n_classes (int): The number of classes in the model's output.
            bound_opts (dict): Options for bounding the model.
            bounded_model (BoundedModule): The bounded version of the original model.
            input_shape (tuple): The shape of the input tensor.
            optimizer_func (torch.optim.Optimizer): The optimiser function.
            optimizer (torch.optim.Optimizer): The optimiser instance.
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
        example_input = torch.ones(input_shape, device=device, dtype=next(model.parameters()).dtype)
        self.n_classes = len(model(example_input)[0])
        self.bound_opts = copy.deepcopy(bound_opts)
        self.bounded_model = BoundedModule(model=self.original_model, global_input=example_input, bound_opts=self.bound_opts, device=device)
        self.input_shape = input_shape

        self.optimizer_func = optimizer_func
        self.optimizer = optimizer_func(self.bounded_model.parameters(), lr=lr)

        self.lr_scheduler_func = lr_scheduler_func
        self.lr_decay_kwargs = copy.deepcopy(lr_decay_kwargs)
        self.lr_scheduler = self.lr_scheduler_func(self.optimizer, **self.lr_decay_kwargs)

        self.epoch = 0
        self.selection_state = {}
        self.history = []

        if original_train:
            self.original_model.train()
            self.bounded_model.train()

        self.checkpoint_path = checkpoint_save_path
        if checkpoint_save_path is not None:
            os.makedirs(self.checkpoint_path, exist_ok=True)

        self.checkpoint_save_interval = checkpoint_save_interval
        self.population_bn = population_bn

    def train(self, mode=True):
        """Set the wrapper and its registered models to the requested training mode.

        This method calls the `train` method on both the `original_model` and
        the `bounded_model` to set them into training mode
        """
        return super().train(mode)

    def eval(self):
        """Set the wrapper and its registered models to evaluation mode.

        This method sets both the original model and the bounded model to evaluation mode.
        In evaluation mode, certain layers like dropout and batch normalisation behave differently
        compared to training mode, typically affecting the model's performance and predictions.
        """
        return self.train(False)

    def forward(self, x):
        """Evaluate the bounded network on a batch of nominal inputs.

        Args:
            x (torch.Tensor): Input tensor to be passed through the model.

        Returns:
            (torch.Tensor): Output tensor after passing through the bounded model.
        """
        return self.bounded_model(x)

    def evaluate(self, test_loader, test_samples=np.inf, eval_method='ADAPTIVE',
                 adversarial_method='PGD', alpha_iterations=20, autoattack_kwargs=None,
                 propagation_batch_size=1):
        """Evaluate natural, certified and adversarial classification accuracy.

        The nominal raw-input radius is converted using the loader's normalisation
        metadata. Incomplete verification establishes a lower bound on robust
        accuracy, while adversarial searches identify counterexamples. Choose a
        single bounding method or progressively apply methods with ADAPTIVE.

        Args:
            test_loader (DataLoader): DataLoader containing the test dataset.
            test_samples (int, optional): Number of test samples to evaluate.
            eval_method (str or list, optional): The certification method to use. Options are
                'FORWARD', 'IBP', 'CROWN', 'ALPHA-CROWN',
                'CROWN-IBP', 'ADAPTIVE', or a list of methods
                (which results in an ADAPTIVE evaluation using
                these methods). 'ZONOTOPE' aliases 'FORWARD'.
            adversarial_method (str): Empirical evaluation method: 'PGD' or 'AUTOATTACK'.
            alpha_iterations (int): Number of optimisation iterations for alpha-CROWN relaxation
                slopes.
            autoattack_kwargs (dict, optional): Optional AutoAttack version, seed and verbosity settings.
            propagation_batch_size (int): Maximum forward-bound chunk size; default 1.

        Returns:
            (Tuple): Evaluation results in terms of std_acc, cert_acc and adv_acc.

        adversarial_method selects PGD (default) or AUTOATTACK; autoattack_kwargs
        configures its version/seed/verbosity. alpha_iterations controls ALPHA-CROWN.
        """
        parameter = next(self.bounded_model.parameters())
        example = torch.empty(self.input_shape, device=self.device, dtype=parameter.dtype)
        eps_std = loader_radius(self.eps, test_loader, example)
        return eval_model(self.bounded_model, test_loader, n_classes=self.n_classes, eps=eps_std, test_samples=test_samples, method=eval_method, device=self.device,
                          adversarial_method=adversarial_method, alpha_iterations=alpha_iterations,
                          autoattack_kwargs=autoattack_kwargs,
                          propagation_batch_size=propagation_batch_size)

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
        engine="abcrown",
        neuralsat_command=None,
        neuralsat_batch_size=1000,
    ):
        """Evaluate robustness using incomplete screening and a complete verifier.

        CTRAIN first applies inexpensive bounds and PGD, then sends unresolved
        inputs to alpha-beta-CROWN or NeuralSAT through ONNX and VNNLIB files.
        The per-query time limit controls the solver budget; cases that remain
        unresolved are retained in the persisted results.

        Args:
            test_loader (DataLoader): DataLoader for the test set.
            test_samples (int, optional): Number of test samples to evaluate.
            timeout (int, optional): Per-instance timeout for the verification process in
                seconds.
            no_cores (int, optional): Number of CPU cores to use for verification. Only
                relevant, if MIP refinement is used in abCROWN.
            abcrown_batch_size (int, optional): Batch size for abCROWN.  Decrease, if you run
                out of memory.
            abcrown_config_dict (dict, optional): Configuration dictionary for abCROWN according
                to the tools documentation.
            results_path (str, optional): Path to save abCROWN results and logs.
            warm_start (bool, optional): Reuse existing results from the results file.
            start_idx (int, optional): First dataset index to verify.
            end_idx (int, optional): Exclusive end dataset index to verify.
            results_filename (str, optional): JSON file name under results_path.
            engine (str): 'abcrown' (default) or 'neuralsat'.
            neuralsat_command (list): Argv prefix; None uses the bundled solver.
            neuralsat_batch_size (int): Maximum NeuralSAT parallel branches.

        Returns:
            (tuple): A tuple containing: std_acc (float): Standard accuracy of the model on the test set, certified_acc (float): Certified accuracy of the model on the test set and adv_acc (float): Adversarial accuracy of the model on the test set.
        """
        if engine not in ("abcrown", "neuralsat"):
            raise ValueError("engine must be 'abcrown' or 'neuralsat'")
        if engine == "neuralsat":
            neuralsat_command = validate_neuralsat_command(neuralsat_command)
            validate_neuralsat_options(timeout, self.device, neuralsat_batch_size)
            if abcrown_config_dict is not None:
                raise ValueError("abcrown_config_dict cannot be used with NeuralSAT")
        eps_std = self.eps / test_loader.std if test_loader.normalised else torch.tensor(self.eps)
        eps_std = torch.reshape(eps_std, (*eps_std.shape, 1, 1)).to(self.device)
        std_acc = eval_acc(self.bounded_model, test_loader=test_loader, test_samples=test_samples, device=self.device)
        certified_acc, adv_acc = eval_complete(
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
            engine=engine,
            neuralsat_command=neuralsat_command,
            neuralsat_batch_size=neuralsat_batch_size,
        )
        return std_acc, certified_acc, adv_acc

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """Return the bounded network parameters and persistent buffers for inference.

        The state dictionary contains the model parameters and persistent buffers.

        Returns:
            (dict): A dictionary containing the model's state.
        """
        return self.bounded_model.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)

    def load_state_dict(self, state_dict, strict = True):
        """Load bounded network parameters and persistent buffers from an inference state
        dictionary.

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
        """Iterate over the bounded network parameters exposed to the optimiser."""
        return self.bounded_model.parameters(recurse=recurse)
    # TODO: Add onnx export/loading

    def resume_from_checkpoint(self, checkpoint_path:str, train_loader, val_loader=None, end_epoch=None):
        """Restore the training trajectory and continue at its next epoch.

        The checkpoint supplies model, optimiser, learning-rate scheduler and
        selection state. Recreate the original schedule and loader protocol before
        resuming. These classifier checkpoints do not restore RNG or loader state;
        checkpoints without scheduler state cannot resume the trajectory reliably.

        Args:
            checkpoint_path (str): Path to the checkpoint file.
            train_loader (DataLoader): DataLoader for the training dataset.
            val_loader (DataLoader, optional): DataLoader for the validation dataset.
            end_epoch (int, optional): Epoch to prematurely end training at. Defaults to None.
        Loads model, optimiser and LR scheduler state, then resumes at the next
        epoch. Legacy checkpoints without scheduler state are rejected. RNG and
        data-loader state are not restored; this is not a bitwise replay.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if 'lr_scheduler_state_dict' not in checkpoint:
            raise ValueError(
                "This legacy checkpoint has no LR scheduler state; it cannot resume the "
                "training schedule reliably. Load its model_state_dict for inference instead."
            )
        model_state_dict = checkpoint['model_state_dict']
        self.load_state_dict(model_state_dict)
        from CTRAIN.train.certified.util import synchronise_bn
        synchronise_bn(self.original_model, self.bounded_model, train_loader, self.device,
                       loss_fusion_model=getattr(self, 'loss_fusion_model', None))
        self.epoch = checkpoint['epoch']
        self.selection_state = checkpoint.get('selection_state', {})
        self.history = checkpoint.get('history', [])
        optimizer_state_dict = checkpoint['optimizer_state_dict']
        self.optimizer.load_state_dict(optimizer_state_dict)
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])

        return self.train_model(train_loader, val_loader, start_epoch=self.epoch, end_epoch=end_epoch)

    def hpo_smac(self, train_loader, val_loader, budget=5*24*60*60, defaults=dict(), eval_samples=1000, output_dir='./smac_hpo', deterministic=False, seed=42, nat_loss_weight=1., adv_loss_weight=1., cert_loss_weight=1.):
        """Optimise a weighted combination of validation accuracies using SMAC3.

        This single-objective interface trains configurations from the method's
        search space and returns the selected configuration and loads its trained weights. Use hpo
        to explore the natural-certified accuracy trade-off through a Pareto front.

        Args:
            train_loader (DataLoader): DataLoader for the training dataset.
            val_loader (DataLoader): DataLoader for the validation dataset.
            budget (int, optional): Time budget for the HPO process in seconds.
            defaults (dict, optional): Default hyperparameter values.
            eval_samples (int, optional): Number of samples to use for loss computation.
            output_dir (str, optional): Directory to store HPO results.
            deterministic (bool, optional): Whether SMAC3 should treat the objective function as
                deterministic. Speeds up the optimisation.
            seed (int, optional): Random seed for reproducibility of the HPO.
            nat_loss_weight (float, optional): Weight for the natural accuracy in the loss
                function.
            adv_loss_weight (float, optional): Weight for the adversarial accuracy in the loss
                function.
            cert_loss_weight (float, optional): Weight for the certified accuracy in the loss
                function.

        Returns:
            (Configuration): The best hyperparameter configuration found during the optimisation.
        """
        os.makedirs(output_dir, exist_ok=True)
        if os.listdir(output_dir):
            assert False, 'Output directory for HPO is not empty!'

        os.makedirs(f'{output_dir}/nets', exist_ok=True)
        os.makedirs(f'{output_dir}/smac/', exist_ok=True)

        eps_std = self.eps / train_loader.std if train_loader.normalised else torch.tensor(self.eps)
        scenario = Scenario(
            configspace=get_config_space(self, self.num_epochs, eps_std, defaults=defaults),
            deterministic=deterministic,
            walltime_limit=budget,
            n_trials=np.inf,
            output_directory=f'{output_dir}/smac/',
            seed=seed,
            use_default_config=True if len(defaults.values()) > 0 else False
        )
        initial_design = HyperparameterOptimizationFacade.get_initial_design(scenario, n_configs=len(scenario.configspace))
        smac = HyperparameterOptimizationFacade(
            scenario,
            partial(self._hpo_runner, epochs=self.num_epochs, train_loader=train_loader, val_loader=val_loader, cert_eval_samples=eval_samples, output_dir=output_dir, nat_loss_weight=nat_loss_weight, adv_loss_weight=adv_loss_weight, cert_loss_weight=cert_loss_weight),
            initial_design=initial_design,
            overwrite=True,
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
        """Search for Pareto-optimal natural and certified validation accuracies.

        The multi-objective procedure uses Optuna with BoTorch or NSGA-II to explore
        the method-specific configuration space. It retains the trade-off solutions
        so users can select the network appropriate to their application.

        The study and checkpoints are persisted under output_dir. The returned front
        uses trials meeting min_nat_acc and min_cert_acc when feasible trials exist;
        otherwise it retains the unconstrained front and records infeasibility.
        No checkpoint is loaded implicitly: select a returned checkpoint_path and
        load the desired network explicitly.

        Args:
            train_loader (DataLoader): DataLoader for the training dataset.
            val_loader (DataLoader): DataLoader for validation.
            budget_time (int): Wall-clock budget in seconds.
            budget_trials (int or float): Maximum number of trials. ``np.inf`` disables this
                limit.
            defaults (dict): Default hyperparameter values for the ConfigSpace.
            eval_samples (int): Number of validation samples used per trial.
            output_dir (str): Directory for Optuna DBs and trial checkpoints.
            min_nat_acc (float): Feasibility threshold for natural accuracy.
            min_cert_acc (float): Feasibility threshold for certified accuracy.
            seed (int): Random seed.
            sampler (str or optuna.samplers.BaseSampler): ``"botorch"``, ``"nsgaii"``, or a
                sampler instance.
            complete_verify (bool): Use complete verification for the certified objective.
            study_name (str): Name of the persisted Optuna study.

        Returns:
            (list[dict]): Pareto-optimal trials with their configs, metrics,
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
        """Optimise a weighted combination of validation accuracies using Optuna.

        This interface is useful when the desired trade-off between natural,
        adversarial and certified accuracy can be expressed by fixed weights.

        By default, the optimised objective is ``nat_acc + cert_acc``. The
        objective can be changed with ``nat_acc_weight``, ``adv_acc_weight``,
        and ``cert_acc_weight``.

        Returns:
            (dict): Best trial with its scalar objective value, config, metrics,
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
        """Sample a valid method configuration from the ConfigSpace through an Optuna trial."""
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
        """Reconstruct a method configuration from a completed Optuna trial and fixed defaults."""
        config = {}
        for hp_name in config_space:
            hp = config_space[hp_name]
            if hp.__class__.__name__ == "Constant":
                config[hp_name] = _constant_value(hp)
            else:
                config[hp_name] = trial.params[hp_name]
        return config

    def _get_optuna_sampler(self, optuna, sampler, constraints_func, seed):
        """Construct the requested multi-objective Optuna sampler with optional feasibility
        constraints.
        """
        if not isinstance(sampler, str):
            return sampler
        if sampler == "nsgaii":
            return optuna.samplers.NSGAIISampler(constraints_func=constraints_func, seed=seed)
        if sampler != "botorch":
            raise ValueError("sampler must be 'botorch', 'nsgaii', or an Optuna sampler instance")
        try:
            from optuna_integration.botorch import BoTorchSampler
        except (ImportError, ModuleNotFoundError) as exc:
            raise ImportError(
                "The BoTorch sampler requires optuna-integration with BoTorch support. "
                "Install optuna-integration[botorch] or use sampler='nsgaii'."
            ) from exc
        return BoTorchSampler(constraints_func=constraints_func, seed=seed, device=torch.device(self.device))

    def _get_optuna_single_objective_sampler(self, optuna, sampler, seed):
        """Construct the requested scalar-objective Optuna sampler."""
        if not isinstance(sampler, str):
            return sampler
        if sampler == "botorch":
            return self._get_optuna_sampler(optuna, sampler, None, seed)
        if sampler == "tpe":
            return optuna.samplers.TPESampler(seed=seed)
        if sampler == "random":
            return optuna.samplers.RandomSampler(seed=seed)
        if sampler == "nsgaii":
            return optuna.samplers.NSGAIISampler(seed=seed)
        raise ValueError("sampler must be 'botorch', 'tpe', 'random', 'nsgaii', or an Optuna sampler instance")

    def _constrained_pareto_trials(self, trials):
        """Select feasible Pareto trials when available, otherwise retain the unconstrained front."""
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
        """Return whether the first objective vector weakly improves every maximisation objective
        and strictly improves at least one.
        """
        return all(value >= other for value, other in zip(values, other_values)) and any(
            value > other for value, other in zip(values, other_values)
        )

    def _optuna_trial_result(self, trial, config_space, output_dir):
        """Build a trial record containing configuration, accuracies, feasibility and checkpoint path."""
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
        """Resolve the optimiser constructor selected by the sampled configuration."""
        optimizer_name = config["optimizer_func"]
        if optimizer_name == "sgd":
            return torch.optim.SGD
        if optimizer_name == "adam":
            return torch.optim.Adam
        if optimizer_name == "adamw":
            return torch.optim.AdamW
        if optimizer_name == "radam":
            return torch.optim.RAdam
        raise ValueError(f"Unknown optimizer_func: {optimizer_name}")

    def _lr_decay_kwargs_from_config(self, config, epochs):
        """Translate sampled decay offsets into scheduler milestones and gamma."""
        lr_decay_milestones = [
            config["warm_up_epochs"] + config["ramp_up_epochs"] + config["lr_decay_epoch_1"],
            config["warm_up_epochs"] + config["ramp_up_epochs"] + config["lr_decay_epoch_1"] + config["lr_decay_epoch_2"],
        ]
        return {
            "milestones": [epoch for epoch in lr_decay_milestones if epoch <= epochs],
            "gamma": config["lr_decay_factor"],
        }

    def _evaluate_hpo_model(self, model_wrapper, val_loader, cert_eval_samples, output_dir, config_hash, complete_verify):
        """Evaluate a trial using incomplete bounds or the requested complete-verification workflow."""
        if complete_verify:
            return model_wrapper.evaluate_complete(
                test_loader=val_loader,
                test_samples=cert_eval_samples,
                timeout=60,
                results_path=f"{output_dir}/complete_verify/{config_hash}",
            )
        return model_wrapper.evaluate(test_loader=val_loader, test_samples=cert_eval_samples)

    def _hpo_runner(self, config, seed, epochs, train_loader, val_loader, output_dir, cert_eval_samples=1000, complete_verify=False):
        """Define the method-specific training and evaluation interface for HPO.

        Concrete wrappers implement this function. The base implementation raises
        NotImplementedError.
        """
        raise NotImplementedError('HPO can only be run on the concrete Wrappers!')
