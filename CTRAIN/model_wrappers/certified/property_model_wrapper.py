"""Configure certified training for logical output properties over input boxes.

The wrapper owns the model, bounds, optimizer and resumable state. The epoch
loop lives in `CTRAIN.train.certified.properties.property_train_model`.
"""
import copy
import math
import random
import warnings
from pathlib import Path
import numpy as np
import torch
from torch import nn
from CTRAIN.properties import PropertySpec
from CTRAIN.bound.properties import PropertyBounds, METHODS
from CTRAIN.eval.properties import initial_partition, partition_state, restore_partition, validate_partition
from CTRAIN.properties.training import TRAINING_METHODS, resolve_training_options


class PropertyModelWrapper(nn.Module):
    """Train a task model while encouraging whole-domain output properties.

    Each `PropertySpec` pairs an affine/Boolean output formula with one or more
    input boxes. Bounds are computed over entire boxes, independently of the
    task examples. The verification graph shares the task model's parameters;
    an optional fixed computation can expose quantities such as next states.

    Training minimizes task loss plus the selected weighted property loss.
    Bound methods penalize lower margins; adversarial and hybrid methods use
    pointwise or combined surrogates. A low training loss does not establish a
    certificate: use `certify_properties` to check every covering cell afresh.
    Both training and verification use evaluation-mode model behavior, with
    gradients enabled during training and caller modes restored afterwards.

    Args:
        model (torch.nn.Module): Task model to optimize and return from training.
        input_shape (tuple[int, ...]): Task input shape, excluding the batch axis.
        properties (Iterable[PropertySpec]): Nonempty specifications with unique
            names. Copied at construction to isolate them from caller mutation.
        task_loss (Callable): Loss called as `task_loss(prediction, target)`;
            its result is averaged. Modules are moved to `device`.
        num_epochs (int): Total trajectory length, also the default training end.
        bound_method (str): Training and epoch-selection bound method: `IBP`,
            `CROWN-IBP` or `CROWN`. When training_method is supplied, this
            controls fresh whole-box epoch selection only.
        training_method (str | None): IBP, CROWN-IBP, CROWN, MTL-IBP, CC-IBP,
            EXP-IBP, SABR, PGD, IBP-R or GLS. None retains bound_method training.
            Hybrid, adversarial, regularized and smoothed property methods are
            experimental adaptations, not established paper reproductions.
        training_options (dict | None): Parameters of the selected method,
            copied and validated before conversion. Supports loss='hinge' or
            'cross_entropy' and optional gradient_clip for all methods. Hybrids
            support PGD controls and ReLU retention; expressive methods add
            alpha, CROWN-IBP adds beta, SABR adds subselection_ratio, IBP-R
            adds rho/masking, and GLS adds estimator, samples, sigma, sigma_lr
            and scale decay. See the
            property guide for defaults and the support table.
        verification_model (torch.nn.Module | None): Graph whose outputs the
            formulas reference. Defaults to `model`. Must contain `model` by
            identity and may not add trainable parameters.
        verification_input_shape (tuple[int, ...] | None): Verification input
            shape without the batch axis; defaults to `input_shape`. Property
            boxes use these coordinates, with no implicit normalization.
        device (str | torch.device): Model and training device; defaults to CPU.
        lr (float): Positive optimizer learning rate.
        optimizer_func (Callable): Optimizer constructor accepting model
            parameters and `lr`; defaults to Adam.
        lr_scheduler_func (Callable | None): Optional scheduler constructor.
            The trainer steps it once after each epoch.
        lr_scheduler_kwargs (dict | None): Scheduler constructor arguments.
        property_weight (float): Nonnegative multiplier of the property loss.
        margin (float): Positive target lower margin for the hinge penalty.
            Logical certification instead requires a lower margin above zero.
        weight_ramp_epochs (int): Epochs for linearly increasing the property
            weight from zero to `property_weight`. Zero applies it immediately.
        property_batch_size (int): Number of covering cells sampled with
            replacement per task batch; also the bound-evaluation batch size.
        refinement (bool): Split uncertified cells after each epoch, then
            reevaluate the refined cover. Defaults to False.
        split_count (int): Maximum number of cells split per epoch. Each split
            adds one cell and divides the parent's sampling weight equally.
        max_cells (int): Refinement cell limit per original input box, not a
            limit on the total cover across all specifications.
        max_depth (int): Maximum number of splits along a cell's lineage.
        population_bn (bool): Recalibrate task-model BatchNorm statistics on
            training inputs after each epoch, before selection. Otherwise
            running statistics stay fixed during training.
        checkpoint_save_path (str | pathlib.Path | None): Optional directory
            for resumable epoch checkpoints. Existing files are not overwritten.
        checkpoint_save_interval (int): Save every this many completed epochs
            and at the end of each training call.
        bound_opts (dict | None): auto_LiRPA options. ``property_bound='atomic'``
            optionally bounds affine leaves and combines them with monotone
            Boolean logic, avoiding the full nonlinear formula graph. This
            remains sound but can be weaker for disjunctions. Defaults to matrix
            convolution and adaptive ReLU bounds. Automatic parameter-gradient
            toggling is disabled so evaluation-mode training can differentiate.

    Attributes:
        original_model (torch.nn.Module): Task model sharing verification weights.
        partition (list[dict]): Covering cells with lineage and sampling weights.
            Every original box starts with equal weight; overlapping boxes are
            still separate sampling regions.
        epoch (int): Number of completed trajectory epochs.
        history (list[dict]): Per-epoch task/property losses, validation loss,
            whole-cover certification flag, cell count and minimum lower bound.
        configuration (dict): Serializable property and training configuration
            used to reject incompatible checkpoints.
        inference_selected (bool): True after restoring validation-selected
            weights. Reload an epoch checkpoint before resuming the trajectory.
    """
    def __init__(self, model, input_shape, *, properties, task_loss, num_epochs,
                 bound_method='CROWN-IBP', verification_model=None,
                 verification_input_shape=None, device='cpu', lr=1e-3,
                 optimizer_func=torch.optim.Adam, lr_scheduler_func=None,
                 lr_scheduler_kwargs=None, property_weight=1., margin=1e-3,
                 weight_ramp_epochs=0, property_batch_size=32, refinement=False,
                 split_count=8, max_cells=1024, max_depth=12, population_bn=False,
                 checkpoint_save_path=None, checkpoint_save_interval=10, bound_opts=None,
                 training_method=None, training_options=None):
        super().__init__()
        training_method, training_options = resolve_training_options(
            bound_method if training_method is None else training_method, training_options)
        if not isinstance(model, nn.Module) or not callable(task_loss):
            raise ValueError('model must be a module and task_loss must be callable')
        properties = copy.deepcopy(tuple(properties))
        if not properties or any(not isinstance(p, PropertySpec) for p in properties):
            raise ValueError('properties must contain PropertySpec instances')
        if len({p.name for p in properties}) != len(properties):
            raise ValueError('Property names must be unique')
        for name, value in dict(num_epochs=num_epochs, property_batch_size=property_batch_size,
                                split_count=split_count, max_cells=max_cells,
                                checkpoint_save_interval=checkpoint_save_interval).items():
            if type(value) is not int or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        for name, value in dict(max_depth=max_depth, weight_ramp_epochs=weight_ramp_epochs).items():
            if type(value) is not int or value < 0:
                raise ValueError(f'{name} must be a nonnegative integer')
        if weight_ramp_epochs > num_epochs or bound_method not in METHODS:
            raise ValueError('Invalid ramp duration or bound method')
        for name, value in dict(lr=lr, margin=margin, property_weight=property_weight).items():
            if isinstance(value, bool) or not math.isfinite(value) or value < 0 or (name != 'property_weight' and value == 0):
                raise ValueError(f'Invalid {name}')
        if type(population_bn) is not bool or type(refinement) is not bool:
            raise ValueError('population_bn and refinement must be bool')
        self.input_shape = tuple(input_shape)
        self.verification_input_shape = tuple(verification_input_shape or self.input_shape)
        if not self.input_shape or not self.verification_input_shape or any(
                type(n) is not int or n < 1 for n in (*self.input_shape, *self.verification_input_shape)):
            raise ValueError('Input shapes exclude the batch dimension and require positive integers')
        if verification_model is not None and not isinstance(verification_model, nn.Module):
            raise ValueError('verification_model must be a module')
        self.original_model = model.to(device)
        self.verification_model = model if verification_model is None else verification_model.to(device)
        if not isinstance(self.verification_model, nn.Module) or not any(
                m is model for m in self.verification_model.modules()):
            raise ValueError('verification_model must contain the supplied task model by identity')
        source = {id(p) for p in model.parameters()}
        if any(p.requires_grad and id(p) not in source for p in self.verification_model.parameters()):
            raise ValueError('verification_model may not add trainable parameters')
        self.reference = next(model.parameters(), None)
        if self.reference is None or not self.reference.is_floating_point():
            raise ValueError('Task model requires floating-point trainable parameters')
        self.device, self.properties = torch.device(device), properties
        self.task_loss = task_loss.to(device) if isinstance(task_loss, nn.Module) else task_loss
        self.bounds = PropertyBounds(self.verification_model, self.verification_input_shape,
                                     properties, bound_opts)
        self.training_method, self.training_options = training_method, training_options
        if training_method == 'GLS':
            from CTRAIN.train.certified.gls import GLSOptimizer
            options = {k: training_options[k] for k in ('estimator', 'samples', 'sigma', 'sigma_lr', 'gradient_clip')}
            self.optimizer = GLSOptimizer(model, optimizer_func, lr, **options)
        else:
            self.optimizer = optimizer_func(model.parameters(), lr=lr)
        if training_method not in METHODS:
            warnings.warn(f'General-property {training_method} training is experimental; '
                          'certification and selection use fresh full-box bounds.', UserWarning, stacklevel=2)
        if training_method == 'EXP-IBP' and training_options['loss'] == 'hinge':
            warnings.warn('EXP-IBP with hinge loss can suppress gradients when either factor is zero.',
                          UserWarning, stacklevel=2)
        self.lr_scheduler = (None if lr_scheduler_func is None else
                             lr_scheduler_func(self.optimizer, **(lr_scheduler_kwargs or {})))
        self.num_epochs, self.bound_method = num_epochs, bound_method
        self.property_weight, self.margin = property_weight, margin
        self.weight_ramp_epochs, self.property_batch_size = weight_ramp_epochs, property_batch_size
        self.refinement, self.split_count = refinement, split_count
        self.max_cells, self.max_depth = max_cells, max_depth
        self.population_bn = population_bn
        self.checkpoint_path = None if checkpoint_save_path is None else Path(checkpoint_save_path)
        self.checkpoint_save_interval = checkpoint_save_interval
        self.partition = initial_partition(properties)
        self.epoch, self.history = 0, []
        self.best_score, self.best_weights = None, None
        self.inference_selected = False
        self.loader_rng = None
        self.configuration = dict(properties=[p.to_dict() for p in properties],
            input_shape=list(self.input_shape), verification_input_shape=list(self.verification_input_shape),
            num_epochs=num_epochs, bound_method=bound_method, margin=margin,
            property_weight=property_weight, weight_ramp_epochs=weight_ramp_epochs,
            property_batch_size=property_batch_size, refinement=refinement,
            split_count=split_count, max_cells=max_cells, max_depth=max_depth,
            population_bn=population_bn, lr=lr, optimizer=optimizer_func.__name__,
            scheduler=None if lr_scheduler_func is None else lr_scheduler_func.__name__,
            scheduler_kwargs=lr_scheduler_kwargs or {}, bound_opts=bound_opts or {})
        self.configuration.update(training_method=training_method, training_options=copy.deepcopy(training_options))

    def forward(self, inputs):
        """Return task-model predictions, without appending property margins."""
        return self.original_model(inputs)

    def train_model(self, train_loader, val_loader=None, start_epoch=None, end_epoch=None):
        """Run the property trainer and return the task model.

        Args:
            train_loader (torch.utils.data.DataLoader): Nonempty loader of
                `(input, target)` pairs in task-model input coordinates.
            val_loader (torch.utils.data.DataLoader | None): Optional task-loss
                validation loader. Enables model selection: certification first,
                then validation loss; before certification, minimum lower bound
                first and validation loss as a tie-breaker.
            start_epoch (int | None): Zero-based starting epoch. Defaults to the
                completed `epoch` count and must equal it. Resume by loading a
                checkpoint rather than skipping optimizer or partition state.
            end_epoch (int | None): Exclusive ending epoch; defaults to
                `num_epochs`. Supports a partial trajectory run.

        Returns:
            torch.nn.Module: Validation-selected task model when a validation
            loader is supplied; otherwise the final trajectory model. Its
            caller mode is restored. Optimizer state is not rewound on selection.
        """
        from CTRAIN.train.certified.properties import property_train_model
        return property_train_model(self, train_loader, val_loader, start_epoch, end_epoch)

    def certify_properties(self, **kwargs):
        """Compute fresh bounds on every obligation in the declared domain cover.

        Keyword arguments are forwarded to
        `CTRAIN.eval.properties.certify_properties`. Useful options are `method`
        (defaults to `CROWN`, independently of the training method), `refine`
        (defaults to True), and `max_cells`/`max_depth`.
        `properties=[...]` evaluates separate specifications, starting from their
        original boxes with fresh bound graphs and the same bound options. This
        leaves the training properties and partition unchanged.
        `method='ALPHA-CROWN'` optimizes slopes; set its iteration budget through
        `bound_opts={'optimize_bound_args': {'iteration': 100}}` at construction.
        This method is available for certification, not training. `complete=True` enables
        alpha-beta-CROWN fallback and requires `results_path`; `timeout` is the
        solver time limit per unresolved cell. Evaluation refinement does not
        replace the stored training partition.

        Returns:
            dict: Whole-cover `certified` flag, per-property summaries, per-cell
            bounds/outcomes, method, elapsed time and numerical scope. All cells
            must have strictly positive lower margins or an exported-graph
            proof. Unknown bounds do not establish a violation. These are
            floating-point certificates, without a directed-rounding proof.
        """
        from CTRAIN.eval.properties import certify_properties
        return certify_properties(self, **kwargs)

    def save_checkpoint(self, path, train_loader=None):
        """Save the current training trajectory to a new file.

        Includes model, optimizer/scheduler, partition, epoch, selection history
        and Python/NumPy/PyTorch RNG states. When supplied, the loader's explicit
        generator state is saved too. Automatic epoch saves occur before
        validation-selected inference weights are restored.

        Args:
            path (str | pathlib.Path): New checkpoint filename.
            train_loader (torch.utils.data.DataLoader | None): Optional loader
                whose explicit generator should be included for resumption.

        Raises:
            ValueError: Validation-selected weights have replaced trajectory
                weights. Load an epoch checkpoint before saving a trajectory.
            FileExistsError: The destination already exists.
        """
        if self.inference_selected:
            raise ValueError('Selected inference weights are not a resumable trajectory; load an epoch checkpoint first')
        path = Path(path)
        if path.exists():
            raise FileExistsError(f'Checkpoint already exists: {path}')
        path.parent.mkdir(parents=True, exist_ok=True)
        state = dict(configuration=self.configuration, epoch=self.epoch,
            model=self.original_model.state_dict(), optimizer=self.optimizer.state_dict(),
            scheduler=None if self.lr_scheduler is None else self.lr_scheduler.state_dict(),
            partition=partition_state(self.partition), history=self.history,
            best_score=self.best_score, best_weights=self.best_weights,
            rng=torch.get_rng_state(), numpy_rng=np.random.get_state(), python_rng=random.getstate(),
            cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            loader_rng=(train_loader.generator.get_state() if train_loader is not None
                        and train_loader.generator is not None else None))
        torch.save(state, path)

    def load_checkpoint(self, path):
        """Restore a trusted epoch checkpoint and return this wrapper.

        Reconstruct the same model, verification graph and task loss before
        loading. Property/training configuration must match. This restores the
        trajectory, optimizer/scheduler, validated cover, history and RNG state;
        loader generator restoration is deferred until the next training call.
        It also clears the inference-selection guard and synchronizes bound
        graph buffers. Arbitrary multi-worker/CUDA determinism is not guaranteed.

        Args:
            path (str | pathlib.Path): Trusted local checkpoint. Loading uses
                Python deserialization because the checkpoint includes RNG state.

        Returns:
            PropertyModelWrapper: This wrapper, ready to resume training.

        Raises:
            ValueError: Configuration differs or the saved partition fails to
                cover the declared input boxes.
        """
        # Only load trusted local training checkpoints (they include Python RNG state).
        state = torch.load(path, map_location='cpu', weights_only=False)
        configuration = copy.deepcopy(state['configuration'])
        configuration.setdefault('training_method', configuration['bound_method'])
        configuration['training_method'], configuration['training_options'] = resolve_training_options(
            configuration['training_method'], configuration.get('training_options'))
        if configuration != self.configuration:
            raise ValueError('Checkpoint property/training configuration differs')
        partition = restore_partition(state['partition'])
        validate_partition(partition, self.properties)
        self.original_model.load_state_dict(state['model'])
        self.optimizer.load_state_dict(state['optimizer'])
        if self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(state['scheduler'])
        self.partition = partition
        self.epoch, self.history = state['epoch'], state['history']
        self.best_score, self.best_weights = state['best_score'], state['best_weights']
        self.inference_selected = False
        self.loader_rng = state['loader_rng']
        torch.set_rng_state(state['rng'])
        np.random.set_state(state['numpy_rng'])
        random.setstate(state['python_rng'])
        if state['cuda_rng'] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state['cuda_rng'])
        self.bounds.sync_buffers()
        return self
