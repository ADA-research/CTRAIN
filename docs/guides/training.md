# Training contracts

Wrappers configure the library's trainers and call `train_model`. Trainers own
schedules and epoch loops; losses and bounds own their computations.

## Trainer lifecycle

Certified trainers use the following lifecycle, keeping their numerical update
steps explicit:

1. Validate the epoch range, loaders, checkpoint interval and method options.
2. Prepare schedules and any method-specific initialization.
3. For each logical loader batch, compute scheduled values, run the method's
   optimizer update, accumulate metrics and advance batch LR schedules.
4. After each epoch, synchronize/recalibrate BatchNorm, advance epoch schedules,
   evaluate validation data, record history and save the trajectory.
5. With validation enabled, restore selected inference weights after saving.

Shared validation, sample-weighted loss accumulation, selection bookkeeping and
checkpoint writing live in library helpers. Each trainer retains its own batch
computation: set-based propagation accumulates chunk gradients before one update,
GLS evaluates a closure at sampled parameters, and properties sample regions
independently of task examples and may refine their cover after an epoch.

Certified wrappers expose `history`; new classifier entries contain the completed
`epoch`, sample-weighted training `loss`, final batch's training `eps` and the LR
after scheduled stepping. Property history additionally records task/property
losses, phase, effective property weight and whole-cover diagnostics. These
training losses can be surrogates; they are not certificate metrics.

Existing positional trainer entry points remain supported. New GLS, set-based
and IBP-R integrations should use `validation_eps`; the legacy `eval_eps` keyword
still works and conflicting values are rejected. GLS and set-based wrappers also
forward `lr_decay_schedule_unit='epoch'` or `'batch'`, including HPO recreation.
Property LR scheduling remains once per epoch.

Property training takes explicit models, bounds, task loss, optimizer, options
and a `PropertyTrainingState`. The state holds the partition, completed epoch,
history, selection and pending loader RNG state. The wrapper delegates to this
API and exposes its established `partition`, `epoch`, `history`, `best_score`,
`best_weights`, `inference_selected` and `loader_rng` attributes as state views.
Legacy `property_train_model(wrapper, loader, ...)` calls remain supported.
Loss computations use `compute_property_training_loss` with explicit inputs;
`get_property_training_loss(wrapper, ...)` remains a compatibility adapter.
The runnable CPU regression `tests/test_training_lifecycle.py` checks the shared
contracts, direct property training, compatibility, checkpoint migration, RNG
consumption and partial-run resumption.

## Schedules and radii

`num_epochs` describes the full schedule. `start_epoch` counts completed epochs;
`end_epoch` is exclusive. Empty ranges do not initialize or update weights.
Preserve the full epoch budget and schedule when running partial trajectories.

Linear ramps retain their batch-boundary convention: GLS evaluates at the start
of a batch; set-based epsilon and property-weight ramps evaluate at its end.
Other methods retain their smoothed schedules.

## Checkpointing

Automatic saves occur at the configured interval and at the end of each call,
before restoring selected inference weights. Resumption restores the trajectory,
not inference-selected weights. Property checkpoints additionally retain the
partition, configuration and RNG state. See [checkpoint contracts](checkpoints.md).
