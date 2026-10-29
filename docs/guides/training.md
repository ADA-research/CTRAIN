# Training

CTRAIN provides a common workflow for certified training: load data, construct a PyTorch model, select a method wrapper and call `train_model`. Wrappers expose configuration and forward it to the corresponding trainer. Trainers manage schedules, optimisation, BatchNorm updates, validation and checkpoint saving, while losses and bound functions implement the method's computations.

This guide explains the training behaviour shared across methods. The individual [method pages](../index.md#overview) describe their objectives and method-specific parameters.

## Configuring a Training Run

In the following example, we train CNN7 on MNIST using [IBP](../ibp.md). First, we load the data with a validation holdout and construct the network:

```python
from CTRAIN.data_loaders import load_mnist
from CTRAIN.model_definitions import CNN7_Shi
from CTRAIN.model_wrappers import ShiIBPModelWrapper

train_loader, val_loader, test_loader = load_mnist(batch_size=128, seed=0)
input_shape = (1, 28, 28)
model = CNN7_Shi(in_shape=input_shape)
```

Now, we configure 70 epochs, including one warm-up epoch and 40 ramp-up epochs. The learning rate decreases by a factor of 0.2 at epochs 50 and 60:

```python
wrapped_model = ShiIBPModelWrapper(
    model=model,
    input_shape=input_shape,
    eps=0.1,
    num_epochs=70,
    warm_up_epochs=1,
    ramp_up_epochs=40,
    lr=5e-4,
    lr_decay_kwargs={"milestones": [50, 60], "gamma": 0.2},
    population_bn=False,
    device="cuda",
)
wrapped_model.train_model(train_loader, val_loader=val_loader)
```

Supplying validation data enables the method's model-selection policy. Omitting `val_loader` returns the final weights. After training, evaluate the selected or final model through the common [evaluation interface](evaluation.md).

## Schedules and Radii

Warm-up typically uses clean cross-entropy before the trainer increases the perturbation radius towards its target. Training at the full target then continues for the remaining epochs. The exact radius schedule is method-specific: the IBP-based methods use their configured epsilon scheduler, while GLS and set-based training use linear ramps with their documented batch-boundary conventions.

The wrapper's `eps` describes the evaluation radius. Where supported, `train_eps_factor` multiplies the training target while leaving evaluation at `eps`. For example, `eps=2 / 255` and `train_eps_factor=1.1` train towards `2.2 / 255` and evaluate at `2 / 255`. Radii are specified before loader normalisation; see [input coordinates](data_and_models.md#input-coordinates-and-loader-metadata).

Attack search ranges can be configured independently. The `pgd_eps_factor` scales the search radius in supported hybrid methods without enlarging their nominal bound region. SABR and expressive losses apply it to input search, TAPS to latent search, and STAPS to both. Explicit input `pgd_ptb` bounds override this factor for the input attack.

Learning-rate schedules are configured through the wrapper's scheduler and decay arguments. GLS and set-based training additionally expose `lr_decay_schedule_unit="epoch"` or `"batch"`; property schedulers advance once per epoch. Keep milestone units consistent with the selected schedule.

### Partial Runs

`num_epochs` defines the complete schedule. The `start_epoch` and exclusive `end_epoch` arguments allow part of that schedule to run without redefining warm-up, ramp-up or learning-rate milestones. For example, on a fresh wrapper configured for 70 epochs, `wrapped_model.train_model(train_loader, end_epoch=15)` runs epochs 0 through 14 of the original schedule.

When continuing an interrupted experiment, restore a trajectory checkpoint so that optimiser, scheduler and selection state accompany the weights. Setting `start_epoch` alone does not restore them. An empty epoch range performs no initialisation or updates. See [saving and resumption](checkpoints.md#training-trajectories).

## Shared Options

The following settings are available on the wrappers that expose them:

| Option | Effect |
|---|---|
| `population_bn=False` | Optionally recalibrate BatchNorm on training inputs after each epoch, before selection and checkpointing. |
| `pgd_eps_factor=1.0` | Scale the attack search range by a positive finite factor. |
| `relu_upper_retention=1.0` | Retain this fraction of unstable ReLU upper bounds in supported training IBP computations; requires `(0, 1]`. |
| `gradient_clip` | Limit the global gradient norm, with defaults and estimator-specific behaviour documented per method. |

A retention of `0.8` keeps 80% of unstable upper bounds. This corresponds to shrinkage `0.2` in the original TAPS convention. Values below one produce an unsound training surrogate. CTRAIN restores ordinary interval propagation after each computation, including failures, and later certification uses fresh bounds over the nominal region.

Set-based training has separate input-set and propagation options. General-property methods use `training_options`, including `pgd_region_factor` for their attack region. Consult the relevant method page before transferring options between workflows.

## BatchNorm and Modes

Hybrid classification methods search for adversarial points in evaluation mode, then evaluate their losses in the caller's training mode. This is CTRAIN's chosen BatchNorm policy and can differ from the original method implementation. Bounded, split and loss-fusion graphs may hold separate buffers; the trainers synchronise those used for evaluation and checkpointing.

With `population_bn=True`, the trainer recalibrates statistics on training inputs after each epoch. This happens before validation selection and checkpoint saving. Recalibration preserves module modes and BatchNorm momentum, including on failure.

[Set-based training](../set_based.md#batchnorm-and-batching) differentiates through clean nominal chunk statistics, so its propagation chunk size affects training. [General-property training](../properties.md#training-and-domains) instead uses fixed BatchNorm statistics in evaluation mode with parameter gradients enabled. Its optional population recalibration uses the task training inputs.

## Training History and Model Selection

The wrappers expose `history` with per-epoch results. Classifier entries include the completed epoch, sample-weighted training loss, final batch's training radius and learning rate after scheduled stepping. Property histories additionally record task and property losses, the current phase, effective property weight and whole-cover diagnostics.

Training losses describe the objective used for optimisation. A subregion loss, shrunk bound or latent proxy does not establish certification of the complete input region. Use fresh evaluation for reporting.

With validation data, selection starts at the point defined by the method, usually after ramp-up. Criteria include robust validation loss, certified accuracy, adversarial accuracy and latent-margin proxy accuracy. The [selection table](checkpoints.md#validation-selection) records each policy. Selected inference weights and resumable training trajectories are saved separately because restoring the former does not rewind the optimiser.

## General-Property Training

`PropertyModelWrapper` follows the same data/model/wrapper pattern, but combines a supplied task loss with satisfaction-margin penalties over declared property boxes. Task examples and property boxes are sampled independently. The trainer can refine unresolved boxes while preserving a cover of the original domain.

Property warm-up optimises only the task loss. The subsequent weight ramp increases the property contribution, rather than introducing an implicit image perturbation radius. See the [property guide](../properties.md) for objectives, supported strategies and complete-domain certification.

For lower-level extensions, the API reference documents the [training utilities](../api/train/certified/util.md), [epsilon scheduler](../api/train/certified/eps_scheduler.md) and [property trainer](../api/train/certified/properties.md).
