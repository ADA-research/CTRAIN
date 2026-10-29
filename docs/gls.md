# Gaussian Loss Smoothing

Tighter bounding methods can provide more precise approximations of a network's reachable outputs, but the resulting training objectives may be difficult to optimise. [Balauca et al. (2025), *Gaussian Loss Smoothing Enables Certified Training with Tight Convex Relaxations*](https://arxiv.org/pdf/2403.07095), address this challenge by smoothing the loss over Gaussian perturbations of the network parameters. This enables certified training with tighter relaxations, such as those used by CROWN.

CTRAIN provides Gaussian loss smoothing through `GLSModelWrapper`, supporting Randomized Gradient Smoothing (RGS) and Policy Gradients with Parameter-based Exploration (PGPE). Both methods estimate updates using losses at perturbed parameter values. The resulting network is the centre model, which can be evaluated and verified through CTRAIN's common interfaces.

## Training Objective

Let `theta` denote the network parameters and `L_bound` the certified classification loss computed using the selected bounding method. Gaussian loss smoothing considers the expected loss under Gaussian parameter perturbations:

```text
L_GLS(theta) = E_noise[L_bound(theta + noise)]
```

The parameter `sigma` controls the standard deviation of the Gaussian distribution. Since computing this expectation exactly is intractable, RGS and PGPE approximate the update using a finite collection of perturbed networks. The perturbations act on the model parameters; the input region remains the one specified by `eps` and the data loader.

### Randomized Gradient Smoothing

RGS computes ordinary gradients of the loss at independently sampled parameter values and averages them to update the centre model. It therefore requires gradients through the selected bound computation. The `samples` argument controls the number of sampled networks per optimiser step and defaults to two.

RGS uses a fixed noise scale, which can optionally be reduced during training through `sigma_decay_milestones` and `sigma_decay_factor`. It is the first-order approximation studied in the paper; for discontinuous objectives, averaging sampled gradients need not give an unbiased gradient of the Gaussian-smoothed loss.

### Policy Gradients with Parameter-based Exploration

PGPE estimates updates from loss values at mirrored parameter perturbations, evaluating both `theta + noise` and `theta - noise`. It updates the centre parameters and learns a separate Gaussian standard deviation for each parameter, using an additional centre-loss evaluation as a baseline for the scale updates. Thus, its update does not require backpropagating through the certified loss.

For PGPE, `samples` must be a positive even integer, counting the perturbed loss evaluations in the mirrored pairs. The centre-loss baseline is evaluated in addition to these samples. The `sigma` argument provides the initial standard deviation and `sigma_lr` controls its learning rate. CTRAIN limits each scale change to 2% per update.

## Training with CTRAIN

In the following example, we train a CNN on CIFAR-10 using RGS with CROWN bounds, for robustness against ℓ∞ perturbations with radius ε = 2/255. First, we import the required functions and load the dataset using CTRAIN. The loader supplies the normalisation metadata and input limits required for bound propagation.

```python
from CTRAIN.data_loaders import load_cifar10
from CTRAIN.model_definitions import CNN_GLS
from CTRAIN.model_wrappers import GLSModelWrapper

input_shape = (3, 32, 32)
train_loader, test_loader = load_cifar10(batch_size=128, val_split=False)
model = CNN_GLS(in_shape=input_shape, depth=5, width=16)
```

`CNN_GLS` provides the architectures used in the GLS paper. Here, `depth=5` and `width=16` select CNN5, without BatchNorm or an embedded normalisation layer. Input normalisation is handled by the data loader.

### Configuring and Training the Wrapper

Now, we wrap the network with `GLSModelWrapper`. We choose two RGS samples per update, an initial noise scale of `1e-3`, and CROWN for the certified loss. The wrapper uses Adam by default; here, we explicitly set its learning rate to `5e-5`.

We configure one warm-up epoch on clean cross-entropy, followed by 80 ramp-up epochs during which the perturbation radius increases linearly to its target. Training then continues at the full radius. We reduce both the learning rate and the RGS noise scale at epochs 120 and 140, using separate decay factors for the two schedules.

```python
wrapped_model = GLSModelWrapper(
    model=model,
    input_shape=input_shape,
    eps=2 / 255,
    num_epochs=160,
    estimator="rgs",
    samples=2,
    sigma=1e-3,
    bound_method="CROWN",
    lr=5e-5,
    warm_up_epochs=1,
    ramp_up_epochs=80,
    lr_decay_kwargs={"milestones": (120, 140), "gamma": 0.2},
    sigma_decay_milestones=(120, 140),
    sigma_decay_factor=0.4,
    device="cuda",
)
wrapped_model.train_model(train_loader)
```

The `bound_method` option supports `"IBP"`, `"CROWN-IBP"` and `"CROWN"`, using CTRAIN's existing auto_LiRPA bound computations. CROWN-IBP uses IBP for intermediate bounds, while CROWN uses linear relaxations throughout. The default adaptive ReLU relaxation used by CROWN corresponds to DeepPoly.

### Using PGPE

To use PGPE, select `estimator="pgpe"`, choose an even `samples` value and configure the scale learning rate through `sigma_lr`. Omit `sigma_decay_milestones`, since PGPE learns its scales during optimisation. The main configuration differences are:

| Option | RGS | PGPE |
|---|---|---|
| `samples` | Number of independent gradient samples | Number of perturbed loss evaluations, in mirrored pairs |
| `sigma` | Fixed standard deviation, with optional scheduled decay | Initial standard deviation for learned parameterwise scales |
| `sigma_lr` | Unused | Learning rate for the scales |
| `sigma_decay_milestones` | Epochs at which the fixed scale is reduced | Must be empty |

The paper's PGPE experiments start from adversarially pretrained models. To fine-tune such a model in CTRAIN, load its native PyTorch weights before constructing the wrapper and set `shi_init=False` to preserve the initial weights.

### Evaluating the Model

Finally, we set the model to evaluation mode and assess its natural, certified and adversarial accuracy using the common evaluation interface:

```python
wrapped_model.eval()
std_acc, cert_acc, adv_acc = wrapped_model.evaluate(test_loader, eval_method="CROWN")
```

Evaluation uses the centre network with sound bounds over the full nominal input box. Complete verification is available through `wrapped_model.evaluate_complete`, using α,β-CROWN or NeuralSAT. The [evaluation guide](guides/evaluation.md) explains the available methods and interpretation of results.

## Training Behaviour and Configuration

Shi initialisation is enabled by default. Shi regularisation and L1 regularisation are optional and both default to zero. The trainer uses clean cross-entropy during warm-up and certified cross-entropy at positive perturbation radii, with the configured regularisers added to the objective. Gaussian parameter sampling applies during both phases.

Sampled parameters and buffers are restored between evaluations and on failure. For models with BatchNorm, running buffers update once per training batch at the centre model. Optional population recalibration is available through `population_bn`. If `gradient_clip` is set, RGS clips each sampled gradient before averaging, while PGPE clips the estimated centre gradient. Further shared options are described in the [training guide](guides/training.md).

Without a validation loader, `train_model` returns the final weights. Supplying `val_loader` enables selection by the centre model's certified validation accuracy after ramp-up, at the configured evaluation radius and using the selected bounding method. Trajectory checkpoints retain the centre and scale optimiser states, the Gaussian scales and the learning-rate scheduler. Recreate the same estimator and sample count when resuming; the [checkpoint guide](guides/checkpoints.md) explains saving and restoration.

## Hyperparameter Optimisation

GLS supports the shared [hyperparameter optimisation interface](guides/hpo.md), including multi-objective searches over natural and certified validation accuracy. The search space contains the shared training parameters and the initial Gaussian scale `sigma`, searched logarithmically over `[1e-5, 1e-2]` with default `1e-3`.

The estimator, sample count, bound method and remaining GLS options are preserved when the wrapper is recreated for each trial. Thus, choose these settings when constructing the wrapper and use separate validation data to investigate the remaining training parameters.

## Implementation and API Reference

CTRAIN implements the RGS and PGPE update procedures using its existing certified losses and bound computations. Parameter samples are evaluated sequentially within the trainer. When translating PGPE configurations from the [released implementation](https://github.com/stefanrzv2000/GLS-Cert-Training/blob/main/gls_wrapper.py), note that its `popsize` counts mirrored pairs; the corresponding CTRAIN setting is `samples=2 * popsize`.

The implemented training bounds are IBP, CROWN-IBP and CROWN. HBox and the paper's one-step optimised α-CROWN training variant are outside the current wrapper's scope. The example above demonstrates the library workflow; reproducing reported accuracies additionally requires matching the experiment's architecture, initialisation, preprocessing, training configuration and verification protocol.

The API reference documents the [GLS wrapper](api/model_wrappers/gls_model_wrapper.md), [trainer and optimiser](api/train/certified/gls.md), and the underlying [IBP](api/bound/ibp.md) and [CROWN/CROWN-IBP](api/bound/crown.md) bound computations.
