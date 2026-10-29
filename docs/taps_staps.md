# TAPS and STAPS

TAPS, short for Training via Adversarial Propagation through Subnetworks, combines IBP and adversarial training by splitting a network into a feature extractor and a classifier. Proposed by [Mao et al. (2023), *Connecting Certified and Adversarial Training*](https://proceedings.neurips.cc/paper_files/paper/2023/file/e8b0c97b34fdaf58b2f48f8cca85e76a-Paper-Conference.pdf), it propagates an input region through the feature extractor using IBP, then adversarially searches the resulting feature box to train the classifier. STAPS combines this approach with [SABR](sabr.md), using a smaller, adversarially selected input region for feature propagation.

CTRAIN provides the two methods through `TAPSModelWrapper` and `STAPSModelWrapper`. They share the latent search, gradient connector and loss computation, while differing in how the input region is selected and propagated.

## Training Objectives

### TAPS

TAPS first computes lower and upper feature bounds using IBP. Within this feature box, PGD searches for latent adversarial points that reduce the classification margins. A separate point is sought for each competing class, allowing different points to contribute to the estimated worst-case margins.

The gradient connector links these latent points to the feature-box endpoints, allowing the feature extractor and classifier to be trained jointly. CTRAIN combines cross-entropy from the latent margin estimates with an auxiliary IBP cross-entropy over the classifier's feature box, multiplying the two batch-mean losses. The `gradient_expansion_alpha` option scales the latent-loss contribution during backpropagation without changing the forward loss value.

### STAPS

STAPS first uses SABR to select a contained input subregion, then propagates that region through the feature extractor. It applies the same latent search and loss computation as TAPS to the resulting feature box. Consequently, STAPS exposes separate controls for the input PGD search used by SABR and the latent PGD search used by TAPS.

Both methods optimise training surrogates. Latent-search margin estimates are not sound certificates for the full input region; the trained model is verified separately using the ordinary evaluation interfaces.

## Training with CTRAIN

In the following example, we train CNN7 on CIFAR-10 for robustness against ℓ∞ perturbations with radius ε = 2/255. First, we import the required functions and load the dataset.

```python
from CTRAIN.data_loaders import load_cifar10
from CTRAIN.model_definitions import CNN7_Shi
from CTRAIN.model_wrappers import TAPSModelWrapper, STAPSModelWrapper

input_shape = (3, 32, 32)
train_loader, test_loader = load_cifar10(batch_size=128, val_split=False)
```

### Choosing the Network Split

The `block_sizes` argument specifies the numbers of layers in the feature extractor and classifier. It must contain two positive integers whose sum equals the number of layers in the model's sequential `.layers` definition. CTRAIN checks that this split corresponds to a sequential, single-input LiRPA graph.

For `CNN7_Shi`, the sequential definition contains 20 layers, including convolutions, BatchNorm, ReLU, Flatten and linear layers. The split `block_sizes=(15, 5)` places the convolutional layers in the feature extractor and the final five layers in the classifier. This is a valid example split; the appropriate choice depends on the architecture and experiment.

CTRAIN normalises inputs in the data loader. Reference models that include a normalisation layer therefore use different layer indices, which must be translated before specifying a split. When loading trained weights, use the architecture and split associated with that training configuration.

### Training with TAPS

Now, we construct a CNN7 model and wrap it with `TAPSModelWrapper`. We configure one warm-up epoch using clean cross-entropy and 80 ramp-up epochs using IBP training as the perturbation radius increases. Once the full training radius is reached, the trainer uses the TAPS objective.

```python
taps_model = TAPSModelWrapper(
    model=CNN7_Shi(in_shape=input_shape),
    input_shape=input_shape,
    eps=2 / 255,
    num_epochs=160,
    block_sizes=(15, 5),
    warm_up_epochs=1,
    ramp_up_epochs=80,
    lr_decay_kwargs={"milestones": (120, 140), "gamma": 0.2},
    pgd_steps=20,
    pgd_alpha=0.2,
    pgd_decay_steps=(14,),
    gradient_expansion_alpha=5,
    device="cuda",
)
taps_model.train_model(train_loader)
```

Here, `pgd_steps`, `pgd_alpha` and `pgd_restarts` configure the latent search. Its step-size schedule is controlled through `pgd_decay_steps` and `pgd_alpha_decay_factor`. Passing `pgd_alpha=None` uses the reference initial step-size rule `max(0.2, 2 / steps)` for positive step counts.

### Training with STAPS

To train a separate model with STAPS, we use `STAPSModelWrapper` and additionally configure SABR's subregion selection. Warm-up again uses clean cross-entropy, while ramp-up uses SABR. At the full training radius, the trainer switches to the combined STAPS objective.

```python
staps_model = STAPSModelWrapper(
    model=CNN7_Shi(in_shape=input_shape),
    input_shape=input_shape,
    eps=2 / 255,
    num_epochs=160,
    block_sizes=(15, 5),
    warm_up_epochs=1,
    ramp_up_epochs=80,
    lr_decay_kwargs={"milestones": (120, 140), "gamma": 0.2},
    pgd_steps=20,
    pgd_alpha=0.2,
    pgd_decay_steps=(14,),
    gradient_expansion_alpha=2,
    sabr_subselection_ratio=0.1,
    sabr_pgd_steps=10,
    sabr_pgd_alpha=0.2,
    sabr_pgd_decay_milestones=(),
    relu_upper_retention=0.8,
    device="cuda",
)
staps_model.train_model(train_loader)
```

The two searches have distinct parameter groups:

| Search | Step count | Relative step size | Restarts | Step-size decay milestones |
|---|---|---|---|---|
| Latent classifier search | `pgd_steps` | `pgd_alpha` | `pgd_restarts` | `pgd_decay_steps` |
| SABR input search | `sabr_pgd_steps` | `sabr_pgd_alpha` | `sabr_pgd_restarts` | `sabr_pgd_decay_milestones` |

The `sabr_subselection_ratio` determines the width of the selected input region. In this example, `relu_upper_retention=0.8` separately retains 80% of unstable ReLU upper bounds during training. It corresponds to shrinkage `0.2` in the original TAPS convention. Its default is `1.0`; retention below one supplies an additional unsound training surrogate.

### Evaluating the Model

Finally, we set the trained model to evaluation mode and assess its natural, certified and adversarial accuracy. For example, the following call evaluates the STAPS model using CROWN and PGD:

```python
staps_model.eval()
std_acc, cert_acc, adv_acc = staps_model.evaluate(test_loader, eval_method="CROWN")
```

The same interface is available on `taps_model`. Certification uses fresh, sound bounds on the full model and nominal input region. Complete verification is available through `evaluate_complete`, using α,β-CROWN or NeuralSAT. See the [evaluation guide](guides/evaluation.md) for the available methods and solver settings.

## Training Behaviour and Model Selection

The `gradient_link_thresh` and `gradient_link_tol` options control the gradient connector between latent points and feature bounds. The `pgd_eps_factor` optionally scales the search range around the latent box midpoint. In STAPS it also scales the input PGD search, while the nominal bound region and contained SABR subregion remain unchanged.

CTRAIN performs adversarial searches in evaluation mode, then evaluates the adversarial points and their losses in the caller's training mode. This is CTRAIN's chosen BatchNorm policy and differs from the frozen-statistics training policy in the original TAPS implementation. Optional population recalibration is available through `population_bn`; further shared options are described in the [training guide](guides/training.md).

Without a validation loader, both wrappers return the final weights. Supplying `val_loader` enables selection after ramp-up by latent-margin proxy accuracy at the fixed evaluation radius. This proxy records whether the latent search finds a non-positive margin for any competing class; it must not be reported as certified accuracy. Use separate validation data for selection and the ordinary evaluation methods for final reporting.

Trajectory checkpoints retain optimiser, scheduler and selection state separately from the selected inference weights. The [checkpoint guide](guides/checkpoints.md) explains saving and restoration.

## Hyperparameter Optimisation

Both wrappers support the shared [hyperparameter optimisation interface](guides/hpo.md), including multi-objective searches over natural and certified validation accuracy. Their search spaces include the shared training hyperparameters, latent PGD settings, gradient-expansion factor and a split point used to choose a ReLU boundary. STAPS additionally searches SABR's subregion and input-attack settings.

The split chosen for an HPO trial can therefore differ from the one supplied for an ordinary training run. Retain the selected configuration with its weights when comparing or continuing experiments.

## Implementation and API Reference

The wrappers configure the corresponding trainers, which own the warm-up, ramp-up, method-specific loss and model-selection phases. Bounds are obtained through auto_LiRPA, with small adapters for the graph split, latent search and gradient connector.

Paper tables and released scripts differ in some schedules, splits and TinyImageNet input sizes. The examples above demonstrate the library interfaces and a valid CNN7 split; reproducing reported accuracies requires matching the chosen source configuration, preprocessing, BatchNorm policy and verification protocol.

The API reference documents the [TAPS wrapper](api/model_wrappers/taps_model_wrapper.md), [STAPS wrapper](api/model_wrappers/staps_model_wrapper.md), [TAPS trainer](api/train/certified/taps.md), [STAPS trainer](api/train/certified/staps.md), [loss](api/train/certified/losses/taps.md) and [bound computations](api/bound/taps.md).
