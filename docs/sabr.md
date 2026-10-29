# SABR

SABR, short for Small Adversarial Bounding Regions, combines adversarial search with interval bound propagation. Proposed by [Müller et al. (2023), *Certified Training: Small Boxes are All You Need*](https://arxiv.org/abs/2210.04871), the method uses PGD to identify an adversarial input and then computes IBP bounds over a smaller region around it. By propagating a carefully selected subset of the input region, SABR aims to reduce approximation errors and the associated over-regularisation of standard IBP training.

CTRAIN provides SABR through `SABRModelWrapper`, reusing the library's PGD attack, IBP bounds, training schedules and regularisers. The same subregion-selection procedure is also used by [STAPS](taps_staps.md#staps) to obtain feature bounds before its latent adversarial search.

## Training Objective

For an input and its label, SABR first constructs the nominal perturbation box, clipped to the feasible input limits. PGD searches for an adversarial example, which is then used to locate a smaller box within the nominal region. IBP propagates this selected box through the network, and the resulting lower classification-margin bounds are used to compute cross-entropy.

The `sabr_subselection_ratio` controls the size of the selected box. For nominal endpoints `x_L` and `x_U`, its half-width is computed coordinatewise:

```text
tau = sabr_subselection_ratio * (x_U - x_L) / 2
```

CTRAIN clamps the adversarial point into `[x_L + tau, x_U - tau]` to obtain the box centre. Thus, the selected region remains contained in the nominal box, including near image boundaries where coordinates have unequal or zero widths. The ratio must lie in `(0, 1]`; a value of `0.1` retains 10% of each nominal interval's width, while `1.0` selects the full nominal box.

With ordinary interval propagation, IBP bounds are sound for the selected subregion. However, a training loss over that subregion does not certify robustness throughout the full perturbation region. The trained model is assessed separately through the common evaluation interfaces.

## Training with CTRAIN

In the following example, we train the CNN7 architecture proposed by Shi et al. on CIFAR-10 for robustness against ℓ∞ perturbations with radius ε = 2/255. First, we import the required functions, load the dataset and construct the network.

```python
from CTRAIN.data_loaders import load_cifar10
from CTRAIN.model_definitions import CNN7_Shi
from CTRAIN.model_wrappers import SABRModelWrapper

input_shape = (3, 32, 32)
train_loader, test_loader = load_cifar10(batch_size=128, val_split=False)
model = CNN7_Shi(in_shape=input_shape)
```

Now, we wrap the network with `SABRModelWrapper`. We configure one warm-up epoch using clean cross-entropy, followed by 80 ramp-up epochs during which the perturbation radius gradually increases to its target. Training then continues at the full radius, using a subregion ratio of `0.1` and eight PGD steps to select each region.

```python
wrapped_model = SABRModelWrapper(
    model=model,
    input_shape=input_shape,
    eps=2 / 255,
    num_epochs=160,
    warm_up_epochs=1,
    ramp_up_epochs=80,
    lr_decay_kwargs={"milestones": (120, 140), "gamma": 0.2},
    sabr_subselection_ratio=0.1,
    pgd_steps=8,
    pgd_alpha=0.5,
    pgd_decay_milestones=(4, 7),
    relu_upper_retention=0.8,
    device="cuda",
)
wrapped_model.train_model(train_loader)
```

Here, `relu_upper_retention=0.8` additionally retains 80% of unstable ReLU upper bounds in the training IBP computation. This optional training surrogate is independent of the input-subregion ratio; its default is `1.0`, which leaves ReLU propagation unchanged. Retention below one is unsound even for the selected region and is not used for ordinary certification.

### Evaluating the Model

Finally, we set the model to evaluation mode and assess its natural, certified and adversarial accuracy. The following call uses CROWN for incomplete verification and PGD for adversarial evaluation:

```python
wrapped_model.eval()
std_acc, cert_acc, adv_acc = wrapped_model.evaluate(test_loader, eval_method="CROWN")
```

Evaluation uses the full nominal input box and fresh, sound bounds. Complete verification can be invoked through `wrapped_model.evaluate_complete`, using α,β-CROWN or NeuralSAT. See the [evaluation guide](guides/evaluation.md) for the available methods and solver settings.

## Training Behaviour and Configuration

The `pgd_steps`, `pgd_alpha` and `pgd_restarts` options configure the input attack. Step-size decay is controlled through `pgd_decay_milestones` and `pgd_alpha_decay_factor`. The optional `pgd_eps_factor` scales the PGD search range without widening the nominal box or the selected SABR region. At the lower-level loss and bound interfaces, explicit `pgd_ptb` endpoints override the search-range factor.

During training, PGD runs in evaluation mode, while the loss on the adversarial points and the subsequent bound computation use the caller's training mode. The trainer owns initialisation, schedules, Shi and L1 regularisation, checkpointing and model selection. Optional population BatchNorm recalibration is available through `population_bn`; further shared options are described in the [training guide](guides/training.md).

Without a validation loader, `train_model` returns the final weights. To select a model using validation data, load a separate validation split and pass it through `val_loader`. Selection uses the smallest SABR validation data loss after ramp-up at the fixed evaluation radius. Trajectory checkpoints retain the training and selection state; the [checkpoint guide](guides/checkpoints.md) explains saving and resumption.

## Hyperparameter Optimisation

SABR supports the shared [hyperparameter optimisation interface](guides/hpo.md), including multi-objective searches over natural and certified validation accuracy. In addition to the shared training parameters, its search space includes the subregion ratio, PGD step count, relative step size and search-range factor.

The subregion ratio is searched logarithmically over `[0.001, 0.5]`, using the configuration key `sabr:subselection_ratio`. Direct wrapper construction accepts the full interval `(0, 1]`. Reserve separate validation data for configuration search and use the test split to assess the selected model.

## Implementation and API Reference

The wrapper delegates to `sabr_train_model`; the loss and bound functions own the PGD search, subregion construction and IBP computations. This shared implementation is also used by STAPS. The example above demonstrates the library workflow; reproducing reported accuracies additionally requires matching the architecture, preprocessing, training configuration and verification protocol.

The API reference documents the [SABR wrapper](api/model_wrappers/sabr_model_wrapper.md), [trainer](api/train/certified/sabr.md), [loss](api/train/certified/losses/sabr.md) and [bound computations](api/bound/sabr.md).
