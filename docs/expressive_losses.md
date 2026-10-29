# Expressive Losses

The family of expressive losses combines adversarial and certified training through convex combinations. Proposed by [De Palma et al. (2024), *Expressive Losses for Verified Robustness via Convex Combinations*](https://arxiv.org/abs/2305.13991), these methods use a single coefficient to control the contribution of adversarial examples and IBP bounds to the training objective. This allows users to explore different trade-offs between natural accuracy and certified robustness.

CTRAIN provides three representatives of this family: MTL-IBP, CC-IBP and Exp-IBP. They share implementations of PGD, IBP bounding and the training procedure, while differing in how they combine the adversarial and certified components. Each method is available through a separate model wrapper with the common training, evaluation and hyperparameter optimisation interface.

## Training Objectives

For each training input, the methods obtain an adversarial example using PGD and lower bounds on the classification margins using IBP. A classification margin is the difference between the logit of the correct class and that of a competing class. Positive margins for every competing class indicate correct classification.

Let `L_adv` denote the batch-mean cross-entropy on the adversarial examples and `L_IBP` the corresponding loss computed from IBP margin bounds. The three methods combine these quantities as follows.

### MTL-IBP

MTL-IBP takes a weighted arithmetic mean of the two losses:

```text
L_MTL = alpha * L_IBP + (1 - alpha) * L_adv
```

Thus, its coefficient directly determines the relative weight of the adversarial and IBP classification losses in the objective.

### CC-IBP

CC-IBP forms the convex combination before applying cross-entropy. Let `a` denote the margins at the adversarial examples and `b` the IBP lower margin bounds. The mixed margins are:

```text
mixed_margins = alpha * b + (1 - alpha) * a
```

Cross-entropy is then evaluated on these mixed margins and averaged over the batch. The combination therefore takes place at the level of classification margins, coupling the two components before computing the loss. [CC-Dist](cc_dist.md) extends this objective with feature distillation from an adversarially trained teacher.

### Exp-IBP

Exp-IBP combines the losses geometrically, which is equivalent to taking a convex combination of their logarithms:

```text
L_Exp = L_IBP ** alpha * L_adv ** (1 - alpha)
```

CTRAIN combines the two batch-mean losses, following the authors' implementation. It evaluates this expression in log space, handling zero losses to keep the computation and its gradients well defined.

For all three objectives, `alpha=0` recovers the adversarial classification loss and `alpha=1` recovers the IBP classification loss. Configured regularisation still applies at these endpoints. The coefficient must lie in `[0, 1]` and defaults to `0.5`. Since the combination differs between methods, the same coefficient need not yield the same natural–certified accuracy trade-off.

## Training with CTRAIN

In the following example, we train the CNN7 architecture proposed by Shi et al. on CIFAR-10 using MTL-IBP, for robustness against ℓ∞ perturbations with radius ε = 2/255. First, we import the required functions, load the dataset and define the network.

```python
from CTRAIN.data_loaders import load_cifar10
from CTRAIN.model_definitions import CNN7_Shi
from CTRAIN.model_wrappers import MTLIBPModelWrapper

input_shape = (3, 32, 32)
train_loader, test_loader = load_cifar10(batch_size=128, val_split=False)
model = CNN7_Shi(in_shape=input_shape)
```

Now, we wrap the network with `MTLIBPModelWrapper` and configure the training process. We use one warm-up epoch on clean cross-entropy, followed by 80 ramp-up epochs during which the perturbation radius gradually increases to its target. The remaining epochs train at the full radius. Here, `mtl_ibp_alpha=0.004` gives the adversarial loss most of the weight, while retaining an IBP component.

```python
wrapped_model = MTLIBPModelWrapper(
    model=model,
    input_shape=input_shape,
    eps=2 / 255,
    num_epochs=160,
    mtl_ibp_alpha=0.004,
    warm_up_epochs=1,
    ramp_up_epochs=80,
    lr_decay_kwargs={"milestones": (120, 140), "gamma": 0.2},
    pgd_steps=8,
    pgd_alpha=0.25,
    pgd_eps_factor=2.1,
    l1_reg_weight=3e-6,
    device="cuda",
)
wrapped_model.train_model(train_loader)
```

To use CC-IBP or Exp-IBP, import the corresponding wrapper and replace the coefficient argument as shown below. The other training options have the same meaning across the three wrappers.

| Method | Model wrapper | Coefficient argument |
|---|---|---|
| MTL-IBP | [`MTLIBPModelWrapper`](api/model_wrappers/mtl_ibp_model_wrapper.md) | `mtl_ibp_alpha` |
| CC-IBP | [`CCIBPModelWrapper`](api/model_wrappers/cc_ibp_model_wrapper.md) | `cc_ibp_alpha` |
| Exp-IBP | [`ExpIBPModelWrapper`](api/model_wrappers/exp_ibp_model_wrapper.md) | `exp_ibp_alpha` |

The coefficient should be chosen for the method and task of interest. The [hyperparameter optimisation interface](#hyperparameter-optimisation) can be used to investigate its effect alongside the other training parameters.

### Evaluating the Model

Finally, we set the model to evaluation mode and assess its natural, certified and adversarial accuracy. The following call uses CROWN for incomplete verification and PGD for adversarial evaluation:

```python
wrapped_model.eval()
std_acc, cert_acc, adv_acc = wrapped_model.evaluate(test_loader, eval_method="CROWN")
```

Complete verification is available through `wrapped_model.evaluate_complete`, using α,β-CROWN or NeuralSAT. Certification evaluates the model over the full nominal input box using sound bounds. The mixed training objective itself is a surrogate for learning robustness; certification is performed separately through the [evaluation interfaces](guides/evaluation.md).

## Training Behaviour and Configuration

The three wrappers delegate to the shared expressive trainer, which owns initialisation, schedules, Shi and L1 regularisation, checkpointing and model selection. During training, PGD searches run in evaluation mode, while the loss on the resulting adversarial examples is computed in training mode. This preserves CTRAIN's chosen BatchNorm policy across the three methods.

The `pgd_steps`, `pgd_alpha` and `pgd_restarts` options configure the training attack. In the example, `pgd_eps_factor=2.1` widens the PGD search region without changing the nominal IBP region. Evaluation attacks have their own settings. The shared options also include optional population BatchNorm recalibration through `population_bn` and training-only ReLU shrinkage through `relu_upper_retention` known as ReLU shrinking employed in SABR. See the [training guide](guides/training.md) for these options and their defaults.

Without a validation loader, `train_model` returns the final weights. To select a model using validation data, load a separate validation split and pass it through `val_loader`. Selection then uses the smallest robust validation data loss after ramp-up, computed with the chosen expressive objective at the evaluation radius. Trajectory checkpoints retain the model, optimiser, learning-rate scheduler and selection state; the [checkpoint guide](guides/checkpoints.md) explains saving and resumption.

## Hyperparameter Optimisation

All three wrappers support the common `hpo` method, which searches for Pareto-optimal trade-offs between natural and certified validation accuracy. Reserve separate validation data for this search and use the test split to assess the selected model. The [HPO guide](guides/hpo.md) explains the available samplers, budgets and model-selection workflow.

The method-specific search space includes the mixing coefficient, PGD step count, relative step size and search-range factor, alongside the shared training hyperparameters. CTRAIN searches the coefficient logarithmically over `[0.001, 0.5]`; direct wrapper construction accepts the full interval `[0, 1]`. Configuration keys use the method name as a prefix, such as `mtl_ibp:mtl_ibp_alpha`, `cc_ibp:cc_ibp_alpha` and `exp_ibp:exp_ibp_alpha`.

## Implementation and API Reference

The objectives follow the Expressive Losses paper and the [authors' training implementation](https://github.com/alessandrodepalma/expressive-losses/blob/main/train.py). CTRAIN provides separate wrappers and loss functions for the three methods, reusing the common attack and bound computations. Lower-level loss calls that request bounds return the underlying IBP margin bounds; CC-IBP's mixed margins belong to its training objective.

The example above demonstrates the library workflow. Reproducing reported accuracies additionally requires matching the experiment's architecture, preprocessing, data splits, training configuration and verification protocol.

The API reference documents the [shared wrapper](api/model_wrappers/expressive_ibp_model_wrapper.md), [expressive trainer](api/train/certified/expressive_ibp.md) and [loss functions](api/train/certified/losses/convex_combinations.md). Method-specific training entry points are documented for [MTL-IBP](api/train/certified/mtl_ibp.md), [CC-IBP](api/train/certified/cc_ibp.md) and [Exp-IBP](api/train/certified/exp_ibp.md).
