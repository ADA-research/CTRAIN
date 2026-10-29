# IBP-R

IBP-R augments adversarial training with an IBP-based regulariser that encourages tighter approximations of the network's ReLU activations. Proposed by [De Palma et al. (2022), *IBP Regularization for Verified Adversarial Robustness via Branch-and-Bound*](https://arxiv.org/abs/2206.14772), the method combines adversarial attacks over enlarged input regions with a penalty on the areas of convex ReLU relaxations. This encourages networks that retain the predictive performance of adversarial training while facilitating subsequent verification.

CTRAIN provides IBP-R through `IBPRModelWrapper`, reusing the library's PGD attack and auto_LiRPA interval propagation. The wrapper configures the training radius, attack and regularisation parameters, while the trainer owns the schedules, optimisation and model selection.

## Training Objective

IBP-R uses PGD to obtain adversarial examples and IBP to compute pre-activation bounds at each ReLU. When a ReLU's interval crosses zero, its convex relaxation introduces a triangular approximation region. For lower and upper pre-activation bounds, the area is:

```text
area = relu(-lower) * relu(upper) / 2
```

Stable ReLUs contribute zero area. CTRAIN sums these areas over all neurons and ReLU layers, then averages over the batch. The coefficient `rho` weights this penalty in the objective. Since the area already includes the factor `1 / 2`, a paper-table entry of `rho / 2 = 1e-4` corresponds to `rho=2e-4` in the wrapper.

During ramp-up, the trainer increases both the perturbation radius and an adversarial mixing weight `kappa` linearly from zero to their targets. The classification and area terms are combined as follows:

```text
L_train = (1 - kappa) * L_clean + kappa * (L_adv + rho * R)
```

Here, `L_clean` and `L_adv` are the clean and adversarial cross-entropy losses, and `R` is the batch-mean ReLU area penalty. Once `kappa` reaches one, training uses adversarial cross-entropy plus the area penalty. Configured L1 and Shi regularisation are added separately by the trainer.

### Regularisation Masking

With `regularisation_mask=True`, the area penalty includes only samples whose adversarial predictions are correct. The denominator remains the original batch size, so masking removes contributions without renormalising over the retained samples. The classification loss still uses all samples. Masking defaults to `False`.

## Training with CTRAIN

In the following example, we train the CNN7 architecture proposed by Shi et al. on CIFAR-10 for robustness against ℓ∞ perturbations with radius ε = 2/255. First, we import the required functions, load the dataset and construct the model.

```python
from CTRAIN.data_loaders import load_cifar10
from CTRAIN.model_definitions import CNN7_Shi
from CTRAIN.model_wrappers import IBPRModelWrapper

input_shape = (3, 32, 32)
train_loader, test_loader = load_cifar10(batch_size=100, val_split=False)
model = CNN7_Shi(in_shape=input_shape)
```

We use CNN7 here to demonstrate the common wrapper interface. The original study used smaller five-layer and four-layer networks at 2/255 and 8/255, respectively.

Now, we wrap the model with `IBPRModelWrapper`. The `train_eps_factor` corresponds to the paper's radius multiplier α: it enlarges both the training IBP region and the input-attack region. Here, `train_eps_factor=2.1` gives a target training radius of 4.2/255, while evaluation continues to use the nominal radius of 2/255.

We configure 800 epochs, including 600 ramp-up epochs. The wrapper uses SGD by default, with an initial learning rate of `0.01` and exponential decay by a factor of `0.95`. In this example, we delay decay until after ramp-up and disable CTRAIN's additional Shi initialisation and regularisation to retain the original IBP-R objective.

```python
wrapped_model = IBPRModelWrapper(
    model=model,
    input_shape=input_shape,
    eps=2 / 255,
    num_epochs=800,
    warm_up_epochs=0,
    ramp_up_epochs=600,
    train_eps_factor=2.1,
    rho=2e-4,
    regularisation_mask=False,
    lr=0.01,
    lr_decay_kwargs={"gamma": 0.95},
    lr_decay_after_ramp_up=True,
    l1_reg_weight=2e-5,
    pgd_steps=8,
    pgd_alpha=0.25,
    shi_init=False,
    shi_reg_weight=0,
    device="cuda",
)
wrapped_model.train_model(train_loader)
```

The loader supplies the normalisation metadata and input limits, so radii are specified in raw pixel units and the resulting regions are clipped to feasible inputs. Data loading and custom preprocessing are described in the [data and models guide](guides/data_and_models.md).

### Evaluating the Model

Finally, we set the model to evaluation mode and assess its natural, certified and adversarial accuracy. The following call uses CROWN for incomplete verification and PGD for adversarial evaluation at the original `eps`:

```python
wrapped_model.eval()
std_acc, cert_acc, adv_acc = wrapped_model.evaluate(test_loader, eval_method="CROWN")
```

Complete verification is available through `wrapped_model.evaluate_complete`, using α,β-CROWN or NeuralSAT. The area penalty encourages verifiability during training; certificates are obtained separately over the full nominal input region. See the [evaluation guide](guides/evaluation.md) for the available methods and solver settings.

## Training Behaviour and Configuration

The `train_eps_factor` must be finite and at least one. The separate `pgd_eps_factor` optionally scales only the attack search range, leaving the training IBP region unchanged. The `pgd_steps`, `pgd_alpha` and `pgd_restarts` arguments configure the attack; step-size decay is controlled through `pgd_decay_milestones` and `pgd_alpha_decay_factor`.

CTRAIN runs attacks in evaluation mode and evaluates the resulting adversarial points and bounds in the caller's training mode. PGD step sizes use CTRAIN's maximum clipped half-width scaling, and the area penalty is computed from fresh IBP intervals for the current batch.

By default, CTRAIN enables Shi initialisation and adds Shi regularisation during warm-up and ramp-up with `shi_reg_weight=0.5` and `shi_reg_decay=True`. The example disables these additions explicitly. Similarly, `lr_decay_after_ramp_up` defaults to `False`, so delayed decay must be requested when following the paper's schedule. Gradient clipping defaults to 10.

Optional population BatchNorm recalibration is available through `population_bn`. The `relu_upper_retention` option defaults to `1.0`; values below one produce an unsound training surrogate and are not used for ordinary certification. Further shared settings are described in the [training guide](guides/training.md).

Without a validation loader, `train_model` returns the final weights. Supplying `val_loader` enables selection by adversarial validation accuracy after ramp-up, at the original evaluation radius. This is an empirical selection metric. The trainer recalibrates BatchNorm, when requested, before selection and checkpointing. The [checkpoint guide](guides/checkpoints.md) explains trajectory saving and resumption, including the distinction from selected inference weights.

### Translating Published Configurations

Figure 5 in Appendix F.1 of the [paper's second version](https://arxiv.org/html/2206.14772v2) reports separate configurations for the two CIFAR-10 radii and for masked and unmasked regularisation. The corresponding CTRAIN settings are:

| Radius | Batch size | `train_eps_factor` | `rho` | `l1_reg_weight` | `regularisation_mask` |
|---|---|---|---|---|---|
| 2/255 | 100 | 2.1 | `2e-4` | `2e-5` | `False` or `True` |
| 8/255, unmasked | 150 | 1.7 | `0.01` | `1e-5` | `False` |
| 8/255, masked | 150 | 1.6 | `0.02` | `1e-5` | `True` |

The [authors' released README](https://github.com/alessandrodepalma/ibpr) contains a different 8/255 command: it enables masking but uses a radius factor of `1.7` and `rho / 2 = 0.005`, corresponding to `rho=0.01`. This is a distinct configuration from the masked Figure 5 setting and should be kept separate when comparing experiments.

## Hyperparameter Optimisation

IBP-R supports the shared [hyperparameter optimisation interface](guides/hpo.md), including multi-objective searches over natural and certified validation accuracy. Its search space includes the shared training parameters, `rho`, regularisation masking and PGD settings. The training-radius factor is searched over `[1, 3]`, and `rho` logarithmically over `[1e-6, 0.1]`, using the configuration key `ibp_r:rho`.

SGD is the default optimiser for this search, with Adam, AdamW and RAdam also available. HPO trials use CTRAIN's shared milestone-based learning-rate schedule; the exponential schedule in the example applies to direct wrapper training. Fixed BatchNorm, ReLU-retention, Shi-initialisation and PGD-decay options are retained when recreating trial wrappers. Reserve separate validation data for the search and use the test split to assess the selected model.

## Implementation and API Reference

CTRAIN implements equation (6) and the optional regularisation mask from Section 3 of the paper using PyTorch and auto_LiRPA. The authors' implementation uses JAX. When comparing results, account for differences in attack bookkeeping, step-size scaling, BatchNorm policy, initialisation and L1 regularisation; CTRAIN's L1 helper regularises linear and convolutional weights.

The paper also introduces UPB branching and uses the OVAL verification framework. These are outside the training wrapper's scope; post-training verification uses CTRAIN's existing backends. The example above demonstrates the library workflow, and a matching configuration alone does not establish reproduction of reported accuracies.

The API reference documents the [IBP-R wrapper](api/model_wrappers/ibp_r_model_wrapper.md), [trainer](api/train/certified/ibp_r.md), [loss and area penalty](api/train/certified/losses/ibp_r.md) and underlying [IBP bounds](api/bound/ibp.md).
