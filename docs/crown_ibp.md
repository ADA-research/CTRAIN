# CROWN-IBP

CROWN-IBP combines efficient interval propagation with tighter backward linear-relaxation bounds. Proposed by [Zhang et al. (2020), *Towards Stable and Efficient Training of Verifiably Robust Neural Networks*](https://arxiv.org/abs/1906.06316), it uses IBP to obtain intermediate activation bounds and CROWN to bound the classification margins. This provides a way to stabilise the early stages of training, when IBP bounds can be particularly loose.

CTRAIN provides CROWN-IBP through `CrownIBPModelWrapper`. It shares the scheduling, initialisation, regularisation and evaluation infrastructure of [Shi-style IBP training](ibp.md), adding controls for the mixture of bound estimates and optional loss fusion.

## Training Objective

CROWN-IBP first propagates the input box forward using IBP to obtain intermediate bounds. These intervals constrain a backward CROWN pass that computes lower bounds on the differences between the correct-class logit and competing logits.

In the explicit-margin path, selected through `loss_fusion=False`, CTRAIN interpolates between the IBP and CROWN-IBP lower margin bounds:

```text
b_mixed = beta * b_CROWN_IBP + (1 - beta) * b_IBP
```

Cross-entropy is then computed from the mixed margins. The `start_beta` and `end_beta` arguments control the initial and final CROWN-IBP contribution, defaulting to `1` and `0`. Thus, the default schedule gradually shifts from CROWN-IBP towards IBP as the perturbation radius increases.

The trainer separately supports a clean-loss mixing coefficient:

```text
L_train = kappa * L_clean + (1 - kappa) * L_certified
```

The `start_kappa` and `end_kappa` arguments both default to zero, so positive-radius training uses the certified classification objective. At zero radius, training uses clean cross-entropy. Shi and L1 regularisation are configured separately.

## Training with CTRAIN

In the following example, we train CNN7 on CIFAR-10 for robustness against ℓ∞ perturbations with radius ε = 2/255. First, we import the required functions, load the dataset and construct the network.

```python
from CTRAIN.data_loaders import load_cifar10
from CTRAIN.model_definitions import CNN7_Shi
from CTRAIN.model_wrappers import CrownIBPModelWrapper

input_shape = (3, 32, 32)
train_loader, test_loader = load_cifar10(batch_size=128, val_split=False)
model = CNN7_Shi(in_shape=input_shape)
```

Now, we configure the wrapper with one warm-up epoch and 80 ramp-up epochs. During ramp-up, the perturbation radius increases and the bound mixture transitions from its initial to final coefficient. Here, we explicitly select the margin-based loss computation described above.

```python
wrapped_model = CrownIBPModelWrapper(
    model=model,
    input_shape=input_shape,
    eps=2 / 255,
    num_epochs=160,
    warm_up_epochs=1,
    ramp_up_epochs=80,
    start_beta=1.0,
    end_beta=0.0,
    loss_fusion=False,
    lr_decay_kwargs={"milestones": (120, 140), "gamma": 0.2},
    device="cuda",
)
wrapped_model.train_model(train_loader)
```

The loader supplies input limits and normalisation metadata, so the radius is specified in raw pixel units. The wrapper delegates the epoch loop and schedules to the library trainer; changing the training objective does not require a separate reproduction script.

### Evaluating the Model

Finally, we set the model to evaluation mode and assess its natural, certified and adversarial accuracy. This call uses CROWN-IBP for certification and PGD for adversarial evaluation:

```python
wrapped_model.eval()
std_acc, cert_acc, adv_acc = wrapped_model.evaluate(
    test_loader, eval_method="CROWN-IBP"
)
```

Other incomplete methods can be selected through `eval_method`, and [complete verification](complete_verification.md) is available through `evaluate_complete`. Evaluation uses the ordinary model and the full nominal input region, independently of the training bound mixture or loss-fusion graph.

## Loss Fusion

The wrapper defaults to `loss_fusion=True`. In this mode, auto_LiRPA includes the cross-entropy computation in a separate bounded graph, allowing the trainer to bound the fused loss directly and reduce the memory cost of the computation. The trainer supplies the current inputs and labels to this graph before computing its bounds.

The fused graph has its own BatchNorm buffers. CTRAIN synchronises the relevant buffers with the ordinary model and supports population recalibration before selection and checkpointing. Its optimiser and learning-rate scheduler operate on the fused training parameters.

The fused training path currently reports an unavailable (`NaN`) certified-accuracy statistic because it does not return the per-class margin bounds used for that metric. This statistic should be obtained separately through the ordinary evaluation API. The explicit-margin path in the example makes the bound mixture directly accessible without changing the evaluation interface.

## Training Behaviour and Configuration

The trainer applies Shi initialisation when starting from epoch zero and supports the shared Shi and L1 regularisers. The `shi_reg_weight` and `shi_reg_decay` options control warm-up/ramp-up regularisation, while `l1_reg_weight` controls the L1 contribution. These additions should be considered when comparing a configuration with the original CROWN-IBP experiments.

The `train_eps_factor` scales the target training radius, while evaluation continues to use the nominal `eps`. Optional population BatchNorm recalibration is available through `population_bn`. Both the beta and kappa endpoints must lie in `[0, 1]`. Further shared options are described in the [training guide](guides/training.md).

Without a validation loader, `train_model` returns the final weights. Supplying `val_loader` enables selection after ramp-up by the smallest configured validation data loss at the fixed evaluation radius, using the selected bound mixture and loss-fusion setting. Trajectory checkpoints retain the training and selection state; the [checkpoint guide](guides/checkpoints.md) explains saving and resumption.

## Hyperparameter Optimisation

CROWN-IBP supports the shared [hyperparameter optimisation interface](guides/hpo.md), including multi-objective searches over natural and certified validation accuracy. Its search space includes the shared training parameters, clean-loss coefficients and final beta value, with the initial beta fixed to one. The chosen loss-fusion and population BatchNorm settings are retained when constructing trial wrappers.

Reserve separate validation data for the search and evaluate the selected model on the test split. The [hyperparameter optimisation tutorial](examples/hyperparameter_optimisation.ipynb) demonstrates this workflow.

## Implementation and API Reference

`CrownIBPModelWrapper` configures `crown_ibp_train_model`, while the loss function obtains the IBP and CROWN-IBP bounds through auto_LiRPA. The example demonstrates the library workflow; source-specific architectures, schedules, loss-fusion behaviour and verification protocols must be matched before drawing reproduction conclusions.

The API reference documents the [CROWN-IBP wrapper](api/model_wrappers/crown_ibp_model_wrapper.md), [trainer](api/train/certified/crown_ibp.md), [loss](api/train/certified/losses/crown_ibp.md) and [bound computations](api/bound/crown.md).
