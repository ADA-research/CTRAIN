# Interval Bound Propagation

Interval Bound Propagation (IBP) is a computationally efficient approach to certified training. It propagates lower and upper activation bounds through a network using interval arithmetic, obtaining a sound enclosure of the reachable outputs for a given input region. These bounds can then be used to construct a certified training loss, as demonstrated by [Gowal et al. (2018)](https://arxiv.org/abs/1810.12715).

CTRAIN provides IBP training through `ShiIBPModelWrapper`, incorporating the initialisation and regularisation improvements proposed by [Shi et al. (2021), *Fast Certified Robust Training with Short Warmup*](https://proceedings.neurips.cc/paper/2021/hash/988f9153ac4fd966ea302dd9ab9bae15-Abstract.html). These improvements address excessively wide initial bounds and imbalanced ReLU activation states, helping stabilise certified training.

## Training Objective

For each input, CTRAIN constructs a perturbation box and clips it to the feasible input limits. IBP propagates this region through the network to compute lower bounds on the classification margins, which are the differences between the correct-class logit and each competing logit.

Cross-entropy evaluated from these lower margin bounds gives an upper bound on the worst-case classification loss over the input box. If every lower margin bound is positive, the network correctly classifies the input throughout that region. A non-positive bound leaves the property unresolved by IBP; tighter verification methods may still establish robustness.

At positive perturbation radii, the trainer supports a scheduled combination of clean and IBP losses:

```text
L_train = kappa * L_clean + (1 - kappa) * L_IBP
```

The `start_kappa` and `end_kappa` arguments specify the clean-loss weights at the start and end of ramp-up. Both default to zero, giving the IBP classification objective during positive-radius training. At zero radius, the trainer uses clean cross-entropy. Configured Shi and L1 regularisers are added separately.

## Training with CTRAIN

In the following example, we train CNN7 on CIFAR-10 for robustness against ℓ∞ perturbations with radius ε = 2/255. First, we import the required functions, load the dataset and construct the model.

```python
from CTRAIN.data_loaders import load_cifar10
from CTRAIN.model_definitions import CNN7_Shi
from CTRAIN.model_wrappers import ShiIBPModelWrapper

input_shape = (3, 32, 32)
train_loader, test_loader = load_cifar10(batch_size=128, val_split=False)
model = CNN7_Shi(in_shape=input_shape)
```

The loader supplies normalisation metadata and input limits. Thus, the wrapper accepts `eps` in raw pixel units and constructs the appropriate regions for the normalised inputs. See the [data and models guide](guides/data_and_models.md) for custom data-loading workflows.

### Configuring and Training the Wrapper

Now, we wrap the network with `ShiIBPModelWrapper`. We configure one warm-up epoch using clean cross-entropy, followed by 80 ramp-up epochs during which the perturbation radius gradually increases to its target. Training then continues at the full radius. The learning rate is reduced by a factor of 0.2 at epochs 120 and 140.

```python
wrapped_model = ShiIBPModelWrapper(
    model=model,
    input_shape=input_shape,
    eps=2 / 255,
    num_epochs=160,
    warm_up_epochs=1,
    ramp_up_epochs=80,
    lr_decay_kwargs={"milestones": (120, 140), "gamma": 0.2},
    device="cuda",
)
wrapped_model.train_model(train_loader)
```

The wrapper delegates to the library trainer, which applies Shi initialisation when starting from epoch zero and manages the schedules, regularisation, optimisation and checkpoints. The underlying PyTorch architecture remains available through `wrapped_model.original_model`.

### Evaluating the Model

Finally, we set the model to evaluation mode and assess its natural, certified and adversarial accuracy. The following call uses IBP for certification and PGD for adversarial evaluation:

```python
wrapped_model.eval()
std_acc, cert_acc, adv_acc = wrapped_model.evaluate(test_loader, eval_method="IBP")
```

Other supported evaluation methods include CROWN-IBP, CROWN and α-CROWN. Complete verification through [α,β-CROWN or NeuralSAT](complete_verification.md) is available via `evaluate_complete`. These choices concern post-training evaluation and can be selected independently of the training method.

## Training Behaviour and Configuration

The `shi_reg_weight` option controls the regularisers that encourage bound tightness and balanced ReLU activation states during warm-up and ramp-up. It defaults to `0.5`; `shi_reg_decay=True` reduces their contribution as the perturbation radius grows. L1 regularisation is controlled separately through `l1_reg_weight`, which defaults to `1e-6`.

The `train_eps_factor` scales the target training radius, while evaluation continues to use the nominal `eps`. Optional population BatchNorm recalibration is available through `population_bn`, which defaults to `False`. When enabled, recalibration uses training data after each epoch, before validation selection and checkpointing. Further shared options are described in the [training guide](guides/training.md).

Without a validation loader, `train_model` returns the final weights. Supplying `val_loader` enables selection after ramp-up by the smallest configured validation data loss at the evaluation radius. With the default zero clean-loss weight, this is the IBP validation loss. The [checkpoint guide](guides/checkpoints.md) explains saving and resumption, including the distinction between selected inference weights and training trajectories.

## Hyperparameter Optimisation

The wrapper supports the shared [hyperparameter optimisation interface](guides/hpo.md), including multi-objective searches over natural and certified validation accuracy. Its configuration space includes the training schedules, optimiser settings, regularisation weights, training-radius factor and clean-loss mixing coefficients.

Reserve separate validation data for the search and use the test split to assess the selected model. The [hyperparameter optimisation tutorial](examples/hyperparameter_optimisation.ipynb) demonstrates how to obtain and select models from a Pareto front.

## Implementation and API Reference

The wrapper delegates to `shi_train_model`, while the IBP loss and bound functions own the margin and interval computations. Bound propagation uses auto_LiRPA. [CROWN-IBP](crown_ibp.md) extends the training workflow with a backward linear-relaxation bound and has its own method page.

The example above demonstrates the library workflow; reproducing published accuracies additionally requires matching the architecture, preprocessing, schedules, regularisation and evaluation protocol.

The API reference documents the [Shi-IBP wrapper](api/model_wrappers/shi_ibp_model_wrapper.md), [trainer](api/train/certified/ibp.md), [loss](api/train/certified/losses/ibp.md), [IBP bounds](api/bound/ibp.md), [initialisation](api/train/certified/initialisation/shi.md) and [Shi regularisers](api/train/certified/regularisers/shi.md).
