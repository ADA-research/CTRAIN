# Set-Based Training

Set-based training optimises both the predictions and the size of a network's reachable output enclosure. Proposed by [Koller et al. (2025), *Set-Based Training for Neural Network Verification*](https://arxiv.org/abs/2401.14961), the approach propagates zonotopes through the network and differentiates the resulting enclosure. By reducing its size, training aims to improve robustness and facilitate subsequent verification.

CTRAIN provides this approach through `SetBasedModelWrapper`, using PyTorch autograd and auto_LiRPA's forward-propagation operators. Unlike [IBP](ibp.md), which represents each layer through independent intervals, zonotopes retain shared generators that describe correlations between activations. The training objective combines cross-entropy at the output enclosure's centre with a penalty on its size.

## Representation and Propagation

A zonotope represents inputs through a centre `c` and generator directions `G`. CTRAIN additionally carries an independent interval radius `r` to represent approximation errors:

```text
Z = { c + sum_q xi_q * G_q + eta : |xi_q| <= 1, |eta_i| <= r_i }
```

Affine layers transform the centre and generators while propagating the independent radius through absolute weights. ReLU layers use a linear relaxation and add an enclosure error for unstable activations. Both the relaxation and its error remain differentiable, allowing the loss to influence the propagated set as well as its centre.

The `error_propagation` option determines how these approximation errors are carried. Its default, `"interval"`, keeps them as independent intervals. With `"generators"`, the errors become additional diagonal generators, preserving more correlations at the cost of memory. Both modes retain the input box's diagonal generators and all enclosure errors.

Propagation reuses auto_LiRPA's `LinearBound` representation and forward operators. Small local adapters address numerical and autograd limitations in the installed version, including narrow unstable ReLU intervals and affine derivatives at zero weights. They do not replace global LiRPA operators. The [propagation reference](api/bound/set_based_zono.md) documents the lower-level representation and supported operations.

## Training Objective

For a propagated output enclosure with centre `c`, generators `G` and independent radius `r`, CTRAIN computes the following batch-mean objective:

```text
size = sqrt(sum(G**2) + sum(r**2))
L_train = (1 - tau) * CE(c, y) + tau * size / (K * eps_t)
```

Here, `K` is the number of output classes and `eps_t` is the scheduled training radius before loader normalisation. The coefficient `tau`, which lies in `[0, 1]` and defaults to `0.1`, balances centre classification and enclosure size. At zero radius, the objective reduces to ordinary clean cross-entropy. Gradients include the centre, generators, independent radius and relaxation parameters; the size term has a zero gradient at zero norm.

### Choosing the Input Set

With `input_set="box"`, CTRAIN propagates the full clipped perturbation box. Each input coordinate contributes a diagonal generator, so memory use grows with input dimension. The alternative `input_set="attack"` constructs a smaller set from `num_generators` randomised one-step gradient-sign attacks, defaulting to eight.

For attack sets, CTRAIN averages the candidate positions to obtain the centre and uses their centred directions as generators. These directions are scaled coordinatewise so that the resulting set remains inside the nominal clipped box. The `pgd_eps_factor` scales only the attack search range; it must be positive and finite and cannot enlarge the training region.

Attack-set construction is detached from the training graph and uses input gradients without modifying parameter gradients. These selected sets are training surrogates; certification always considers the full nominal perturbation box.

## Training with CTRAIN

In the following example, we train the CNN6 architecture on MNIST for robustness against ℓ∞ perturbations with radius ε = 0.1. First, we import the required functions, load the dataset and construct the model.

```python
from CTRAIN.data_loaders import load_mnist
from CTRAIN.model_definitions import CNN6_SetBased
from CTRAIN.model_wrappers import SetBasedModelWrapper

input_shape = (1, 28, 28)
train_loader, test_loader = load_mnist(batch_size=128, val_split=False)
model = CNN6_SetBased(input_shape)
```

Now, we wrap the model with `SetBasedModelWrapper`. We configure one epoch of clean warm-up, followed by a linear radius ramp over 20 epochs. The remaining epochs use the full training radius. The propagation chunk size controls how many examples are propagated together, while the loader batch size determines the number of examples contributing to each optimiser update.

```python
wrapped_model = SetBasedModelWrapper(
    model=model,
    input_shape=input_shape,
    eps=0.1,
    num_epochs=70,
    warm_up_epochs=1,
    ramp_up_epochs=20,
    lr=5e-4,
    lr_decay_kwargs={"milestones": [50, 60], "gamma": 0.1},
    tau=0.1,
    input_set="box",
    error_propagation="interval",
    propagation_batch_size=1,
    device="cuda",
)
wrapped_model.train_model(train_loader)
```

The loader supplies normalisation metadata and feasible input limits. Radii are specified in raw pixel units, then transformed into model coordinates and clipped to valid inputs. Further details are provided in the [data and models guide](guides/data_and_models.md).

Finally, we set the model to evaluation mode and compute natural, certified and adversarial accuracy:

```python
wrapped_model.eval()
std_acc, cert_acc, adv_acc = wrapped_model.evaluate(test_loader)
```

Set-based wrappers default to `eval_method="FORWARD"`. This evaluates true-class margin bounds over the full clipped input box using auto_LiRPA forward propagation with same-slope ReLU relaxations. A sample is certified when every competing-class margin has a positive lower bound. The returned accuracies are fractions.

## BatchNorm and Batching

During positive-radius training, each propagation chunk first makes a clean nominal pass to compute BatchNorm statistics. Set propagation reuses these statistics without updating the buffers again. The mean and biased variance retain their computation graph, so training differentiates through the nominal statistics. Attack construction uses detached copies of the same statistics.

Running statistics update as `0.99 * old + 0.01 * batch`, independently of the PyTorch module's momentum. At zero radius, warm-up computes statistics over the full logical loader batch. Consequently, changing `propagation_batch_size` changes training statistics, buffer-update counts and attack candidates as well as memory use; it is part of the training configuration.

Chunk losses and gradients are weighted by their sample counts, with one optimiser update per logical loader batch. Gradient clipping defaults to 10. Optional `population_bn=True` recalibrates BatchNorm on training inputs after each epoch, before validation selection and checkpointing. Evaluation uses fixed running statistics from the current model.

## Training Behaviour and Configuration

Supported models consist of sequential convolutional layers with zero padding, linear layers, BatchNorm1d/2d with running statistics, ReLU, `Flatten(1, -1)` and identity layers. Nested sequential models and CTRAIN models exposing `.layers` are supported. Branches, functional computation, reused modules and unsupported layers are rejected before training.

The default optimiser is Adam with learning rate `5e-4`. The `train_eps_factor` optionally scales the training target while evaluation continues to use the original `eps`. Learning-rate scheduling supports epoch or batch units through `lr_decay_schedule_unit`; the example uses epoch milestones. Unlike Shi-IBP, this wrapper does not automatically apply Shi initialisation or regularisation.

Full-box input generators can be costly for image models, especially with `error_propagation="generators"`, where activation widths also increase the number of generators. `propagation_batch_size` limits the sample dimension only. Attack sets offer fewer input generators, but change the training objective's input region.

Without validation data, `train_model` returns the final weights. With a validation loader, the trainer selects by forward-bound certified accuracy after ramp-up at the nominal evaluation radius. Trajectory checkpoints retain model, optimiser, scheduler and selection state; RNG and loader state are not replayed. See the [checkpoint guide](guides/checkpoints.md) for saving and resumption.

### Evaluation and Complete Verification

Evaluation constructs a fresh LiRPA model from the current weights and running BatchNorm buffers. The wrapper also supports the other [incomplete evaluation methods](guides/evaluation.md) and [complete verification](complete_verification.md) through `evaluate_complete`. Generator-based error propagation is a training option; ordinary forward bounds are used for validation and default evaluation.

Margin propagation may tighten certificates compared with separately bounding each output logit and then subtracting their enclosures. All certificates cover the full nominal input box, including when training used attack sets. As with the other floating-point bounding methods, these computations do not include a directed-rounding arithmetic proof; very small radii can also be affected by cancellation when recovering independent radii from affine bias intervals.

## Hyperparameter Optimisation

Set-based training supports the shared [hyperparameter optimisation interface](guides/hpo.md). Its configuration space includes `tau`, learning rate, optimiser, warm-up and ramp durations, training-radius factor and learning-rate decay factor. Input-set, error-propagation, chunk-size and population-BatchNorm settings are retained when recreating trial wrappers. Use separate validation data for the search and reserve the test split for final assessment.

## Implementation and API Reference

CTRAIN implements the centre-classification and enclosure-size objective using auto_LiRPA propagation and PyTorch autograd. The authors' CORA implementation uses its own set-based propagation and backpropagation. CTRAIN's centred, containment-preserving attack construction differs from the paper appendix's uncentred attack displacements and CORA's FGSM variants. Its sample-weighted chunk gradients also differ from the pinned CORA trainer's sum of chunk means. These differences matter when comparing training configurations; the example does not establish reproduction of published scores.

The API reference documents the [set-based wrapper](api/model_wrappers/set_based_model_wrapper.md), [trainer](api/train/certified/set_based.md), [loss and attack-set construction](api/train/certified/losses/set_based.md) and [set propagation](api/bound/set_based_zono.md).
