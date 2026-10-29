# General Input-Output Properties

Beyond local robustness for image classification, CTRAIN supports training against more general input-output specifications. These include output-range constraints, safety conditions for controllers and relations between multiple predictions. Users supply a task loss and a collection of specifications, each consisting of closed input boxes and an output property expressed as a finite Boolean combination of affine inequalities.

For a network to satisfy a specification, its output property must hold for every input in every declared box. CTRAIN formulates these requirements through satisfaction margins, analogous to the classification margins used for local robustness. This allows existing certified training methods to be adapted to more general specifications while retaining a common interface for training and verification.

CTRAIN provides this workflow through `PropertyModelWrapper`. The wrapper combines the supplied task loss with a property objective, delegates optimisation to the property trainer and exposes fresh certification over the declared domains.

## Training Objective

For network output `y`, an affine comparison `a(y) >= b(y)` has satisfaction margin `a(y) - b(y)`. Conjunction takes the minimum of its constituent margins, while disjunction takes their maximum:

```text
m(a >= b) = a(y) - b(y)
m(And(phi, psi)) = min(m(phi), m(psi))
m(Or(phi, psi)) = max(m(phi), m(psi))
```

A positive margin therefore implies satisfaction of the formula. CTRAIN appends the operations required to compute these margins to the network and uses auto_LiRPA to obtain lower bounds over each input box. Outer conjunctions are split into separate requirements, while nested disjunctions retain their logical structure.

At each optimisation step, the trainer samples `K` property boxes independently of the task examples. With the default hinge objective, it combines the task loss with the sum of lower-margin deficits:

```text
L_train = L_task + (lambda / K) * sum_k sum_j max(0, delta - lower_margin[k, j])
```

Here, `lambda` is controlled by `property_weight`, and `delta` by `margin`. Each property penalty vanishes once its lower bound reaches the positive target margin. This avoids encouraging further increases in satisfaction margins that may come at the expense of task performance. The target margin is distinct from the logical satisfaction threshold of zero.

## Training with CTRAIN

In the following example, we train a small controller whose two actions should remain within `[-0.1, 0.1]` for every state in a declared box. The task loss encourages the controller to match a proportional control law. We use generated state-target pairs to illustrate the workflow; verification will cover the complete box rather than only these sampled states.

First, we import the required functions and define the specification. `Outputs()` provides symbolic references to the network outputs, and `Box` specifies the input domain.

```python
import torch
from torch.utils.data import DataLoader, TensorDataset

from CTRAIN.model_wrappers import PropertyModelWrapper
from CTRAIN.properties import Outputs, And, Or, Box, PropertySpec

y = Outputs()
property_spec = PropertySpec(
    "bounded_actions",
    And(y[0] >= -0.1, y[0] <= 0.1,
        y[1] >= -0.1, y[1] <= 0.1),
    [Box([-1.0, -1.0], [1.0, 1.0])],
)
```

Now, we construct the controller and task data. The model receives two state coordinates and returns two actions. As in ordinary PyTorch training, the loader supplies input-target pairs for the task loss.

```python
model = torch.nn.Sequential(
    torch.nn.Linear(2, 32),
    torch.nn.ReLU(),
    torch.nn.Linear(32, 2),
)
states = 2 * torch.rand(256, 2) - 1
control_targets = -0.05 * states
train_loader = DataLoader(
    TensorDataset(states, control_targets), batch_size=32, shuffle=True,
)
```

We wrap the model with `PropertyModelWrapper` and choose CROWN-IBP for the property bounds. During the first five epochs, the trainer optimises only the task loss. It then increases the property weight linearly over 20 epochs before using its full value for the remaining epochs.

```python
wrapped_model = PropertyModelWrapper(
    model=model,
    input_shape=(2,),
    properties=[property_spec],
    task_loss=torch.nn.MSELoss(),
    num_epochs=50,
    training_method="CROWN-IBP",
    property_weight=20.0,
    margin=1e-3,
    warm_up_epochs=5,
    weight_ramp_epochs=20,
    property_batch_size=8,
    refinement=True,
    max_cells=128,
    max_depth=8,
    device="cpu",
)
wrapped_model.train_model(train_loader)
```

Finally, we compute fresh lower bounds for every requirement over the complete domain. A low training loss alone does not establish that the specification holds.

```python
report = wrapped_model.certify_properties(method="CROWN")
print("Whole domain certified:", report["certified"])
print("Cell outcomes:", [cell["outcome"] for cell in report["cells"]])
```

The [controller tutorial](examples/general_properties.ipynb) demonstrates the bounded-action workflow step by step. The [MNIST arithmetic tutorial](examples/mnist_arithmetic.ipynb) trains independently predicted digit values and their sum with an affine consistency property.

## Semantics and Compiler

Output expressions support addition, subtraction and multiplication by finite scalar constants. Comparisons may use `>=`, `<=`, `>` or `<`, or the corresponding named constructors `Ge`, `Le`, `Gt` and `Lt`. Use `And` and `Or` to compose them. For example, the following formula requires bounded first actions and allows either of two conditions on the second action:

```python
alternative_actions = And(
    y[0] >= -0.1,
    y[0] <= 0.1,
    Or(y[1] >= 0.03, y[1] <= -0.03),
)
```

Python's `and`, `or` and chained comparisons cannot represent symbolic formulas. Exact equality, disequality and products between symbolic expressions are unsupported. Approximate equality can be expressed through two inequalities with an explicit tolerance. Named comparisons additionally accept a positive `scale` for normalising margins with different units.

Strict and non-strict comparisons share the same quantitative margin, but their Boolean evaluation distinguishes equality. Certification requires a strictly positive lower margin; a zero bound remains unresolved, including for non-strict inequalities. A negative lower bound alone does not establish a violation, since the relaxation may be loose.

Affine requirements use auto_LiRPA's specification matrix directly, preserving correlations between outputs. Composite formulas use native minimum and maximum operators in a graph sharing the original model's parameters. For complete verification, CTRAIN exports equivalent affine/ReLU constructions. The [property API](api/properties.md) and [bound API](api/bound/properties.md) describe the supported expressions and graph interfaces.

## Choosing a Training Strategy

The `training_method` argument selects the property objective; its default is `"CROWN-IBP"`. Method-specific parameters are supplied through `training_options`.

| Strategy | Property objective | Main options |
|---|---|---|
| [IBP](ibp.md), [CROWN-IBP](crown_ibp.md), CROWN | Bound each requirement over the full sampled box. | CROWN-IBP uses `beta=1.0` to weight CROWN-IBP margins against IBP margins. |
| [MTL-IBP](expressive_losses.md) | Combine bound-based and adversarial property losses. | `alpha=0.5` weights the IBP loss. |
| [CC-IBP](expressive_losses.md) | Combine bound-based and adversarial margins before applying the loss. | `alpha=0.5` weights the IBP margins. |
| [Exp-IBP](expressive_losses.md) | Combine the two property losses geometrically. | `alpha=0.5`; with hinge loss, a zero factor can suppress gradients. |
| [SABR](sabr.md) | Bound an adversarially selected subregion contained in the nominal box. | `subselection_ratio=0.1`. |
| PGD | Evaluate the property loss at adversarial input points. | `pgd_steps=8`, `pgd_restarts=1`. |
| [IBP-R](ibp_r.md) | Combine adversarial property loss with a ReLU-relaxation area penalty. | `rho=2e-4`, `regularisation_mask=False`. |
| [Gaussian loss smoothing](gls.md) | Smooth the task and property objective over network parameters. | `estimator="rgs"`, `samples=2`, `sigma=1e-3`, `bound_method="CROWN"`. |

For example, MTL-IBP with a stronger emphasis on bound-based property loss can be selected through:

```python
mtl_model = PropertyModelWrapper(
    model=model,
    input_shape=(2,),
    properties=[property_spec],
    task_loss=torch.nn.MSELoss(),
    num_epochs=50,
    training_method="MTL-IBP",
    training_options={"alpha": 0.8, "pgd_steps": 20, "loss": "cross_entropy"},
    property_weight=20.0,
)
```

All strategies support `loss="hinge"` or `loss="cross_entropy"`. Cross-entropy uses CTRAIN's padded-margin loss and does not use the hinge target margin. PGD-based strategies additionally expose step-size, restart and decay controls. Their `pgd_region_factor` scales the search region around the box midpoint without changing the declared property domain; SABR keeps its selected subregion inside that domain.

The hybrid, adversarial, regularised and smoothed property strategies are experimental adaptations of the corresponding classification methods. They retain this workflow's fixed-statistics BatchNorm policy. TAPS/STAPS, set-based training and CC-Dist are not supported as property-training strategies. The classification wrappers' Shi initialisation and regularisation recipes are also separate from the property-bound objective.

Fresh epoch diagnostics, refinement and model selection use full-box bounds: the selected bound method for IBP, CROWN-IBP and CROWN, the configured `bound_method` for GLS, and IBP for the other strategies. Supported `relu_upper_retention` settings below one affect training surrogates only. Certification independently chooses its bound method and never treats an attack point, SABR subregion or shrunk interval as a certificate for the whole box.

## Training and Domains

Property boxes use exactly the coordinates consumed by the verification model. No dataset normalisation or perturbation radius is inferred. Multiple boxes describe a finite union; original boxes may overlap. Initially, each original box receives equal sampling weight, rather than weight proportional to its volume.

With `refinement=True`, the trainer splits unresolved cells at the midpoint of their widest normalised coordinate. Children cover their parent and inherit half its sampling weight. Refinement can tighten bounds without increasing the original region's contribution to the training objective. The `max_cells` and `max_depth` limits apply per original box. A fraction of certified cells does not establish certification of the complete domain.

By default, `property_batch_size` controls how many cells are sampled per task batch and how many are bounded together during certification. Sampling is with replacement and independent of task data. `match_property_batch_size=True` instead samples one property box per task example, including in partial batches.

Large finite collections sharing a single formula can supply a `property_dataset`, an indexable object yielding `Box` instances and exposing a serialisable `to_dict()` descriptor with reconstruction settings and content hashes. The dataset supplies the complete region collection, generated on demand, while the single `PropertySpec` supplies the formula and initial graph domain. Certification evaluates every dataset box in batches. Adaptive refinement and complete verification require explicit property boxes rather than dataset mode.

A `verification_model` can append fixed dynamics or other supported computations to the task model. It must contain the supplied task model by identity and introduce no additional trainable parameters. `verification_input_shape` specifies its input shape when this differs from the task model's input shape.

Training evaluates both models with fixed BatchNorm statistics and parameter gradients enabled. Optional `population_bn=True` recalibrates statistics on task training inputs after each epoch. With validation, model selection starts after the weight ramp and prioritises complete property certification, then task loss; until certification is achieved, it prioritises the minimum lower bound. Without validation, the final weights are returned.

The [checkpoint guide](guides/checkpoints.md#general-property-checkpoints) explains trajectory saving, configuration matching and RNG restoration. Reconstruct a property dataset before loading its checkpoint; mismatched dataset descriptors are rejected. Selected inference weights are distinct from a resumable training trajectory.

## Certification and Complete Verification

`certify_properties` computes fresh bounds and reports results for individual cells and the complete domain. Its default bound method is CROWN. Other choices include IBP, CROWN-IBP and `"ALPHA-CROWN"`; slope-optimisation iterations can be configured through `bound_opts={"optimize_bound_args": {"iteration": 100}}` when constructing the wrapper.

A complete-domain certificate requires every requirement to hold on every covering cell. Reports distinguish certification, unresolved cases, timeouts and witnessed violations. These results use floating-point LiRPA bounds; they do not include a directed-rounding arithmetic proof.

For unresolved cells, `complete=True` first searches for a Boolean counterexample and then passes scalar-margin ONNX graphs and VNNLIB queries to CTRAIN's abCROWN integration. Supply a fresh results directory and a positive timeout:

```python
complete_report = wrapped_model.certify_properties(
    method="CROWN",
    complete=True,
    results_path="./verification/bounded_actions",
    timeout=300,
)
```

An `unsat` result certifies the exported graph. A `sat` response is recorded as a violation only when an accessible witness has been checked; otherwise it remains unresolved. Export checks compare native, affine/ReLU and ONNX Runtime outputs, but these numerical checks do not prove equivalence for all inputs. The report records which graph was certified. This property workflow currently uses abCROWN; the [classification complete-verification interface](complete_verification.md) additionally supports NeuralSAT.

## User-Selected Networks and Specifications

Existing ONNX models and VNNLIB specifications can also be used for training or repair. First, we load the model and specification; then, we construct the property wrapper with a task loss appropriate for the original task:

```python
from CTRAIN.data_loaders import load_vnnlib_problem

imported_model, imported_spec, provenance = load_vnnlib_problem(
    "my_network.onnx", "my_property.vnnlib",
)
repair_model = PropertyModelWrapper(
    model=imported_model,
    input_shape=tuple(imported_spec.domains[0].lower.shape),
    properties=[imported_spec],
    task_loss=torch.nn.MSELoss(),
    num_epochs=20,
)
```

For score-preserving repair, task loaders can supply the original model's scores as targets. The [ONNX/VNNLIB tutorial](examples/onnx_vnnlib.ipynb) demonstrates the complete workflow.

By default, VNNLIB output assertions describe unsafe states and are complemented to obtain the desired property. For example, unsafe `Y_0 <= Y_1` becomes the desired condition `Y_0 > Y_1`. Use `output_semantics="safe"` for files that already describe the desired condition. `load_vnnlib_property` imports only the specification when a PyTorch model is already available.

The importer supports affine output expressions, strict and non-strict inequalities, nested Boolean logic and finite unions of closed input boxes. Coupled input inequalities, mixed input/output logic, exact equality and nonlinear terms are unsupported. Input normalisation is never inferred. The `input_shape` option resolves dynamic dimensions or reshapes flattened coordinates. The loader checks converted predictions and records file hashes; optional `expected_sha256` values check caller-supplied digests. These checks provide numerical and file provenance checks, rather than a proof of graph equivalence or source authenticity.

### ACAS Presets

For the supported ACAS Xu examples, `load_acasxu` loads locally available artifacts and checks their pinned manifest. It performs no download:

```python
from CTRAIN.data_loaders import load_acasxu

acas_model, acas_spec, manifest = load_acasxu(
    "/path/to/pinned/artifacts", case="7",
)
acas_wrapper = PropertyModelWrapper(
    model=acas_model,
    input_shape=(5,),
    properties=[acas_spec],
    task_loss=torch.nn.MSELoss(),
    num_epochs=20,
    refinement=True,
)
```

`download_acasxu(new_directory, revision)` explicitly downloads the preset artifacts and records a pinned hash manifest. `acasxu_property` exposes the preset specifications for caller-supplied boxes. See the [data-loader reference](api/data_loaders/data_loaders.md) for the available cases and import options.

## API Reference

The API reference documents the [property wrapper](api/model_wrappers/property_model_wrapper.md), [specification language](api/properties.md), [trainer](api/train/certified/properties.md), [property losses](api/train/certified/losses/properties.md), [bounds](api/bound/properties.md) and [certification functions](api/eval/properties.md).
