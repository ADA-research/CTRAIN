# General output properties (experimental)

`PropertyModelWrapper` is a separate certified-training workflow for finite
Boolean combinations of affine output comparisons over declared input boxes.
Existing robustness wrappers retain their APIs. This is a prototype, not a
claim to support arbitrary first-order logic or temporal specifications.

The API reference is generated from the source docstrings:
[wrapper and configuration](api/model_wrappers/property_model_wrapper.md) and
[training algorithm](api/train/certified/properties.md).

```python
import torch
from CTRAIN import PropertyModelWrapper
from CTRAIN.properties import Outputs, And, Or, Box, PropertySpec

y = Outputs()
property_spec = PropertySpec(
    'safe_outputs',
    And(y[0] >= -.1, y[0] <= .1,
        Or(y[1] >= .3, y[2] <= -.3)),
    [Box([-1., -1.], [1., 1.])],
)
model = torch.nn.Sequential(torch.nn.Linear(2, 32), torch.nn.ReLU(),
                            torch.nn.Linear(32, 3))
wrapper = PropertyModelWrapper(
    model, (2,), properties=[property_spec], task_loss=torch.nn.MSELoss(),
    num_epochs=50, training_method='CROWN-IBP', property_weight=20.,
    refinement=True,
    warm_up_epochs=5, weight_ramp_epochs=20,
)
# train_loader and val_loader yield (input, target) pairs.
# wrapper.train_model(train_loader, val_loader)
# report = wrapper.certify_properties(method='CROWN-IBP')
```

The schedule fits inside `num_epochs`: first `warm_up_epochs` optimize the task
loss alone, then `weight_ramp_epochs` linearly increase the property-loss weight,
then the remaining epochs use the full weight. Both durations default to zero.
Warm-up computes no property samples, attacks, losses, certification scans or
refinement. Its history records `certified=None` and `property_evaluated=False`.
The ramp is updated per batch and starts after warm-up; zero ramp applies the
full weight at the first update after warm-up. Property regions retain their
declared size. Validation-based epoch selection starts once the ramp finishes.
Checkpoint resumption continues the schedule from the restored epoch; old
checkpoints without `warm_up_epochs` are interpreted as zero warm-up.

All case-study training runners accept `--warm-up-epochs` and
`--weight-ramp-epochs`. MNIST and ACAS HPO tune warm-up over 0–5 epochs (capped
for short runs) and retain the property-weight ramp search, with ranges that
fit the total epoch budget. Fixed baselines retain their configured schedules.

## Choosing a training strategy

Large collections sharing one formula can supply a `property_dataset` to
`PropertyModelWrapper`. This finite, indexable object returns a `Box` for each
index and exposes `to_dict()` with reconstruction settings and content hashes.
The single `PropertySpec` supplies the formula and an initial domain for graph
construction; the dataset supplies the full training and certification domain
collection. Boxes are sampled uniformly with replacement, independently of task
batches, and generated only when needed. No VNNLIB files or full endpoint pool
are required. MNIST addition uses this mode for every training pair.

Epoch diagnostics and `wrapper.certify_properties()` evaluate every dataset box
in batches and return aggregate results. Dataset mode does not support adaptive
refinement or complete verification; pass explicit `properties=[...]` to certify
selected regions with those features. Checkpoints save the dataset descriptor
and RNG state; the caller reconstructs the same dataset before loading, and
configuration comparison rejects differing descriptors. The generic wrapper
supports separate task and property batch sizes. `match_property_batch_size=True`
instead samples one property box per task example, including partial batches.
MNIST keeps these sizes independent: `--batch-size` controls task examples
and `--property-batch-size` controls independently sampled property boxes per
update, including partial task batches. HPO keeps both sizes fixed rather than
tuning them.

Robustness wrappers still accept `eps`; they do not require a property object.
Wrapper implementations live in `CTRAIN.model_wrappers.certified` and
`CTRAIN.model_wrappers.adversarial`. Existing package exports and direct module
imports remain compatible, including patching through old module paths.

```python
from CTRAIN.model_wrappers.certified import ShiIBPModelWrapper, PropertyModelWrapper
from CTRAIN.model_wrappers.adversarial import PGDModelWrapper

# Classification robustness keeps its existing API.
robust = ShiIBPModelWrapper(classifier, input_shape=(3, 32, 32),
                           eps=8 / 255, num_epochs=100)

# Only general-property training needs declared properties and a task loss.
wrapper = PropertyModelWrapper(
    model, (2,), properties=[property_spec], task_loss=torch.nn.MSELoss(),
    num_epochs=50, training_method='MTL-IBP',
    training_options={'alpha': .8, 'pgd_steps': 20, 'loss': 'cross_entropy'},
    property_weight=20.,
)
```

Choose the training objective with `training_method` (default `CROWN-IBP`);
no second method argument is required. Fresh full-box epoch diagnostics,
refinement and checkpoint selection use the same bound method for `IBP`,
`CROWN-IBP` and `CROWN`, GLS's `training_options['bound_method']` for GLS,
and `IBP` for the other strategies. These computations always use sound,
full-box bounds, without training-time ReLU shrinkage.

There is no separate selection-method argument.
`certify_properties(method='CROWN')` independently chooses certification bounds;
its default remains `CROWN`. No SABR subregion, PGD point, Gaussian sample, or shrunk ReLU interval
is used as a whole-box certificate. Old bound-training checkpoints remain
loadable; switching strategy or parameters requires a new trajectory.

The old `bound_method` constructor argument has been removed. Use
`training_method='IBP'` instead of `bound_method='IBP'`. Old checkpoints are
migrated on load when their training and selection settings match the new
method-derived policy. A checkpoint with different historical selection bounds
is rejected before loading weights; resume that trajectory with its original
CTRAIN version. HPO studies retain their historical `bound_method`
trial-parameter key for the training strategy so existing studies remain usable.

| Strategy | Property training computation | Additional parameters and defaults |
|---|---|---|
| IBP, CROWN-IBP, CROWN | Bound every obligation over the sampled full box. | IBP supports `relu_upper_retention=1.0`; CROWN-IBP supports `beta=1.0` (CROWN-IBP weight against IBP margins). |
| MTL-IBP | Mix IBP and adversarial property losses. | `alpha=0.5` is the IBP weight. |
| CC-IBP | Mix IBP lower margins and adversarial point margins before applying the loss. | `alpha=0.5` is the IBP weight. |
| EXP-IBP | Geometric mixture of mean IBP and adversarial losses. | `alpha=0.5`; zero hinge factors can suppress gradients. |
| SABR | PGD selects a center; IBP bounds a coordinatewise contained subregion. | `subselection_ratio=0.1`; half-width is this fraction of each original half-width. |
| PGD | Apply the property loss at adversarial input points. | The shared PGD controls below. |
| IBP-R | Adversarial property loss plus unstable ReLU triangle areas from fresh IBP. | `rho=2e-4`, `regularisation_mask=False`; the mask tests formula satisfaction at adversarial points. |
| GLS | Existing RGS/PGPE optimizer smooths the task/property loss over parameters. Sampled boxes stay fixed across one optimizer step. | `estimator='rgs'`, `samples=2`, `sigma=1e-3`, `sigma_lr=1e-3`, `bound_method='CROWN'`, `sigma_decay_milestones=[]`, `sigma_decay_factor=0.4`. PGPE requires even samples and learns its scale. |

All strategies accept `loss='hinge'` (default) or `loss='cross_entropy'`, plus
`gradient_clip=None`. Cross-entropy reuses CTRAIN's padded-margin loss and does
not use hinge slack. MTL/CC/Exp/SABR/IBP-R also accept
`relu_upper_retention=1.0`; smaller retention is an unsound training surrogate
whose interval operator is restored after each computation, including failures.

PGD-based strategies accept `pgd_steps=8`, `pgd_step_size=0.5`,
`pgd_restarts=1`, `pgd_early_stopping=False`, `pgd_region_factor=1.0`,
`pgd_decay_factor=0.1`, and `pgd_decay_checkpoints=[]`. The region factor scales
search around the box midpoint without changing the declared property region.
SABR always clamps its subregion into the nominal box. Hinge attacks minimize
the signed formula margin even at satisfied points; cross-entropy attacks
maximize pointwise margin loss. Successful candidates are retained and attacks
use only input gradients.

Point losses and attack success reuse the affine-leaf compiler, evaluating
shared comparisons together instead of rebuilding thousands of scalar
operations for large disjunctions. Boolean success preserves strict comparison
ties; min/max ties use the compiler's valid subgradients.

These hybrid, adversarial, regularized, and smoothed strategies are experimental
adaptations to logical properties. They retain this workflow's fixed-statistics
BatchNorm policy and do not establish paper reproduction scores. TAPS/STAPS
need a latent property search formulation, set-based training needs a property
enclosure-size objective, and CC-Dist needs teacher-feature property semantics;
those names fail explicitly before graph conversion. Classification-specific
Shi initialization/regularization recipes remain on the existing robustness
wrappers; `IBP` here denotes the property-bound objective.

The controller, ACAS repair, and MNIST-addition runners accept, for example:

```bash
python -m case_studies.controller.run --smoke \
  --method SABR --training-options '{"subselection_ratio": 0.2, "pgd_steps": 12}' \
  --output /tmp/controller-sabr-new
```

MNIST config files may include a `training_options` JSON object. Its HPO runner
keeps an explicitly configured strategy fixed while tuning the existing
optimizer, weight, and sampling controls. Both MNIST and ACAS HPO additionally
tune MTL-IBP `alpha` uniformly in `[0, 1]` or SABR `subselection_ratio`
logarithmically in `[0.01, 1]`. The latter is the ratio defining
`tau = subselection_ratio * (upper - lower) / 2`, not an absolute radius.
These two trial options override their configured baseline values; other
algorithm options stay fixed. Fixed MNIST baselines retain their configured values.
Without new settings, the existing three-bound-method search is retained.
ACAS HPO accepts `--training-method` and `--training-options` to configure a strategy;
the suite runner forwards the same options to each repair. Training
configuration and provenance record the effective strategy and parameters.

## Semantics and compiler

`Outputs()[i]` indexes the model's batched vector output. Expressions support
addition, subtraction and multiplication by finite scalar constants. Use
`And(...)` and `Or(...)`, and `>=`, `<=`, `>`, `<` or named `Ge`, `Le`, `Gt`,
`Lt` constructors. Python `and`/`or`, chained comparisons, symbolic products,
exact equality and disequality are rejected. Express approximate equality as
two inequalities with an explicit tolerance. Named comparisons accept a
positive `scale` to normalize margins with different units.

An atom `a(y) >= b(y)` has margin `a(y)-b(y)`. Conjunction uses minimum and
disjunction uses maximum. Strict and non-strict atoms have the same quantitative
margin, but their Boolean evaluation distinguishes equality. A strictly
positive certified lower margin suffices for every supported formula. Zero is
conservatively unresolved, including for non-strict inequalities; a negative
lower bound alone does not establish a violation.

Affine obligations use auto_LiRPA's specification matrix directly, preserving
output correlations. Composite obligations append native minimum/maximum
operators to a graph sharing the original model's parameters. Export replaces
these with equivalent affine/ReLU constructions. Scales are folded into affine
coefficients: explicit division triggers graph rewrites in the bundled LiRPA
version that lose perturbation metadata. Affine graph constants are registered
buffers, and affine evaluation uses `linear` to avoid an intermediate-bound
dtype inconsistency observed with elementwise reduction in float64. No custom
LiRPA bound operator is installed.

## Training and domains

The default remains bound-based training with `IBP`, `CROWN-IBP`, or `CROWN`.
Use `training_method` to select another strategy and `training_options` to set
its parameters. The default loss sums hinge penalties
`max(0, margin - lower_bound)` over top-level conjuncts and adds a weighted
property term to the supplied task loss. Nested disjunctions retain their full
logical structure. A positive training margin supplies slack; it is distinct
from the logical threshold of zero. An optional linear weight ramp is available.

Domains are explicit finite unions of closed boxes, in exactly the coordinates
consumed by the verification model. No implicit dataset normalization or input
radius is applied. Floating-point casts expand box endpoints outward. A
`verification_model` can append fixed dynamics or other supported computations;
it must contain the task model by identity and add no trainable parameters.
The controller example uses this to train actions while checking next states.

Optional refinement splits uncertified cells at the midpoint of their widest
normalized coordinate. Children cover the parent and share its sampling weight,
so splitting does not increase that region's training weight. `max_cells` and
`max_depth` apply per original box. Original boxes may overlap; the training
sampling measure weights each original box equally rather than measuring union
volume. Certification validates the entire partition cover. A fraction of
certified cells is never presented as a whole-domain certificate.

Training uses fixed BatchNorm statistics in evaluation mode with parameter
gradients enabled. `population_bn=True` optionally recalibrates statistics on
training inputs after each epoch, before selection and evaluation. Caller modes
and gradient flags are restored. With validation, selection prioritizes full
certification and then task loss; before certification it prioritizes the
minimum lower bound. Without validation, the final weights are returned.

Epoch checkpoints contain trajectory weights, optimizer/scheduler, partition,
selection history and RNG states, including an explicit loader generator.
Load trusted checkpoints into a wrapper with matching configuration to resume.
Validation-selected inference weights are kept distinct: saving or continuing
a trajectory after selection requires loading an epoch checkpoint first.
Exact CPU resumption is tested for the supplied single-worker deterministic
loader; arbitrary multi-worker or nondeterministic CUDA trajectories are not
claimed reproducible.

## Certification and complete verification

`certify_properties` always computes fresh bounds using the requested method.
`method='ALPHA-CROWN'` uses auto_LiRPA's slope optimization. Configure iterations
through the wrapper's `bound_opts={'optimize_bound_args': {'iteration': 100}}`
(auto_LiRPA defaults to 20). Model weights and existing gradients are preserved;
this option is for certification, while training retains its existing methods.
Reports distinguish certified, unknown, timeout and witnessed violations.
A whole-domain result requires all obligations on every covering cell. These
are floating-point LiRPA certificates without a directed-rounding arithmetic
proof; constants and model operations use the configured model dtype.

For unresolved cells, `complete=True` first searches for a genuine Boolean
counterexample and then exports an equivalent scalar-margin ONNX graph and
VNNLIB query to CTRAIN's existing alpha-beta-CROWN runner. Supply a fresh
`results_path` and a positive `timeout`. Export checks native/ReLU and ONNX
Runtime parity, and normalizes constant aliases for the bundled ONNX importer.
An `unsat` result certifies the exported graph. A solver `sat` response without
an accessible, checked witness stays unknown. Export parity checks do not prove
bitwise or real-arithmetic equivalence between PyTorch and ONNX; reports record
which graph was certified. A small overlapping-disjunction example was actually
proved by this path, in addition to mocked status-handling regressions.

## Case studies and evidence

See [the case-study notebooks](examples/case_studies/index.md) for commands.
The experimental MNIST addition example learns independent classification heads
for the two operand digits (10 classes each) and their sum (19 classes). Three
cross-entropy losses supervise `[a,b,a+b]`, while VNNLIB specifies exact consistency
between their argmax classes over clipped pixel boxes. Its loader normalizes
images and covers every official training image in both operand positions.
Certificates use auto_LiRPA affine logit-margin bounds with sound, conservative
Boolean composition (`bound_opts={'property_bound': 'atomic'}`). Clean classification
accuracy and certified consistency are separate metrics. The sum head remains
independently learned. This is a new MNIST-based benchmark, not a reproduction
of an established named benchmark; certificates do not establish correct digit
recognition. Earlier MSE/regression runs concern a different property and setup.

The controller checks an invariant ring under dynamics `x_next = x + u + w`,
with `|u_i| <= .1` and `|w_i| <= .01`. Its four input boxes cover
`[-1,1]^2` outside the open square `(-.3,.3)^2`; its next-state formula contains
a disjunction identifying that same ring. Thus a whole-domain one-step
certificate implies invariance at subsequent discrete steps under arbitrary
allowed disturbances. This does not certify continuous trajectories between
steps or convergence to a goal.

The recorded CPU seed-1 safety-teacher run used 60 epochs, starting from zero
actions that failed the disturbed invariant. Fresh final CROWN-IBP evaluation
certified all nine obligations across 185 covering cells, with minimum lower
margin approximately `9.68e-4`. Held-out action MSE was `2.22e-5`. All 32 sampled
100-step rollouts were safe; this empirical result is separate from the
certificate. The safety teacher is deliberately not a navigation objective.
Full CROWN remained inconclusive within the selected refinement budget.
The waypoint navigation variant has execution and teacher-feasibility checks,
not an established navigation certificate or performance benchmark.

ACAS Xu examples encode properties 7 and 8 over released score outputs with
explicit tie handling. They require source hashes and check their formulas
against the original unsafe VNNLIB predicates before real training. Original
ONNX predictions label sampled inputs with the source model's lowest-score
advisory. Repair minimizes cross-entropy on negated scores against those fixed
labels; held-out CE and advisory agreement measure fidelity, while score MSE
is diagnostic only. The library uses the installed `onnx2pytorch` converter and checks
predictions against ONNX Runtime before training. Converted affine/ReLU fixtures
are tested through all three bound methods and an optimizer update.
Synthetic smoke runs for both cases are execution checks only. Official
artifacts could not be fetched in this environment, so real ACAS Xu training,
source-file cross-checks and certification have not been performed. No GPU or
multi-seed campaign has been run.

### User-selected networks and specifications

The user can choose any local ONNX/VNNLIB pair within the supported operator
and specification subset, without a network registry or preset manifest:

```python
from CTRAIN.data_loaders import load_vnnlib_problem

model, spec, provenance = load_vnnlib_problem('my_network.onnx', 'my_property.vnnlib')
wrapper = PropertyModelWrapper(
    model, tuple(spec.domains[0].lower.shape), properties=[spec],
    task_loss=torch.nn.MSELoss(), num_epochs=20,
)
```

The importer supports arbitrary contiguous `X_i`/`Y_i` counts, finite closed
input boxes, affine output expressions, inequalities and nested `and`/`or`/`not`.
It preserves strictness and tie behavior. **By default output assertions describe
unsafe states and are complemented to obtain the desired property.** Pass
`output_semantics='safe'` for files already stating the desired condition.
For example, unsafe `(<= Y_0 Y_1)` becomes desired `Y_0 > Y_1`.

`load_vnnlib_property` imports a specification alone, so it can also be paired
with a caller-created PyTorch model. The problem loader infers static ONNX input
dimensions; both functions accept `input_shape` to reshape flattened coordinates
or resolve dynamic image dimensions. Input And/Or combinations of coordinate
bounds produce a finite union of boxes (at most 1024), including ACAS property 6.
Coupled input inequalities, mixed input/output logic, exact equality, nonlinear terms and arbitrary
SMT-LIB commands are rejected explicitly. No input normalization is inferred.
Numeric parsing uses floating-point arithmetic, not an exact-rational proof.

The general loader checks converted ONNX predictions and records file hashes.
Optional `expected_sha256` checks caller-supplied digests before parsing. Unlike
the benchmark presets, local files do not need all four artifact files or a
pinned upstream commit. This does not authenticate a claimed upstream source.

### ACAS presets

ACAS artifact handling and source checks are also available as convenience calls:

```python
from CTRAIN.data_loaders import load_acasxu

model, spec, manifest = load_acasxu('/path/to/pinned/artifacts', case='7')
wrapper = PropertyModelWrapper(
    model, (5,), properties=[spec], task_loss=torch.nn.MSELoss(),
    num_epochs=20, refinement=True,
)
# Supply task loaders containing original-model scores as preservation targets.
# wrapper.train_model(train_loader, val_loader)
# report = wrapper.certify_properties(method='CROWN-IBP')
```

`download_acasxu(new_directory, revision)` explicitly acquires the four preset
files and writes a pinned hash manifest. `load_acasxu` performs no download and
rejects missing hashes or a different source convention. `acasxu_property`
also exposes the presets for a caller-supplied box. See the generated
[loader reference](api/data_loaders/data_loaders.md).

Other pretrained ONNX models can use
[`load_onnx_model(path, probes)`](api/util/util.md): supply a representative batch
in the original input dtype and coordinates. The helper checks runtime parity
and returns a trainable CPU model. The property wrapper still needs to support
the converted graph's operators. Neither import nor sampling checks prove graph
equivalence or certify properties.

The DSL, quantitative logic and graph compilation reuse established ideas from
logical and certified constrained training. The prototype's contribution is
their integration with CTRAIN, domain-cover refinement, explicit certification
results and reproducible examples; no novelty or faithful reproduction of DL2
or ART is claimed.
