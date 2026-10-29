# Evaluation and Verification

CTRAIN evaluates trained classifiers through a common interface for natural accuracy, adversarial attacks and formal verification. Evaluation is independent of the training objective: a model trained with a surrogate loss can be assessed over the full nominal perturbation region using the same methods as an IBP-trained model.

This guide explains the returned metrics and the available evaluation workflows. The [complete-verification page](../complete_verification.md) provides detailed abCROWN and NeuralSAT setup and examples.

## What the Metrics Mean

The `evaluate` and `evaluate_complete` methods return `(standard_accuracy, certified_accuracy, adversarial_accuracy)`, with all accuracies expressed as fractions.

| Metric | Interpretation |
|---|---|
| Standard accuracy | Fraction of examples classified correctly without perturbation. |
| Certified accuracy | Lower bound on robust accuracy established by the selected verification method. |
| Adversarial accuracy | Upper bound on robust accuracy after the counterexamples found by the evaluation procedure. |

Certified and adversarial accuracy enclose the unknown robust accuracy. A failed attack leaves an example unresolved; it does not certify it. Similarly, a negative lower margin from an incomplete verifier may reflect relaxation error and does not by itself demonstrate a violation.

Training loss, attack-selected subregions, shrunk training intervals and TAPS/STAPS latent proxies describe training or selection computations. Final certified accuracy requires fresh verification over the complete nominal region.

## Loading a Model for Evaluation

First, we recreate the MNIST model from [getting started](../getting_started.md) and load its saved inference weights. The architecture, wrapper representation and preprocessing must match the saved model:

```python
import torch

from CTRAIN.data_loaders import load_mnist
from CTRAIN.model_definitions import CNN7_Shi
from CTRAIN.model_wrappers import ShiIBPModelWrapper

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_, test_loader = load_mnist(batch_size=128, val_split=False)
wrapped_model = ShiIBPModelWrapper(
    model=CNN7_Shi(in_shape=(1, 28, 28)),
    input_shape=(1, 28, 28),
    eps=0.1,
    num_epochs=70,
    warm_up_epochs=1,
    ramp_up_epochs=40,
    lr_decay_kwargs={"milestones": [50, 60], "gamma": 0.2},
    device=device,
)
wrapped_model.load_state_dict(
    torch.load("mnist_ibp.pt", map_location=device, weights_only=True),
)
wrapped_model.eval()
```

The loader supplies the normalisation metadata and feasible input limits. CTRAIN converts `eps` from raw pixel units and clips each input box accordingly. Use evaluation transforms and a fixed sample order when comparing methods. See [data coordinates](data_and_models.md#input-coordinates-and-loader-metadata) for custom loaders.

## Incomplete Evaluation and Attacks

Now, we evaluate the first 1,000 test examples using CROWN and the default PGD attack:

```python
std_acc, cert_acc, adv_acc = wrapped_model.evaluate(
    test_loader,
    test_samples=1000,
    eval_method="CROWN",
)
print("Natural / certified / adversarial:", std_acc, cert_acc, adv_acc)
```

The certification method is selected through `eval_method`:

| Method | Bound computation |
|---|---|
| `"IBP"` | Independent interval bounds. |
| `"CROWN-IBP"` | Backward CROWN bounds with IBP intermediate bounds. |
| `"CROWN"` | Backward linear-relaxation bounds. |
| `"ALPHA-CROWN"` | Optimised CROWN relaxation slopes. |
| `"FORWARD"` | Forward LiRPA bounds with same-slope ReLU relaxations. |
| `"ADAPTIVE"` | Apply bounding methods to examples left unresolved by earlier methods. |

Most wrappers default to `"ADAPTIVE"`; set-based wrappers default to `"FORWARD"`. `"ZONOTOPE"` remains an alias for forward evaluation. The lower-level evaluation API also accepts a list of bounding methods for an adaptive sequence.

The combined evaluator's PGD defaults are 100 steps, 30 restarts, relative step size `0.1` and early stopping. These evaluation settings are separate from the attacks used during training. Lower-level attack and evaluation functions expose their own budgets. Evaluation preserves caller modes, buffers, parameter-gradient flags and existing gradients, including on failure.

### Alpha-CROWN and AutoAttack

Alpha-CROWN uses auto_LiRPA's `CROWN-Optimized` method. Its optimisation budget is controlled by `alpha_iterations`, defaulting to 20. AutoAttack can replace PGD for adversarial evaluation:

```python
std_acc, cert_acc, adv_acc = wrapped_model.evaluate(
    test_loader,
    test_samples=1000,
    eval_method="ALPHA-CROWN",
    alpha_iterations=50,
    adversarial_method="AUTOATTACK",
    autoattack_kwargs={"version": "standard", "seed": 0},
)
```

AutoAttack uses the pinned official implementation installed by `ctrain-install-git-deps`. It attacks in pixel coordinates and applies normalisation inside the attacked model. Its radius must be a uniform scalar pixel-space radius. The `version`, `seed` and `verbose` options are configurable, and randomised outcomes can depend on batch partitioning. Attack-based accuracies remain empirical upper bounds.

## Complete Verification

For examples unresolved by incomplete bounds and attacks, complete verification invokes abCROWN or NeuralSAT. Both backends share input capture, ONNX export, VNNLIB generation, screening and result persistence:

```python
std_acc, cert_acc, adv_acc = wrapped_model.evaluate_complete(
    test_loader,
    test_samples=10,
    engine="abcrown",
    timeout=300,
    results_path="./verification/mnist_abcrown",
)
```

Use `engine="neuralsat"` to select NeuralSAT and a separate results directory for that run. The timeout applies to each unresolved solver query. Complete verification can still return unresolved examples when a query times out or the solver cannot finish. See the [backend guide](../complete_verification.md) for configuration and device behaviour.

Use a sequential loader with `drop_last=False` and evaluation transforms. The evaluator captures the transformed inputs so that screening and solver calls concern the same images. Results and backend logs are written to `results_path` under the configured `results_filename`.

### Continuing and Splitting an Evaluation

A `.manifest.json` sidecar associates cached results with the exported model, transformed inputs, labels, class count, radius, limits and engine. `warm_start=True` checks this identity and rejects incompatible or legacy results without a manifest. Preserve it together with the results.

The manifest does not record every solver budget or configuration option. Use a fresh results directory when changing those settings or retrying cached timeouts. A warm start retains existing timeout records rather than treating a longer timeout as a new query automatically.

The `start_idx` and exclusive `end_idx` arguments select a slice within the `test_samples` prefix and retain global sample indices. Certified and adversarial accuracies then concern that slice. The current wrapper's standard accuracy still concerns the complete prefix, so these three values do not share a denominator for a partial slice. Aggregate the per-input JSON records when combining chunked runs.

Per-input records include the method, running time and result. `unsat` denotes a certified query, while `sat` records counterexample status; `None`, `unknown` and `timeout` remain unresolved according to the evaluation stage. Preserve logs when investigating backend failures and distinguish a partial result file from a completed evaluation.

## Numerical and Property Semantics

Incomplete verification uses floating-point LiRPA enclosures without a directed-rounding arithmetic proof. Complete verification concerns the exported ONNX graph under the solver's numerical assumptions. Export parity checks compare sampled outputs; they do not prove graph equivalence throughout every perturbation region. Preserve export warnings and check returned counterexamples on the original model when comparing graphs.

General-property certification returns a structured report rather than classifier accuracies. It records whole-domain and per-property results, covering cells, lower margins and outcomes. Every covering requirement must have a positive lower bound or a proof for the exported graph. A solver `sat` response without an accessible, checked property witness remains unresolved. See [property certification](../properties.md#certification-and-complete-verification).

When reporting results, include the dataset split, sample count and order, preprocessing, nominal radius, verification method, attack settings and solver budgets. The [evaluation tutorial](../examples/model_evaluation.ipynb), [evaluation API](../api/eval/eval.md), [abCROWN API](../api/complete_verification/abCROWN.md) and [NeuralSAT API](../api/complete_verification/neuralsat.md) provide further details.
