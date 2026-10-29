# Complete Verification

To establish whether a trained network satisfies a desired property, CTRAIN provides interfaces to the complete verification systems [α,β-CROWN](https://github.com/Verified-Intelligence/alpha-beta-CROWN) and [NeuralSAT](https://github.com/dynaroars/neuralsat). These systems combine bound computations with search, allowing them to resolve properties that remain undecided by inexpensive incomplete methods. A finite computational budget can still leave a query unresolved.

Both backends are available through the common `evaluate_complete` method. CTRAIN prepares the network and specifications, performs initial screening, invokes the selected verifier and records the results. This workflow can also be used for models trained outside CTRAIN, provided their architecture and preprocessing are reconstructed correctly.

## Verification Workflow

For local robustness, the verification property requires the network to predict the correct class throughout a perturbation box around each evaluated input. CTRAIN constructs these boxes using the configured `eps`, loader normalisation metadata and input limits.

The evaluator first identifies clean classification errors, then applies IBP and CROWN to the remaining inputs and uses PGD to search for adversarial examples. Only unresolved queries are passed to the complete backend. Thus, inputs that can be certified or falsified cheaply do not require a full solver invocation.

To interface with the complete backends, CTRAIN exports the network to ONNX and writes input regions and output constraints in VNNLIB format. The selected verifier receives these files for the unresolved queries. Export includes numerical comparisons with the internal model representation. Verification concerns the exported graph under the solver's numerical assumptions; these comparisons are checks on sampled inputs rather than a proof of equivalence over the whole input region.

## Preparing the Model

The following example evaluates the MNIST model saved as `mnist_ibp.pt` in the [getting-started guide](getting_started.md#saving-the-model). First, we load the test data and recreate the architecture and wrapper configuration used for training.

```python
import torch

from CTRAIN.data_loaders import load_mnist
from CTRAIN.model_definitions import CNN7_Shi
from CTRAIN.model_wrappers import ShiIBPModelWrapper

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_shape = (1, 28, 28)
train_loader, test_loader = load_mnist(
    batch_size=128, val_split=False, data_root="./data", workers=0
)

wrapped_model = ShiIBPModelWrapper(
    model=CNN7_Shi(in_shape=input_shape, n_classes=10),
    input_shape=input_shape,
    eps=0.1,
    num_epochs=70,
    warm_up_epochs=1,
    ramp_up_epochs=40,
    lr_decay_kwargs={"milestones": (50, 60), "gamma": 0.2},
    device=device,
)
wrapped_model.load_state_dict(
    torch.load("mnist_ibp.pt", map_location=device, weights_only=True)
)
wrapped_model.eval()
```

The saved file contains wrapper inference weights, so it is loaded after constructing the bounded model. Native PyTorch weights are instead loaded into the architecture before wrapping. In either case, the architecture, preprocessing, class count and perturbation radius must match the intended experiment; the [model evaluation tutorial](examples/model_evaluation.ipynb) explains the supported checkpoint formats.

Complete verification requires a sequential evaluation loader with a defined batch size and `drop_last=False`, using evaluation transforms. CTRAIN's test loaders provide this configuration. The shared evaluator captures the transformed inputs so that screening and solver calls use the same images and input regions.

## Verification Backends

### abCROWN

α,β-CROWN combines optimised linear-relaxation bounds with branch-and-bound search. Select it through `engine="abcrown"`, which is also the default. Here, we evaluate the first ten test images with a solver budget of 300 seconds per unresolved input.

Choose a fresh `results_path` for a new experiment. To continue an existing evaluation, use the warm-start options described below.

```python
std_acc, cert_acc, adv_acc = wrapped_model.evaluate_complete(
    test_loader,
    engine="abcrown",
    test_samples=10,
    timeout=300,
    abcrown_batch_size=128,
    results_path="./verification/mnist_abcrown",
)
```

The `abcrown_batch_size` controls the solver's batch size and can be reduced when memory is limited. Additional backend settings can be supplied as a nested `abcrown_config_dict`, following the [upstream configuration documentation](https://github.com/Verified-Intelligence/alpha-beta-CROWN). The `no_cores` argument controls parallel MIP solvers when that functionality is enabled. The bundled adapter selects the available compute device automatically.

### NeuralSAT

NeuralSAT combines neural network abstraction with DPLL(T)-based search. To evaluate the same model and input prefix with this backend, select `engine="neuralsat"` and use a separate result directory:

```python
std_acc, cert_acc, adv_acc = wrapped_model.evaluate_complete(
    test_loader,
    engine="neuralsat",
    test_samples=10,
    timeout=300,
    neuralsat_batch_size=1000,
    results_path="./verification/mnist_neuralsat",
)
```

The `neuralsat_batch_size` controls the solver's maximum parallel branches. NeuralSAT uses the wrapper's CPU or CUDA device. Its adapter launches the solver in a subprocess and includes process startup in the timeout. An `abcrown_config_dict` cannot be used with this backend.

The `timeout` applies to each solver query, rather than to the complete evaluation. The total running time also includes screening, export and setup, and depends on how many inputs require complete verification.

## Interpreting Results

Both backends return the tuple `(standard_accuracy, certified_accuracy, adversarial_accuracy)`, with values expressed as fractions:

```python
print(f"Standard accuracy: {std_acc:.2%}")
print(f"Certified accuracy: {cert_acc:.2%}")
print(f"Adversarial accuracy: {adv_acc:.2%}")
```

Certified accuracy counts inputs whose robustness has been established. Adversarial accuracy counts inputs for which no violation has been found, including unresolved cases. These provide lower and upper bounds, respectively, on robust accuracy for the evaluated inputs. Matching bounds indicate that every input in the evaluated set has been resolved.

CTRAIN additionally saves per-input records under `results_path/results_filename`, where `results_filename` defaults to `results.json`. Each record includes a result, the method that produced it and the accumulated running time. The result statuses have the following meanings:

| Result | Meaning |
|---|---|
| `"unsat"` | The counterexample query is unsatisfiable; the property is certified. |
| `"sat"` | A clean error, adversarial attack or solver result reports a property violation. |
| `"timeout"` | The query exhausted its computational budget and remains unresolved. |
| `"unknown"` or `None` | The query remains unresolved, including unsuccessful solver executions. |

The `method` field distinguishes screening results from backend results. Logs are saved under `abCROWN_logs` or `neuralsat_logs` within the result directory. Inspect these records and logs when investigating unresolved queries or solver-reported counterexamples; see the [evaluation guide](guides/evaluation.md#numerical-and-property-semantics) for numerical and witness semantics.

## Resuming and Splitting Evaluations

Set `warm_start=True` to reuse an existing result file. Keep the model, loader order, preprocessing, radius, engine and evaluated input prefix unchanged. A sidecar file, `results.json.manifest.json` by default, binds the saved records to the exported model, captured inputs, labels and perturbation region. Mismatches and legacy result files without a manifest are rejected.

The identity manifest does not capture every solver setting. Use a new result directory when changing budgets or backend configuration. Cached timeout records are retained on a warm start, so retrying them with a larger budget also requires a fresh evaluation.

For longer runs, `start_idx` and `end_idx` select a half-open slice of the prefix defined by `test_samples`, retaining global input indices in the result file. Keep `test_samples` fixed when processing several chunks of the same evaluation. The certified and adversarial accuracies returned for a chunk refer to that slice; the wrapper's standard accuracy is still computed over the full `test_samples` prefix. Use the per-input records when aggregating chunk results.

## Backend Setup

Install CTRAIN and its pinned Git dependencies as described in the [installation guide](getting_started.md#installation). Both complete backends are bundled with the library; their source provenance and licences are retained in `CTRAIN/verification_systems`.

NeuralSAT's bundled runtime requires Python 3.10 or newer and runs with the active interpreter by default. Its own auto_LiRPA and onnx2pytorch forks are isolated in the solver subprocess. Keep CTRAIN's training dependencies in place; the bundled runtime does not require installing NeuralSAT's upstream requirements over them. Its licence probe disables optional MIP functionality when no usable Gurobi licence is available.

To run NeuralSAT from a separate checkout or Python environment, provide an argument-list prefix through `neuralsat_command`, for example `['/path/venv/bin/python', '/path/neuralsat/src/main.py']`. This changes the solver process while retaining CTRAIN's shared export, screening and result handling.

## Implementation and API Reference

The common evaluator handles ONNX export, VNNLIB generation, screening, result persistence and backend dispatch. abCROWN and NeuralSAT adapters handle their respective invocation and status reporting. General output specifications use the separate [property-certification interface](properties.md#certification-and-complete-verification).

The API reference documents the [evaluation functions](api/eval/eval.md), [wrapper interface](api/model_wrappers/ctrain_wrapper.md), [abCROWN integration](api/complete_verification/abCROWN.md) and [NeuralSAT integration](api/complete_verification/neuralsat.md). The [model evaluation tutorial](examples/model_evaluation.ipynb) provides an interactive workflow with optional solver calls.
