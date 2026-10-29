# Getting Started

CTRAIN provides model wrappers that encapsulate ordinary PyTorch networks and expose certified training, evaluation and hyperparameter optimisation through a consistent interface. In this guide, we install the library and train a first model, following the same workflow used throughout CTRAIN: load a dataset, define a neural network, wrap it with a training method, and evaluate the resulting model.

## Installation

First, install CTRAIN in the Python environment in which you intend to run your experiments:

```bash
python -m pip install CTRAIN
ctrain-install-git-deps
```

The second command installs the pinned Git-hosted dependencies used by CTRAIN, including `auto_LiRPA`, `onnx2pytorch` and AutoAttack, and requires network access. If you use Jupyter, select the same Python environment as the notebook kernel.

For development, run the following commands from the root of a CTRAIN repository checkout. The editable installation allows changes to the library's source code to be used directly and includes the testing and documentation tools.

```bash
git submodule update --init --recursive
python -m pip install -e ".[dev]"
ctrain-install-git-deps
```

The reference CPU test environment uses Python 3.11, PyTorch 2.3.1 and torchvision 0.18.1. Your PyTorch and torchvision installations should match each other and the intended CPU or CUDA environment. The optional NeuralSAT backend requires Python 3.10 or newer. See [dependency troubleshooting](guides/troubleshooting.md#dependency-and-interpreter-mismatches) for help with installation or import errors.

## Certified Training with CTRAIN

In the following example, we train the CNN7 architecture proposed by Shi et al. on MNIST using [IBP with Shi-style improvements](ibp.md). We want the network to be certifiably robust against ℓ∞ perturbations with radius ε = 0.1 around its inputs. This example trains on the real MNIST dataset; a CUDA GPU is recommended for the full training run.

### Loading Data and Defining the Model

First, we import PyTorch and the required functions from CTRAIN:

```python
import torch

from CTRAIN.data_loaders import load_mnist
from CTRAIN.model_definitions import CNN7_Shi
from CTRAIN.model_wrappers import ShiIBPModelWrapper
```

Now, we load MNIST and construct the network. Each input is a single-channel image of size 28 × 28 pixels, and the model predicts one of ten digit classes. The device is selected according to whether CUDA is available.

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_shape = (1, 28, 28)

train_loader, test_loader = load_mnist(
    batch_size=128,
    val_split=False,
    data_root="./data",
)
model = CNN7_Shi(in_shape=input_shape, n_classes=10)
```

The loader downloads missing MNIST data to `data_root` by default. It also supplies input limits and normalisation metadata, which CTRAIN uses to construct perturbation regions. Thus, `eps=0.1` is specified in raw pixel units even though the loader normalises the images. Further details on datasets, custom loaders and normalisation are provided in the [data and models guide](guides/data_and_models.md).

### Configuring and Training the Wrapper

To train the network certifiably, we wrap it with `ShiIBPModelWrapper`. Under the hood, the wrapper converts the PyTorch model into a bounded LiRPA graph and passes the training configuration to the library's trainer.

We configure 70 training epochs, including one warm-up epoch on clean cross-entropy and 40 ramp-up epochs during which the perturbation radius gradually increases to its target. Training then continues at the full radius. The learning rate is reduced by a factor of 0.2 at epochs 50 and 60.

```python
wrapped_model = ShiIBPModelWrapper(
    model=model,
    input_shape=input_shape,
    eps=0.1,
    num_epochs=70,
    warm_up_epochs=1,
    ramp_up_epochs=40,
    lr_decay_kwargs={"milestones": (50, 60), "gamma": 0.2},
    device=device,
)
```

We initiate training by calling the wrapper's `train_model` method:

```python
wrapped_model.train_model(train_loader)
```

With the loaders used here, training returns the final weights. To select a model using validation data, load MNIST with `val_split=True`, which returns training, validation and test loaders, and pass the validation loader through `val_loader`. The [training guide](guides/training.md) explains schedules, regularisation and shared configuration options.

### Evaluating the Model

Finally, we set the model to evaluation mode and assess its natural, certified and adversarial accuracy. Here, we use IBP for certification and PGD for adversarial evaluation on the first 1,000 test images:

```python
wrapped_model.eval()
std_acc, cert_acc, adv_acc = wrapped_model.evaluate(
    test_loader,
    test_samples=1_000,
    eval_method="IBP",
)

print(f"Standard accuracy: {std_acc:.2%}")
print(f"Certified accuracy: {cert_acc:.2%}")
print(f"Adversarial accuracy: {adv_acc:.2%}")
```

The returned values are fractions, displayed here as percentages. Certified accuracy measures the proportion of evaluated inputs for which robustness is established throughout the perturbation region. Adversarial accuracy measures the proportion that remain correctly classified after the PGD search. These provide lower and upper bounds, respectively, on robust accuracy for the evaluated samples.

CTRAIN also supports tighter incomplete verification methods and complete verification through α,β-CROWN and NeuralSAT. See the [evaluation guide](guides/evaluation.md) and [model evaluation tutorial](examples/model_evaluation.ipynb) for these workflows.

### Saving the Model

We can save the trained wrapper's inference weights using the standard PyTorch interface:

```python
torch.save(wrapped_model.state_dict(), "mnist_ibp.pt")
```

To use these weights in a later session, recreate the same architecture and wrapper configuration, then load the saved state dictionary:

```python
wrapped_model.load_state_dict(
    torch.load("mnist_ibp.pt", map_location=device, weights_only=True)
)
wrapped_model.eval()
```

Keep the architecture, input normalisation and perturbation radius alongside the weights. To resume training with its optimiser and learning-rate scheduler, use trajectory checkpoints instead; the [checkpoint guide](guides/checkpoints.md) explains how to configure saving and restoration.

## Further Examples

The same workflow applies to the other [certified training methods](index.md#overview): choose the corresponding model wrapper and configure its method-specific hyperparameters. For example, the [expressive losses](expressive_losses.md) combine adversarial and IBP objectives, while [CC-Dist](cc_dist.md) additionally uses a pretrained teacher.

The following tutorials demonstrate CTRAIN's functionality in more detail:

- [Certified training](examples/certified_training.ipynb): Loading MNIST, configuring a wrapper, training and saving weights.
- [Model evaluation](examples/model_evaluation.ipynb): Loading the training example's weights and evaluating natural, certified and adversarial accuracy.
- [Hyperparameter optimisation](examples/hyperparameter_optimisation.ipynb): Searching for configurations and selecting models from their Pareto fronts.
- [General output properties](examples/general_properties.ipynb): Training a small controller with bounded outputs over a declared input box.
- [ONNX and VNNLIB](examples/onnx_vnnlib.ipynb): Importing networks and specifications for training and repair.
- [MNIST arithmetic](examples/mnist_arithmetic.ipynb): Training predicted digit values and their sum with an affine consistency property.

The MNIST notebooks use real data. Run certified training before model evaluation, which loads its saved weights. The ONNX/VNNLIB notebook requires your own model and specification files. More detailed configuration and complete-verification workflows are covered in the guides.

## Development and Local Documentation

With the development and Git dependencies installed, run the CPU test suite from the repository root:

```bash
python -m pytest -q
```

The suite includes isolated notebook smoke checks with offline fixtures and reduced training budgets. Optional CUDA and complete-verifier checks are described in the repository's `tests/README.md`.

To view the documentation locally, start the MkDocs server from the same directory:

```bash
mkdocs serve
```

The documentation build renders the notebooks without executing their training or verification cells.
