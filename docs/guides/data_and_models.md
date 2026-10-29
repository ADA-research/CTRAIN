# Data and Models

CTRAIN separates data loading and model construction from the training method. A dataset loader supplies ordinary PyTorch batches together with the input limits and normalisation metadata required to construct perturbation regions. The model defines the network architecture, while the wrapper selects the training objective and its configuration.

In this guide, we load an image dataset, choose an architecture and explain how input coordinates are handled during training and evaluation. For a complete first training run, see [getting started](../getting_started.md).

## Image Loaders

CTRAIN provides shared loaders for MNIST, CIFAR-10, SVHN and TinyImageNet. Each returns batches of `(images, labels)` and can be used with the classification training methods. The trainers and losses construct the required input sets from these batches.

First, we load CIFAR-10 with a deterministic training holdout:

```python
from CTRAIN.data_loaders import load_cifar10

train_loader, val_loader, test_loader = load_cifar10(
    batch_size=128,
    data_root="./data",
    validation_fraction=0.2,
    seed=0,
)
```

By default, the loader reserves 20% of the training data for validation and returns three loaders. Training uses the configured augmentation, while validation and test images use evaluation transforms. The local split seed makes the holdout repeatable without resetting the global PyTorch random state.

When no validation split is needed, `val_split=False` returns two loaders instead:

```python
train_loader, test_loader = load_cifar10(
    batch_size=128,
    val_split=False,
    data_root="./data",
)
```

The explicit `selection_split` option provides a common three-entry return format. `"validation"` creates a training holdout, `"test"` uses the test split for selection, and `"none"` returns `None` as the second entry. These settings override `val_split`. Test-based selection is useful when following a released reproduction protocol, but should be reported as such. For a new experiment, reserve a training holdout for selection and hyperparameter optimisation, and use the test split for final assessment.

### Downloading and Preprocessing

Missing data is downloaded by default. Use `download=False` to require an existing local dataset, and `workers` to choose the number of data-loading workers. Defaults are two workers, except for TinyImageNet training and test loaders, which use four.

The `train_transforms` argument supplies training augmentation. Optional `input_size` applies a final centre crop after training transforms and a centre crop to evaluation images. Custom `normalisation_stats={"mean": [...], "std": [...]}` can be supplied with `normalise=True`; there must be one finite mean and positive standard deviation per channel.

MNIST uses mean `0.1307` and standard deviation `0.3081`. CIFAR-10 and SVHN use means `[0.4914, 0.4822, 0.4465]` and standard deviations `[0.2023, 0.1994, 0.2010]`. SVHN retains these statistics from CTRAIN's earlier reproduction loader. TinyImageNet uses means `[0.4802, 0.4481, 0.3975]` and standard deviations `[0.2302, 0.2265, 0.2262]`.

TinyImageNet loading accepts either the extracted `tiny-imagenet-200` directory or its parent. Both official validation annotations with flat `val/images` storage and validation class-folder layouts are supported. Existing images are read in place.

The `smoke=True` option substitutes two synthetic training and test images before splitting and fixes the batch size to two. It provides a small execution check without downloading data; it does not assess training quality on the original dataset.

## Input Coordinates and Loader Metadata

Classification wrappers accept `eps` in raw pixel units. For normalised images, CTRAIN divides this radius by the channel standard deviations and clips the resulting interval to the loader's feasible input limits. Thus, `eps=2 / 255` retains its pixel-space meaning even when the network receives normalised CIFAR-10 images.

```text
radius_model = eps / std
x_L = max(x - radius_model, data_min)
x_U = min(x + radius_model, data_max)
```

The loader exposes `mean`, `std`, `normalised`, `min` and `max`. The statistics and limits are tensors; image limits have broadcastable shape `(1, channels, 1, 1)`. With `normalise=False`, the statistics are the identity transformation and the limits are `[0, 1]`. Use the radius once in raw units, rather than dividing it before passing it to the wrapper.

Custom loaders should provide the same metadata for the coordinates they actually return. In particular, standard deviations must be finite and positive, and limits must broadcast to the input tensors. When creating another loader for an existing CTRAIN dataset, preserve these metadata fields alongside the dataset and its transforms.

General-property boxes instead use exactly the verification model's coordinates. No pixel scaling or dataset normalisation is inferred for them. See [property domains](../properties.md#training-and-domains) when training beyond classification robustness.

## Model Families

Now, we construct a network matching the dataset. For CIFAR-10, the input shape is `(3, 32, 32)` and the output has ten classes:

```python
from CTRAIN.model_definitions import CNN7_Shi

input_shape = (3, 32, 32)
model = CNN7_Shi(in_shape=input_shape, n_classes=10)
```

Input shapes normally exclude the batch dimension. The model's channel count, spatial dimensions and number of classes must match the data and any checkpoint being loaded. Individual wrapper references document any older batch-shaped conventions.

CTRAIN includes the following model families:

| Models | Use |
|---|---|
| `CNN7_Shi` | Image classification; `downsample_final=True` supplies the TinyImageNet variant. |
| `GowalConvSmall`, `GowalConvMed`, `GowalConvLarge` | Convolutional classifier variants. |
| `CNN3_Mao`, `CNN5_Mao`, `CNN9_Mao`, `CNN11_Mao` | Additional image-classification architectures. |
| `CNN_GLS` | Architecture and initialisation options used with Gaussian loss smoothing. |
| `CNN6_SetBased` | Architecture used with set-based training. |
| `MNISTAdditionClassifier`, `MNISTAdditionRegressor`, `MNISTSwapGraph` | Arithmetic and operand-swapping models for general-property examples. |
| `ControllerMLP`, `ControllerDynamics` | A trainable controller and its fixed dynamics graph. |

Architecture variants are configured through their model definitions. Supported computation graphs depend on the training method and auto_LiRPA operators. For example, [TAPS/STAPS](../taps_staps.md) require a sequential split, [CC-Dist](../cc_dist.md) uses teacher and student feature representations, and [set-based training](../set_based.md) supports a defined collection of sequential layers. Native PGD training has fewer graph restrictions.

## Imported Models and Specifications

Existing networks can also enter the workflow through ONNX. `load_vnnlib_problem` imports a model and its VNNLIB specification, while `load_vnnlib_property` loads a specification for an existing PyTorch model. The ACAS helpers additionally support pinned benchmark artifacts.

These imports preserve explicit input coordinates and record source hashes. By default, VNNLIB output assertions describe unsafe states, so the importer complements them to obtain the desired training property. See [import semantics](../properties.md#user-selected-networks-and-specifications) and the [ONNX/VNNLIB tutorial](../examples/onnx_vnnlib.ipynb) before choosing a task loss for repair.

The API reference documents the [data loaders](../api/data_loaders/data_loaders.md) and [model definitions](../api/model_definitions/model_definitions.md).
