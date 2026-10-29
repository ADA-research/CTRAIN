# Examples

The following notebooks introduce CTRAIN through short, step-by-step workflows. Each follows the same pattern: import the required functions, load data, construct a model and wrapper, then train or evaluate it.

Start with [certified training](certified_training.ipynb), which trains CNN7 on MNIST and saves `mnist_ibp.pt`. The [model-evaluation notebook](model_evaluation.ipynb) loads that file and evaluates its natural, certified and adversarial accuracy. Run both from the same working directory, or adjust the checkpoint path.

[Hyperparameter optimisation](hyperparameter_optimisation.ipynb) uses a MNIST validation holdout to search for natural/certified accuracy trade-offs and evaluate a selected model. These notebooks use real data and training budgets; a CUDA GPU is recommended for training and HPO.

Three further examples introduce training beyond classification robustness:

- [General output properties](general_properties.ipynb): train a small controller with bounded actions over an input box.
- [ONNX and VNNLIB](onnx_vnnlib.ipynb): load your own network and specification, then train with an output-preservation task loss.
- [MNIST arithmetic](mnist_arithmetic.ipynb): train independently predicted digit values and their sum with an affine consistency property.

Install CTRAIN and its pinned Git dependencies as described in [getting started](../getting_started.md), then open a notebook using the same Python environment as its kernel. Data is stored under `./data` relative to the working directory. The ONNX example requires caller-supplied model and specification files.

The notebooks focus on the basic library interface. The [guides](../guides/training.md) and method pages explain checkpoint continuation, detailed configuration, complete verification and larger studies. MkDocs renders the notebooks without executing them. Automated tests use small offline fixtures and reduced budgets separately from these public examples.
