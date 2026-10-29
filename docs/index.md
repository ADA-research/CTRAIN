# CTRAIN

CTRAIN is a unified, modular and comprehensive Python library for certified training and neural network verification. It provides implementations of state-of-the-art certified training methods through a consistent interface, facilitating their application, comparison and extension. Based on PyTorch and `auto_LiRPA`, CTRAIN integrates into existing machine learning workflows and supports configuring, training and formally evaluating neural networks.

## Overview

CTRAIN integrates multiple state-of-the-art certified training approaches, which are:

- [Interval Bound Propagation (IBP)](ibp.md) ([Gowal et al., 2018](https://arxiv.org/abs/1810.12715) with the improvements of [Shi et al., 2021](https://proceedings.neurips.cc/paper/2021/hash/988f9153ac4fd966ea302dd9ab9bae15-Abstract.html))
- [CROWN-IBP](crown_ibp.md) ([Zhang et al., 2020](https://arxiv.org/abs/1906.06316))
- [SABR](sabr.md) ([Müller et al., 2023](https://arxiv.org/pdf/2210.04871))
- [TAPS](taps_staps.md#taps) ([Mao et al., 2023](https://proceedings.neurips.cc/paper_files/paper/2023/file/e8b0c97b34fdaf58b2f48f8cca85e76a-Paper-Conference.pdf))
- [STAPS](taps_staps.md#staps) (combination of SABR and TAPS, [Mao et al., 2023](https://proceedings.neurips.cc/paper_files/paper/2023/file/e8b0c97b34fdaf58b2f48f8cca85e76a-Paper-Conference.pdf))
- [MTL-IBP, CC-IBP and Exp-IBP](expressive_losses.md), the family of expressive losses combining adversarial and certified objectives ([De Palma et al., 2024](https://arxiv.org/pdf/2305.13991))
- [CC-Dist](cc_dist.md), combining CC-IBP with feature distillation from an adversarially trained teacher ([De Palma, 2026](https://arxiv.org/pdf/2602.02626v1))
- [IBP-R](ibp_r.md), augmenting adversarial training with an IBP-based regulariser ([De Palma et al., 2022](https://arxiv.org/pdf/2206.14772))
- [Gaussian loss smoothing](gls.md), enabling training with tighter bounding methods through Randomized Gradient Smoothing (RGS) or Policy Gradients with Parameter-based Exploration (PGPE) ([Balauca et al., 2025](https://arxiv.org/pdf/2403.07095))
- [Set-based training](set_based.md), propagating zonotopes through the network to optimise reachable output enclosures ([Koller et al., 2025](https://arxiv.org/abs/2401.14961))

These methods share implementations of bound propagation, loss calculation, adversarial attacks and regularisation. This modular design allows researchers to investigate new approaches by modifying individual components without having to reimplement existing baselines from scratch.

Beyond local robustness, CTRAIN supports training for [general input-output properties](properties.md), expressed as Boolean combinations of affine output constraints over input boxes. By formulating these properties through satisfaction margins, analogous to classification margins, CTRAIN enables adaptations of existing certified training methods to more general specifications, such as output-range constraints or relations between multiple predictions.

Furthermore, CTRAIN enables easy evaluation of the adversarial and certified robustness of a given neural network, using the following methods:

- [PGD](api/attacks/pgd.md) ([Madry et al., 2018](https://arxiv.org/abs/1706.06083))
- [AutoAttack](api/eval/eval.md#CTRAIN.eval.autoattack.eval_autoattack), an ensemble of complementary adversarial attacks ([Croce and Hein, 2020](https://github.com/fra31/auto-attack))
- [IBP](api/eval/eval.md#CTRAIN.eval.eval.eval_ibp) ([Gowal et al., 2018](https://arxiv.org/abs/1810.12715))
- [CROWN-IBP](api/eval/eval.md#CTRAIN.eval.eval.eval_crown_ibp) ([Zhang et al., 2020](https://arxiv.org/abs/1906.06316))
- [CROWN](api/eval/eval.md#CTRAIN.eval.eval.eval_crown) ([Zhang et al., 2018](https://arxiv.org/abs/1811.00866))
- [α-CROWN](api/eval/eval.md#CTRAIN.eval.eval.eval_alpha_crown) ([Xu et al., 2021](https://arxiv.org/pdf/2011.13824))
- [α,β-CROWN](complete_verification.md#abcrown) ([Xu et al., 2021](https://arxiv.org/pdf/2011.13824), [Wang et al., 2021](https://arxiv.org/abs/2103.06624))
- [NeuralSAT](complete_verification.md#neuralsat) ([Duong et al., 2025](https://doi.org/10.1007/978-3-031-98679-6_19)), a complete neural network verification system based on the DPLL(T) framework

For incomplete verification, users can select a single method or apply methods adaptively in increasing order of computational cost. Complete verification first attempts to resolve inputs using incomplete methods and PGD before invoking α,β-CROWN or NeuralSAT. The evaluation functions can also be used independently of CTRAIN's training workflow to assess externally trained models.

## Key Features

CTRAIN has the goal of providing a unified, comprehensive, accessible and flexible framework for certified training. Key features include:

- Standardised, modular and highly configurable implementations of popular and performant certified training methods based on the popular `auto_LiRPA` library
- Unified and accessible interface through model wrappers
- Based on PyTorch for easy integration into common machine learning pipelines
- Native support for hyperparameter optimisation with preconfigured, method-specific search spaces, including single-objective optimisation using SMAC3 ([Lindauer et al., 2022](https://www.jmlr.org/papers/volume23/21-0888/21-0888.pdf)) and multi-objective optimisation using Optuna/BoTorch as proposed by Kaulen et al., 2026 (see below)
- Training and verification for general input-output properties beyond local robustness
- Comprehensive evaluation tools for adversarial and certified robustness
  - Support for both incomplete and complete verification methods, including the state-of-the-art complete verification systems α,β-CROWN and NeuralSAT
- Detailed documentation with API reference, setup guide, and usage examples
- Open-source with a permissive MIT license
- Active development and maintenance by the Chair for Artificial Intelligence Methodology at RWTH Aachen University

## Installation and Quick Start

Install CTRAIN and its pinned Git dependencies (including auto_LiRPA):

```sh
pip install CTRAIN
ctrain-install-git-deps
```

For development, run these commands from a checkout of the CTRAIN repository:

```sh
git submodule update --init --recursive
pip install -e ".[dev]"
ctrain-install-git-deps
```

The dependency installer obtains the pinned versions of `auto_LiRPA`, `onnx2pytorch` and AutoAttack used by CTRAIN. Further setup instructions are provided in the [installation guide](getting_started.md#installation).

Then, you can train and evaluate the standard CNN7 architecture proposed by Shi et al. on the CIFAR-10 dataset using IBP in a few lines of code:

```python
from CTRAIN.model_definitions import CNN7_Shi
from CTRAIN.data_loaders import load_cifar10
from CTRAIN.model_wrappers import ShiIBPModelWrapper

train_loader, test_loader = load_cifar10(val_split=False)
in_shape = [3, 32, 32]

model = CNN7_Shi(in_shape=in_shape)
wrapped_model = ShiIBPModelWrapper(model=model, input_shape=in_shape, eps=2/255, num_epochs=160)

wrapped_model.train_model(train_loader)
wrapped_model.eval()
std_acc, cert_acc, adv_acc = wrapped_model.evaluate(test_loader)
```

The model wrapper encapsulates the PyTorch network and exposes training through `train_model`, evaluation through `evaluate` or `evaluate_complete`, and hyperparameter optimisation through `hpo`. To use another certified training method, choose the corresponding wrapper and its method-specific hyperparameters.

## Project Structure

```text
CTRAIN/
├── attacks/               # Adversarial attacks
├── bound/                 # Bound propagation and satisfaction bounds
├── complete_verification/ # Complete verification interfaces
├── data_loaders/          # Dataset loading and normalisation metadata
├── eval/                  # Empirical and certified evaluation
├── model_definitions/     # Neural network architectures
├── model_wrappers/        # Training, evaluation and HPO interfaces
├── properties/            # General input-output specifications
├── train/                 # Training procedures, losses and regularisers
├── util/                  # Shared utility functions
└── verification_systems/  # External verification tools
```

## Documentation

The documentation includes installation instructions, usage examples and an API reference.

### Getting started and guides

The following guides provide more detailed information about the different functionalities of CTRAIN:

- [Getting started](getting_started.md): Installation instructions and a first runnable workflow.
- [Tutorials](examples/README.md): Step-by-step examples of certified training, model evaluation, hyperparameter optimisation and training for general output properties.
- [Data and models](guides/data_and_models.md): Data loading, input shapes, loader metadata and normalisation.
- [Training](guides/training.md): Training schedules, shared configuration options and BatchNorm behaviour.
- [Checkpoints and model selection](guides/checkpoints.md): Saving and restoring models, training state and validation-based model selection.
- [Hyperparameter optimisation](guides/hpo.md): Running and resuming optimisation studies and selecting models from their Pareto fronts.
- [Evaluation and verification](guides/evaluation.md): Evaluating adversarial and certified robustness using attacks, incomplete certification and complete verification with α,β-CROWN and NeuralSAT.
- [General output properties](properties.md): Defining and training against affine and Boolean output constraints over input boxes.
- [Troubleshooting](guides/troubleshooting.md): Common issues related to dependencies, model export and numerical computations.

### API Reference

The API reference provides detailed documentation of CTRAIN's classes, functions and configuration options. In particular, it covers the [model wrappers](api/model_wrappers/ctrain_wrapper.md), [model definitions](api/model_definitions/model_definitions.md), [data loaders](api/data_loaders/data_loaders.md) and [evaluation functions](api/eval/eval.md).

The reference documentation is generated directly from the source code and its docstrings. For explanations of how the different components interact and how to configure complete workflows, refer to the guides above.

## ICML 2026 Paper and Published Models

CTRAIN includes the code and result artifacts for the ICML 2026 paper **“Rethinking Evaluation Paradigms in IBP-based Certified Training”**, which compares certified-training methods through their natural-versus-certified accuracy Pareto fronts. The final, completely verified per-method fronts are available as [published checkpoints](https://huggingface.co/kkaulen/ctrain_pareto_fronts).

Performance on clean data and certification of desired properties usually form a trade-off controlled by hyperparameter settings. CTRAIN implements the paper's multi-objective optimisation procedure through the wrappers' `hpo` method, returning a set of Pareto-optimal configurations from which users can select the model that best fits their application. See the [hyperparameter optimisation tutorial](examples/hyperparameter_optimisation.ipynb) for an example.

The [published-model example](https://github.com/ADA-research/CTRAIN/blob/main/papers/rethinking_evaluation_paradigms/examples/evaluate_mtl_front.py) shows how to inspect a published MTL-IBP front, download a model and evaluate it on CIFAR-10. The full paper code, results and reproduction instructions are provided in the [paper artifacts](https://github.com/ADA-research/CTRAIN/tree/main/papers/rethinking_evaluation_paradigms).

## Implementation and Reproducibility

CTRAIN standardises the shared components of certified training to facilitate comparison between methods. Reproducing a published experiment additionally requires its architecture, preprocessing, data splits, hyperparameters and evaluation protocol. The individual method pages linked in the [overview](#overview) describe the implemented objectives and differences from their source implementations.

## License

This project is licensed under the [MIT License](https://github.com/ADA-research/CTRAIN/blob/main/LICENSE).

## Maintainers

This project is developed at the Chair for Artificial Intelligence Methodology at RWTH Aachen University by Konstantin Kaulen under the supervision of Prof. Holger H. Hoos.

Konstantin Kaulen is the current core-maintainer of the project.

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## Acknowledgements

This project incorporates the α,β-CROWN verifier which is developed by a multi-institutional team led by Prof. Huan Zhang. The α,β-CROWN components are licensed under the BSD 3-Clause license.

Furthermore, CTRAIN depends heavily on the `auto_LiRPA` library, which is developed by the same team and also licensed under the BSD 3-Clause License.

CTRAIN also integrates [NeuralSAT](https://github.com/dynaroars/neuralsat), a complete neural network verifier developed by a team led by Hai Duong and ThanhVu Nguyen. NeuralSAT is licensed under the MIT License.
