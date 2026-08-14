# CTRAIN

CTRAIN is a unified, modular and comprehensive package for training and evaluating certified training techniques for neural networks.

## Overview

CTRAIN integrates multiple state-of-the-art certified training approaches, which are:

- Interval Bound Propagation (IBP) ([Gowal et al., 2018](https://arxiv.org/abs/1810.12715) with the Improvements of [Shi et al., 2021](https://arxiv.org/abs/2102.06700))
- CROWN-IBP ([Zhang et al., 2020](https://arxiv.org/abs/1906.06316))
- SABR ([Müller et al., 2023](https://arxiv.org/pdf/2210.04871))
- TAPS ([Mao et al., 2023](https://proceedings.neurips.cc/paper_files/paper/2023/file/e8b0c97b34fdaf58b2f48f8cca85e76a-Paper-Conference.pdf))
- STAPS (combination of SABR and TAPS, [Mao et al., 2023](https://proceedings.neurips.cc/paper_files/paper/2023/file/e8b0c97b34fdaf58b2f48f8cca85e76a-Paper-Conference.pdf))
- MTL-IBP ([De Palma et al., 2023](https://arxiv.org/pdf/2305.13991))

Furthermore, CTRAIN enables easy evaluation of the adversarial and certified robustness of a given neural network, using the following methods:

- PGD ([Madry et al., 2018](https://arxiv.org/abs/1706.06083))
- IBP ([Gowal et al., 2018](https://arxiv.org/abs/1810.12715))
- CROWN-IBP ([Zhang et al., 2020](https://arxiv.org/abs/1906.06316))
- CROWN ([Zhang et al., 2018](https://arxiv.org/abs/1811.00866))
- α,β-CROWN ([Xu et al., 2020](https://arxiv.org/pdf/2011.13824), [Wang et al., 2021](https://arxiv.org/abs/2103.06624))

## Key Features
CTRAIN has the goal of providing an unified, comprehensive, accessible and flexible framework for certified training. Key features include:

- Standardised, modular and hihly-configurable implementations of popular and performant certified training methods based on the popular `auto_LiRPA` library
- Unified and accessible interface through model wrappers
- Based on PyTorch for easy integration into common machine learning pipelines
- Seamless integration of sophisticated hyperparameter optimisation using SMAC3 ([Lindauer et al., 2022](https://www.jmlr.org/papers/volume23/21-0888/21-0888.pdf))
- Comprehensive evaluation tools for adversarial and certified robustness
  - Support for both incomplete and complete verification methods, including the state-of-the-art complete verification system α,β-CROWN
- Detailed documentation with API reference, setup guide, and usage examples
- Open-source with a permissive MIT license
- Active development and maintenance by the Chair for Artificial Intelligence Methodology at RWTH Aachen University

## Installation and Quick Start
First, install CTRAIN using `pip`:
```sh
pip install CTRAIN
```
Or, to setup the package for development purposes, run:
```sh
git submodule init
git submodule update
pip install --no-deps git+https://github.com/KaidiXu/onnx2pytorch@8447c42c3192dad383e5598edc74dddac5706ee2
pip install --no-deps git+https://github.com/Verified-Intelligence/auto_LiRPA.git@cf0169ce6bfb4fddd82cfff5c259c162a23ad03c"
pip install -e ".[dev]"
```
Then, you can train and evaluate the standard CNN7 architecture proposed by Shi et al. on the CIFAR-10 dataset using the IBP certified training technique in 12 lines of code:
```python
from CTRAIN.model_definitions import CNN7_Shi
from CTRAIN.data_loaders import load_cifar10
from CTRAIN.model_wrappers import ShiIBPModelWrapper

train_loader, test_loader = load_cifar10(val_split=False)
in_shape = [3, 32, 32]

model = CNN7_Shi(in_shape=in_shape)
wrapped_model = ShiIBPModelWrapper(model=model, input_shape=in_shape, eps=2/255, num_epochs=160)

wrapped_model.train_model(train_loader)
std_acc, cert_acc, adv_acc = wrapped_model.evaluate(test_loader)
```
## Project Structure

```
CTRAIN/
├── attacks/            # Implementation of attacks
├── bound/             # Bound computation approaches
├── complete_verification/ # Complete verification
├── data_loaders/      # Dataset loading utilities  
├── eval/              # (Incomplete) Evaluation functionality
├── model_definitions/ # Neural network architectures
├── model_wrappers/    # Model wrappers for different approaches
├── train/            # Robust/Certified Training implementations
├── util/             # Utility functions
└── verification_systems/ # External verification tools
```

## Documentation

Documentation is available in the docs directory, including:

- API Reference
- Setup Guide
- Usage Examples

## ICML 2026 Paper and Published Models

CTRAIN includes the code and result artifacts for the ICML 2026 paper
**“Rethinking Evaluation Paradigms in IBP-based Certified Training”**, which
compares certified-training methods through their natural-versus-certified
accuracy Pareto fronts. The final, completely verified per-method fronts are
available as
[published checkpoints](https://huggingface.co/kkaulen/ctrain_pareto_fronts).

See the [Pareto-front models example](examples/pareto_front_models.md) for a
runnable MTL-IBP workflow that prints a published front, downloads one model,
and evaluates it on CIFAR-10 with CTRAIN. The full paper code, results, and
reproduction instructions are in the
[paper artifacts](https://github.com/ADA-research/CTRAIN/tree/main/papers/rethinking_evaluation_paradigms).


## License

This project is licensed under the [MIT License](LICENSE).

## Maintainers
This project was developed at the Chair for Artificial Intelligence Methodology at RWTH Aachen University by Konstantin Kaulen under the supervision of Prof. Holger H. Hoos.
Konstantin Kaulen is the current core-maintainer of the project.

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## Acknowledgements

This project incorporates the α,β-CROWN verifier which is developed by a multi-institutional team led by Prof. Huan Zhang. The α,β-CROWN components are licensed under the BSD 3-Clause license.
Furthermore, CTRAIN depends heavily on the `auto_LiRPA` library, which is developed by the same team and also licensed under the BSD 3-Clause License.
