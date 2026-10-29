# Hyperparameter Optimisation

Certified training typically involves a trade-off between natural and certified accuracy. Hyperparameters controlling the training objective, perturbation schedule and optimisation can change this trade-off substantially. CTRAIN therefore provides hyperparameter optimisation directly through its classification wrappers, using method-specific configuration spaces and the same training and evaluation interfaces as a manually configured run.

The `hpo` method searches jointly for natural and certified validation accuracy. It returns a Pareto front: configurations for which no other evaluated configuration improves one objective without worsening the other. This allows users to choose a model according to their application's requirements rather than fixing a single weighting before the search.

## Preparing a Search

First, we load MNIST with a training holdout and construct an IBP wrapper. The wrapper specifies the architecture, evaluation radius and per-trial epoch budget. Its method determines the configuration space used by the search.

```python
from CTRAIN.data_loaders import load_mnist
from CTRAIN.model_definitions import CNN7_Shi
from CTRAIN.model_wrappers import ShiIBPModelWrapper

train_loader, val_loader, test_loader = load_mnist(batch_size=128, seed=0)
wrapped_model = ShiIBPModelWrapper(
    model=CNN7_Shi(in_shape=(1, 28, 28)),
    input_shape=(1, 28, 28),
    eps=0.1,
    num_epochs=70,
    warm_up_epochs=1,
    ramp_up_epochs=40,
    lr_decay_kwargs={"milestones": [50, 60], "gamma": 0.2},
    device="cuda",
)
```

Use validation data for both the search objectives and any model selection within a trial. The test split is reserved for assessing the chosen model after the search. Keep the architecture, preprocessing and evaluation sample set fixed when comparing configurations.

## Multi-Objective Optuna

Now, we run a search with at most 20 trials and a wall-clock budget of one day. Each trial trains for the wrapper's 70-epoch budget and evaluates up to 1,000 validation examples:

```python
front = wrapped_model.hpo(
    train_loader=train_loader,
    val_loader=val_loader,
    budget_trials=20,
    budget_time=24 * 60 * 60,
    eval_samples=1000,
    output_dir="./hpo/mnist_ibp",
    seed=0,
)
```

CTRAIN uses Optuna to manage trials and BoTorch as the default sampler. `sampler="nsgaii"` selects an Optuna-only alternative. The search reuses the method's trainer and evaluates robustness at the wrapper's nominal `eps`; training-radius multipliers or ReLU shrinkage do not redefine that evaluation region.

Each returned trial record includes `metrics`, `config`, `checkpoint_path`, constraint values and a feasibility flag. Accuracies are fractions. The returned weights are inference checkpoints, and `hpo` leaves the choice of a final model to the caller.

### Choosing a Model

For example, we can choose the front member with the highest certified validation accuracy, then load its weights into the compatible wrapper:

```python
import torch

selected = max(front, key=lambda trial: trial["metrics"]["cert_acc"])
print("Selected configuration:", selected["config"])
print("Validation metrics:", selected["metrics"])
wrapped_model.load_state_dict(
    torch.load(selected["checkpoint_path"], map_location="cuda", weights_only=True),
)
wrapped_model.eval()
std_acc, cert_acc, adv_acc = wrapped_model.evaluate(test_loader)
```

This selection rule prioritises certification. An application with a natural-accuracy requirement can instead filter by that requirement before selecting. Choose the rule using validation results, then assess the selected model once on the held-out test data.

The `min_nat_acc` and `min_cert_acc` arguments add feasibility thresholds to the search. If feasible trials exist, CTRAIN returns their Pareto front. If none meet the thresholds, it returns an unconstrained fallback front with infeasible entries. Check the `feasible` field before treating a returned configuration as satisfying the requested constraints.

### Continuing a Study

The output directory contains `optuna_study.db` and a `nets` directory with trial weights. Calling `hpo` again with the same output directory and study name reopens the persisted study. `budget_trials` is the total target number of trials, so increasing it from 20 to 30 requests up to ten further trials.

Keep the model, dataset split, evaluation radius, method settings and configuration space compatible when reopening a study. The database records the search, while checkpoint files contain the models; preserve both. A changed experiment protocol should use a new study directory.

## Scalar Optuna and Legacy SMAC

When a single objective is appropriate, `hpo_single_objective` combines the evaluated accuracies with user-specified weights:

```text
objective = nat_acc_weight * nat_acc
          + adv_acc_weight * adv_acc
          + cert_acc_weight * cert_acc
```

Its defaults maximise `nat_acc + cert_acc`. For example, a scalar search can use TPE sampling and leave weight loading to the caller:

```python
best_trial = wrapped_model.hpo_single_objective(
    train_loader=train_loader,
    val_loader=val_loader,
    budget_trials=20,
    eval_samples=1000,
    output_dir="./hpo/mnist_ibp_scalar",
    sampler="tpe",
    nat_acc_weight=1.0,
    cert_acc_weight=1.0,
    load_best=False,
)
```

With the default `load_best=True`, the best inference weights are loaded automatically. This does not restore that trial's optimiser or scheduler. Use [trajectory checkpoints](checkpoints.md) when continuing training. BoTorch and random sampling are also available, and `hpo_smac` retains the earlier scalar SMAC3 workflow.

## Configuration Spaces and Evaluation

Search spaces include shared optimisation and schedule parameters together with method-specific settings. They need not span every value accepted by a wrapper constructor. Fixed options are retained when recreating trial wrappers; the individual method pages describe the searched and preserved parameters. CC-Dist trials also require the compatible frozen teacher.

By default, HPO uses the shared incomplete-verification and attack evaluation. `complete_verify=True` selects the complete-verification path, which can substantially increase trial cost. Evaluation budgets and verifier choices form part of the experimental protocol and should remain consistent across trials.

`PropertyModelWrapper` uses a separate task/property workflow and does not expose this classifier HPO interface. The [HPO tutorial](../examples/hyperparameter_optimisation.ipynb) demonstrates a basic MNIST search, model selection and test evaluation. The API reference documents the [wrapper methods](../api/model_wrappers/ctrain_wrapper.md) and [configuration spaces](../api/model_wrappers/configs.md).
