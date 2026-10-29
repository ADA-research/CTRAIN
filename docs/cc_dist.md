# CC-Dist

CC-Dist extends [CC-IBP](expressive_losses.md) with knowledge distillation from an adversarially trained teacher network, combining adversarial features and certified feature bounds to align the intermediate representations of the student with those of the teacher. Proposed by [De Palma (2026), *Learning Better Certified Models from Empirically-Robust Teachers*](https://arxiv.org/pdf/2602.02626v1), the method aims to improve the trade-off between natural accuracy and certified robustness by transferring knowledge from an empirically robust model.

In CTRAIN, CC-Dist is available through `CCDistModelWrapper`. It reuses the shared implementations of CC-IBP, PGD and IBP bounding, adding a feature-distillation term to the classification objective. The teacher is used during training; the resulting student can subsequently be evaluated and verified independently.

## Training Objective

CC-IBP combines adversarial classification margins with lower margin bounds obtained through IBP, before applying cross-entropy. CC-Dist additionally encourages the student's features to resemble those of the teacher. Here, features are the activations immediately before the final affine classifier.

For each training input, CTRAIN obtains the teacher's features on the unperturbed input, the student's features on a PGD adversarial example, and lower and upper IBP bounds on the student features. It then mixes each feature bound with the adversarial features using the same coefficient `cc_ibp_alpha` as the classification loss. For each feature coordinate, the distillation term takes the larger squared distance from the two mixed endpoints to the teacher feature. These distances are summed over coordinates and averaged over the batch.

The overall objective adds this distillation term, weighted by `distillation_weight`, to the CC-IBP loss. By default, `distillation_weight=None` selects the paper's coefficient of `5 / w`, where `w` is the feature width. Setting `distillation_weight=0` recovers the CC-IBP objective. The shared trainer additionally applies the configured Shi and L1 regularisers.

## Training with CTRAIN

In the following example, we train a CNN7 teacher and student on CIFAR-10 for robustness against ℓ∞ perturbations with radius ε = 2/255. First, we import the required functions and load the dataset using CTRAIN. Its loader supplies the normalisation metadata and input limits required for adversarial attacks and bound propagation.

```python
from CTRAIN.data_loaders import load_cifar10
from CTRAIN.model_definitions import CNN7_Shi
from CTRAIN.model_wrappers import PGDModelWrapper, CCDistModelWrapper

input_shape = (3, 32, 32)
train_loader, test_loader = load_cifar10(batch_size=128, val_split=False)
```

### Training the Teacher

To obtain an empirically robust teacher, we wrap a CNN7 model with `PGDModelWrapper` and call `train_model`. This wrapper trains an ordinary PyTorch network using adversarial examples. By default, it uses SGD with momentum 0.9 and a triangular learning-rate schedule that increases from zero to 0.2 and then decreases to zero. Each training attack uses ten PGD steps with relative step size 0.25.

```python
teacher = PGDModelWrapper(
    model=CNN7_Shi(in_shape=input_shape),
    input_shape=input_shape,
    eps=2 / 255,
    num_epochs=30,
    l1_reg_weight=1e-5,
    device="cuda",
)
teacher.train_model(train_loader)
```

An existing pretrained teacher can also be supplied. In that case, construct its native PyTorch architecture and load the corresponding weights before passing it to the student wrapper.

### Training the Student

Now, we construct a new CNN7 model and wrap it with `CCDistModelWrapper`. The `teacher` argument receives the trained PyTorch model, available through `teacher.original_model`. CTRAIN makes a private, frozen copy of its feature extractor, which supplies the target features throughout student training.

We configure one warm-up epoch using clean cross-entropy, followed by 80 ramp-up epochs during which the perturbation radius gradually increases to its target. The remaining epochs train at the full radius. Here, `cc_ibp_alpha=0.01` controls the mixture of adversarial and IBP terms in both the classification and distillation losses.

```python
student = CCDistModelWrapper(
    model=CNN7_Shi(in_shape=input_shape),
    input_shape=input_shape,
    eps=2 / 255,
    num_epochs=160,
    teacher=teacher.original_model,
    cc_ibp_alpha=0.01,
    warm_up_epochs=1,
    ramp_up_epochs=80,
    lr_decay_kwargs={"milestones": (120, 140), "gamma": 0.2},
    pgd_steps=8,
    pgd_alpha=0.25,
    pgd_eps_factor=2.1,
    l1_reg_weight=3e-6,
    device="cuda",
)
student.train_model(train_loader)
```

The `pgd_eps_factor` enlarges the input region searched by PGD, while IBP still uses the nominal training region. Training schedules, regularisation and the shared BatchNorm and ReLU-retention options are described in the [training guide](guides/training.md).

Without a validation loader, `train_model` returns the final weights. To select weights using validation data, load a separate validation split and pass it through `val_loader`. CC-Dist selects the model with the smallest robust validation data loss after ramp-up.

### Evaluating the Student

Finally, we set the student to evaluation mode and assess its natural, certified and adversarial accuracy using the common evaluation interface. The following call uses CROWN for incomplete verification and PGD for adversarial evaluation:

```python
student.eval()
std_acc, cert_acc, adv_acc = student.evaluate(test_loader, eval_method="CROWN")
```

Complete verification can be invoked through `student.evaluate_complete`, using α,β-CROWN or NeuralSAT. Both forms of certification concern the student over the full nominal input box; the teacher is excluded from the verified network. See the [evaluation guide](guides/evaluation.md) for the available methods, solver settings and interpretation of results.

## Model Requirements and Training Behaviour

For CC-Dist, both teacher and student must be sequential classifiers ending in a `Linear` layer, with matching feature widths and class counts. CTRAIN accepts native `torch.nn.Sequential` models and model definitions exposing their sequential computation through `.layers`, including `CNN7_Shi`. Supported feature layers comprise convolutions, linear layers, BatchNorm, ReLU, Flatten and Identity. Although `PGDModelWrapper` can train more general networks, including residual architectures, the CC-Dist feature extractor currently requires this sequential structure.

During training, the frozen teacher evaluates clean inputs in evaluation mode. Student attacks also run in evaluation mode, while the loss on the resulting adversarial examples is computed in training mode. After warm-up, the student uses the adversarial batch statistics for both feature extraction and IBP. Thus, models with BatchNorm require the LiRPA settings `forward_before_compute_bounds=False` and `bn.mode="forward"`, which are the defaults. Incompatible settings are rejected before training.

The teacher's parameters are excluded from the student's optimiser, saved inference weights and exported model. When resuming student training, supply the same pretrained teacher and training options again. Teacher training can separately be resumed through `PGDModelWrapper.resume_from_checkpoint`, which restores its native weights, optimiser and learning-rate scheduler. Further details are provided in the [checkpoint guide](guides/checkpoints.md).

## Hyperparameter Optimisation

CC-Dist supports the shared [hyperparameter optimisation interface](guides/hpo.md), including multi-objective searches over natural and certified validation accuracy. Train the teacher on the training partition and reserve separate validation data for the search. The teacher remains fixed across trials.

CTRAIN reuses the CC-IBP search space and additionally searches `cc_dist:distillation_weight` logarithmically over `[0.1 / w, 50 / w]`, with default `5 / w`. This search range is a CTRAIN configuration choice around the paper's fixed coefficient. To change its default, pass `defaults={"distillation_weight": value}` to `hpo`, choosing a value within that range. Run the no-distillation control separately with `distillation_weight=0`.

## Implementation and API Reference

CTRAIN provides an independent implementation of Algorithm 1 and equations (8)–(9) of the CC-Dist paper. The example above demonstrates its use through the library; reproducing reported accuracies additionally requires matching the experiment's data, training and verification protocols.

The API reference documents the [CC-Dist wrapper](api/model_wrappers/cc_dist_model_wrapper.md), [trainer](api/train/certified/cc_dist.md) and [distillation loss](api/train/certified/losses/cc_dist.md), as well as the [PGD teacher wrapper](api/model_wrappers/pgd_model_wrapper.md) and [adversarial trainer](api/train/adversarial/pgd.md).
