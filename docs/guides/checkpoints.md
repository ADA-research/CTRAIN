# Checkpoints and Model Selection

CTRAIN supports saving weights for later evaluation and saving training trajectories for continuation. Inference weights describe the model to use for predictions. A trajectory checkpoint additionally records the optimiser, learning-rate scheduler and completed epoch needed to continue the configured training run.

This distinction also matters when validation selects an earlier model. Restoring the selected weights changes the inference model, while the optimiser still belongs to the completed trajectory. Reload a trajectory checkpoint before continuing training.

## Inference Weights

After training, save the wrapper's inference representation with ordinary PyTorch:

```python
import torch

# wrapped_model is the trained wrapper from the training guide.
torch.save(wrapped_model.state_dict(), "mnist_ibp.pt")
```

To load these weights, recreate the same architecture and wrapper configuration, then load the state dictionary and switch to evaluation mode:

```python
from CTRAIN.model_definitions import CNN7_Shi
from CTRAIN.model_wrappers import ShiIBPModelWrapper

wrapped_model = ShiIBPModelWrapper(
    model=CNN7_Shi(in_shape=(1, 28, 28)),
    input_shape=(1, 28, 28),
    eps=0.1,
    num_epochs=70,
    warm_up_epochs=1,
    ramp_up_epochs=40,
    lr_decay_kwargs={"milestones": [50, 60], "gamma": 0.2},
    device="cpu",
)
wrapped_model.load_state_dict(
    torch.load("mnist_ibp.pt", map_location="cpu", weights_only=True),
)
wrapped_model.eval()
```

Most LiRPA wrappers expose bounded-graph state dictionaries. PGD and set-based wrappers expose native PyTorch weights. Loading therefore requires the same representation, as well as matching layer dimensions and class count. The [evaluation tutorial](../examples/model_evaluation.ipynb) loads the inference weights saved by the introductory training notebook.

Store the architecture, preprocessing, perturbation radius and method configuration alongside inference weights. These settings are needed to interpret the model, but are not contained in a plain state dictionary.

## Training Trajectories

Classifier wrappers can save trajectories automatically through `checkpoint_save_path` and `checkpoint_save_interval`. In the following example, we configure a fresh MNIST run and stop after its first 15 epochs:

```python
from CTRAIN.data_loaders import load_mnist

train_loader, val_loader, test_loader = load_mnist(batch_size=128, seed=0)

def make_training_wrapper():
    return ShiIBPModelWrapper(
        model=CNN7_Shi(in_shape=(1, 28, 28)),
        input_shape=(1, 28, 28),
        eps=0.1,
        num_epochs=70,
        warm_up_epochs=1,
        ramp_up_epochs=40,
        lr_decay_kwargs={"milestones": [50, 60], "gamma": 0.2},
        checkpoint_save_path="./checkpoints/mnist_ibp",
        checkpoint_save_interval=10,
        device="cuda",
    )

training_model = make_training_wrapper()
training_model.train_model(train_loader, val_loader=val_loader, end_epoch=15)
```

Automatic saving occurs at the requested interval and at the exclusive end of the training call. This run saves the completed 10-epoch and 15-epoch trajectories. Existing checkpoint files are not silently overwritten, so use a fresh directory for an independent experiment.

Now, we recreate the wrapper and restore the 15-epoch trajectory. The resume call loads its states and continues to the end of the original 70-epoch schedule:

```python
resumed_model = make_training_wrapper()
resumed_model.resume_from_checkpoint(
    "./checkpoints/mnist_ibp/15_checkpoint.pt",
    train_loader,
    val_loader=val_loader,
)
```

Classifier trajectory checkpoints contain model, optimiser, scheduler, completed epoch and loss, together with reporting history and validation-selection state. The common filename is `{epoch}_checkpoint.pt`; set-based training retains `checkpoint_{epoch}.pt`. Resume methods accept explicit paths.

Keep the original `num_epochs`, warm-up, ramp-up and learning-rate schedule when resuming. Use the same validation data, metric and evaluation radius so that saved selection scores remain comparable. Classifier checkpoints do not restore RNG or data-loader state, so resumption does not by itself replay the same stochastic trajectory. Legacy checkpoints without the required scheduler state can be loaded for inference but cannot reliably resume the standard wrapper schedule.

GLS checkpoints also retain the centre/deviation optimiser state and noise scales. PGD teacher checkpoints retain their batch-stepped scheduler. CC-Dist continuation requires the same compatible frozen teacher. Consult the method pages for these additional requirements.

## Validation Selection

Model selection is optional: without `val_loader`, training returns the final weights. With validation, the trainer retains the best eligible model according to its method-specific criterion. Ties preserve the earliest winner, and selection state persists across partial runs and checkpoint resumption.

| Method | Selection criterion |
|---|---|
| Shi-IBP, CROWN-IBP, SABR, MTL-IBP, CC-IBP, Exp-IBP, CC-Dist | Lowest robust validation data loss after ramp-up, at the evaluation radius. |
| TAPS, STAPS | Highest latent-margin proxy accuracy after reaching full radius, at the evaluation radius. |
| IBP-R | Highest adversarial validation accuracy after mixing/ramp-up, at the evaluation radius. |
| GLS | Highest certified validation accuracy after ramp-up, using the configured bound method and evaluation radius. |
| Set-based | Highest full-box forward certified validation accuracy after warm-up/ramp-up. |
| PGD teacher | Lowest adversarial validation cross-entropy at the evaluation radius. |

Robust-loss selection excludes training regularisers and evaluates the configured method's data objective. For CROWN-IBP, this includes the CROWN bound computation. No validation work is performed when the loader is omitted.

These metrics have different meanings. TAPS/STAPS latent-margin accuracy is a proxy, and IBP-R adversarial accuracy is empirical. Report the actual selection criterion alongside the final evaluation method.

Automatic trajectory saves occur before the selected inference weights are restored. The trajectory therefore contains a consistent model/optimiser/scheduler state for its own epoch. A legacy checkpoint without selection state begins a new selection history; earlier winning weights cannot be recovered from its trajectory weights alone.

## General-Property Checkpoints

Property training additionally maintains the declared specifications, domain partition and sampling state. Its checkpoints record this configuration and cover, training history, selection state and Python, NumPy and PyTorch RNG states. An explicit loader generator can also be saved.

For a fresh property wrapper and task loader constructed as in the [property guide](../properties.md), `train_model(task_loader, end_epoch=10)` runs the first ten epochs. Save this trajectory with `save_checkpoint("property_epoch_10.pt", task_loader)`. To continue, recreate the same complete wrapper configuration, call `load_checkpoint("property_epoch_10.pt")`, then `train_model(task_loader)`.

Reconstruct compatible properties and any lazy property dataset before loading. Configuration mismatches are rejected, including different dataset descriptors. Older property checkpoint fields remain supported without discarding domain-cover or RNG state. Captured RNG state does not include arbitrary multi-worker or sampler state.

After warm-up and the weight ramp, selection first prefers a whole-domain certificate. Among certified models, it prefers lower validation task loss. Until certification is achieved, it prefers a larger minimum lower margin, then lower task loss. These diagnostics cover the declared property domains; task-validation data does not define a separate property holdout.

As with classifier training, automatic saves precede inference selection. A property wrapper with selected inference weights must reload a trajectory before saving or continuing a resumable run. The [wrapper reference](../api/model_wrappers/property_model_wrapper.md) documents the property checkpoint options.
