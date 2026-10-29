# Checkpoints and model selection

## Inference weights

`wrapper.state_dict()` stores weights for the wrapper's inference representation.
Most LiRPA wrappers use a bounded graph; PGD and set-based wrappers expose native
weights. Recreate the same architecture and representation before loading.
The [evaluation tutorial](../examples/model_evaluation.ipynb) handles native
state dictionaries and common checkpoint containers with strict loading.

An inference checkpoint does not contain an active optimizer or complete training
configuration. Preserve architecture, preprocessing, class count, radius and
method settings alongside the weights.

## Training trajectories

Classifier trajectory checkpoints retain model, optimizer, completed epoch,
loss and LR scheduler state. `resume_from_checkpoint` restores those states
and continues at the next epoch. Legacy checkpoints without scheduler state
cannot reliably resume the standard wrapper schedule; load their model weights
for inference instead.

New certified-training checkpoints also retain reporting history. Automatic saves happen at the
configured interval and at the exclusive end of each training call, before
inference selection. Existing checkpoint files are never silently overwritten.
The common writer uses `{epoch}_checkpoint.pt`; set-based keeps its established
`checkpoint_{epoch}.pt` filenames for compatibility. Resume APIs accept explicit
paths, including older filenames.

GLS stores its centre/deviation optimizer state and noise scales. PGD teacher
checkpoints retain their per-batch LR scheduler. All classifier trajectory checkpoints also
retain validation selection state (best score, weights and epoch). Consult the individual wrapper for additional
method-specific requirements, such as the same frozen teacher when resuming CC-Dist.

Classifier checkpoints do not save RNG or data-loader state. Restore those
separately if a matched stochastic trajectory is needed; arbitrary worker or
sampler state is not automatically replayed. Keep the original `num_epochs`,
ramp-up and LR schedule when resuming a partial run. Continue selection with
the same validation data, metric and evaluation radius so saved scores remain
comparable. Legacy checkpoints without selection state start a fresh selection
history; their earlier validation winner cannot be recovered from trajectory
weights alone.

Restoring validation-selected **inference weights** does not rewind optimizer
or scheduler state. Saved trajectory checkpoints retain their own epoch's state.
Reload a trajectory before continuing training from a selected model.

## Validation selection

Model selection is optional for every method: omit `val_loader` to return final
weights. Supplying it enables selection; ties keep the earliest winner. Best
scores and weights persist across partial calls and checkpoint resumption.
New robust-loss selectors exclude training regularizers and reuse the configured
attack/loss options in evaluation mode. CROWN-IBP selection includes CROWN bound
computation; no validation work runs when the loader is omitted. With a loader:

| Method | Behavior |
|---|---|
| Shi-IBP, CROWN-IBP, SABR, MTL-IBP, CC-IBP, Exp-IBP, CC-Dist | Minimize the method’s robust validation data loss after ramp-up, at the fixed evaluation radius |
| TAPS, STAPS | Select by latent-margin proxy accuracy once training reaches full radius, at the fixed evaluation radius |
| IBP-R | Select by adversarial validation accuracy after mixing/ramp-up, at the evaluation radius |
| GLS | Select by sound certified validation accuracy at the configured evaluation radius and bound method after ramp-up |
| Set-based | Select by fresh full-box forward certified accuracy after warm-up/ramp-up; keep the earliest model on ties |
| Native PGD teacher | Minimize adversarial validation cross-entropy at the fixed evaluation radius |

A TAPS/STAPS latent proxy is not certified accuracy. An IBP-R attack metric is
also empirical. A selected model and its metric must be reported with the actual
selection policy, rather than a generic claim of validation certification.

## General-property checkpoints

Property checkpoints use the common `model_state_dict`, `optimizer_state_dict`,
optional `lr_scheduler_state_dict`, `epoch`, `loss` and `selection_state` fields,
plus configuration, domain partition, history, and
Python/NumPy/PyTorch RNG states. An explicit loader generator can also be saved.
The caller reconstructs compatible properties/datasets before loading; configuration
mismatches are rejected. This does not capture arbitrary worker/sampler state.
The loader still accepts the previous property fields (`model`, `optimizer`,
`scheduler`, `best_score`, `best_weights`) without discarding cover or RNG state.

After warm-up and weight ramp, property selection first prefers a whole-domain
certificate. Among certified candidates it prefers lower validation task loss;
otherwise it prefers a larger minimum lower margin, then lower validation task
loss. These diagnostics concern the declared property domains. A task-validation
loader does not replace them with a fresh property holdout.

Automatic trajectory saves precede inference-best restoration. A wrapper whose
weights were selected for inference must reload a trajectory before saving a
resumable checkpoint. See the [property guide](../properties.md) for migration,
refinement and lazy-dataset constraints.
