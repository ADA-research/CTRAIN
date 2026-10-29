# Property training

This trainer is normally called through
[`PropertyModelWrapper`](../../model_wrappers/property_model_wrapper.md).
The [property/task losses](losses/properties.md) and
[certification functions](../../eval/properties.md) have separate references.

::: CTRAIN.train.certified.properties.property_train_model
    options:
      docstring_style: google
      show_root_heading: true

## Trajectory state

Direct trainer callers supply this state alongside explicit models, bounds,
losses and optimizer. `PropertyModelWrapper` maintains the same state internally.

::: CTRAIN.train.certified.properties.PropertyTrainingState
