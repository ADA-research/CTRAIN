# Property training losses

The property trainer combines task loss with a method-specific property loss.
Training surrogates are distinct from fresh whole-domain certificates.

::: CTRAIN.train.certified.losses.properties.property_loss

## Training strategies

The dispatcher receives explicit bound/model inputs and validated method options.
The legacy `get_property_training_loss(wrapper, ...)` entry point remains supported.

::: CTRAIN.train.certified.losses.property_methods.compute_property_training_loss
