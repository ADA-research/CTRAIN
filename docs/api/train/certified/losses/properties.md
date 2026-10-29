# Property and task loss API

The [property trainer](../properties.md) combines a task loss with a property
objective. See the [property guide](../../../../properties.md#training-and-domains)
for margin semantics and strategy selection.

## Property margin loss

`property_loss` accepts a batch of lower margins. Hinge loss sums deficits over
obligations, then averages regions unless `reduction='none'`. Cross-entropy uses
the padded-margin objective and ignores the hinge slack parameter. A training
loss is distinct from a whole-domain certificate.

::: CTRAIN.train.certified.losses.properties.property_loss

## Training strategies

The dispatcher receives explicit bound/model inputs and validated method
options. The legacy `get_property_training_loss(wrapper, ...)` entry point
delegates to it for compatibility.

::: CTRAIN.train.certified.losses.property_methods.compute_property_training_loss

## Advisory preservation

For models whose lowest score selects an action, use fixed source-model
`argmin` labels sampled before repair. This empirical task loss does not
certify advisory preservation. The [ONNX/VNNLIB notebook](../../../../examples/onnx_vnnlib.ipynb)
shows the complete workflow.

::: CTRAIN.train.certified.losses.properties.advisory_cross_entropy

## Independent arithmetic heads

`mnist_addition_cross_entropy` expects `[batch, 39]` logits split into digit,
digit and sum heads of widths `10`, `10` and `19`, with `[batch, 3]` integer
labels `[a, b, a+b]`. It averages cross-entropy across the three heads.
See the [arithmetic notebook](../../../../examples/mnist_arithmetic.ipynb).

::: CTRAIN.train.certified.losses.properties.mnist_addition_cross_entropy
