# Set-based zonotope propagation API

`CTRAIN.bound.set_based_zono` contains the specialized representation and
auto_LiRPA adapters used by set-based training. It retains centers, correlated
generators and independent error radii for the training objective; it is not
a general zonotope library.

See [representation and operators](../../set_based.md#representation-and-propagation)
for supported semantics and local LiRPA adapters.

::: CTRAIN.bound.set_based_zono.Zonotope

::: CTRAIN.bound.set_based_zono.bound_zonotope

## Input boxes and output margins

`box_zonotope` builds this representation from finite, ordered, batched lower
and upper endpoints, with one diagonal generator per input coordinate.
`zonotope_margins` returns lower true-class-minus-competitor margins, retaining
generator correlations. Its true-class entry is `inf` and can be ignored when
checking whether every competing-class margin is positive.

For ordinary full-box classification certification, use the
[shared evaluation API](../eval/eval.md) with `method="FORWARD"`.

::: CTRAIN.bound.set_based_zono.box_zonotope

::: CTRAIN.bound.set_based_zono.zonotope_margins
