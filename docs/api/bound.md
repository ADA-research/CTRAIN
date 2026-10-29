# Bound APIs

Use the wrapper [evaluation interfaces](../guides/evaluation.md) for ordinary
robustness reporting. Lower-level integrations can call
[IBP](bound/ibp.md), [CROWN](bound/crown.md), [SABR](bound/sabr.md),
[TAPS](bound/taps.md), or [set propagation](bound/set_based_zono.md).
SABR/TAPS training bounds and approximate ReLU surrogates do not automatically
supply full input-region certificates.

For affine/Boolean output specifications over declared boxes, see
[property bounds](bound/properties.md).

Forward LiRPA evaluation uses same-slope ReLUs and the full clipped input box:

::: CTRAIN.bound.forward.bound_forward
