# Property bound API

`PropertyBounds` adapts affine/Boolean output specifications to auto_LiRPA and
is normally constructed by [PropertyModelWrapper](../model_wrappers/property_model_wrapper.md).
Calling it with a specification index and a batch of boxes returns lower
margins for the specification's top-level obligations. Composite formulas
retain their logical structure.

Boxes use the verification model's coordinates, with no implicit image
normalization. `box_endpoints` expands endpoints outward when casting their
stored dtype. See [property semantics](../../properties.md#semantics-and-compiler)
and the [certification API](../eval/properties.md) for complete reports rather
than interpreting a single bound as a whole-domain result.

::: CTRAIN.bound.properties.PropertyBounds

::: CTRAIN.bound.properties.box_endpoints
