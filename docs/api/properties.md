# Output-property DSL API

The [property guide](../properties.md) explains strictness, Boolean composition,
input coordinates and certification. The [tutorial](../examples/general_properties.ipynb)
constructs and trains a complete example.

::: CTRAIN.properties
    options:
      members:
        - Outputs
        - OutputIndex
        - Affine
        - Ge
        - Le
        - Gt
        - Lt
        - And
        - Or
        - Box
        - PropertySpec
        - PropertyGraph

`load_vnnlib_property` is documented in the
[loader reference](data_loaders/data_loaders.md).
For lower-level integrations, see [property bounds](bound/properties.md),
[certification and counterexample search](eval/properties.md), and
[task/property losses](train/certified/losses/properties.md).
