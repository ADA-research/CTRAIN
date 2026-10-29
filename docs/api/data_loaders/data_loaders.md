# Data loader API

[Data and models](../../guides/data_and_models.md) documents the shared image-loader
options, splits, transforms, normalization metadata and architecture choices.
For imported verification problems, see
[property import semantics](../../properties.md#user-selected-networks-and-specifications).

::: CTRAIN.data_loaders.data_loaders

## ONNX and VNNLIB problems

::: CTRAIN.data_loaders.vnnlib.load_vnnlib_problem
    options:
      docstring_style: google
      show_root_heading: true

::: CTRAIN.properties.vnnlib.load_vnnlib_property
    options:
      docstring_style: google
      show_root_heading: true

## ACAS artifact helpers

::: CTRAIN.data_loaders.acasxu
    options:
      docstring_style: google
      members:
        - acasxu_property
        - download_acasxu
        - load_acasxu
