# HPO configuration-space API

The public wrapper [HPO methods](../../guides/hpo.md) normally construct these
ConfigSpace objects. `get_config_space` selects a method-specific space from
the wrapper; `get_base_config` defines shared training parameters. Individual
builders support inspecting or composing method spaces.

Search ranges are narrower than some constructor ranges. Defaults must fit the
selected space, and scheduling parameters depend on the full epoch budget.
These helpers define search spaces; they do not run trials or restore weights.

::: CTRAIN.model_wrappers.configs
    options:
      show_if_no_docstring: true
