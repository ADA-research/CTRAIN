# Property certification and counterexample API

Prefer `PropertyModelWrapper.certify_properties` for ordinary use. These
functions implement the same [property certification workflow](../../properties.md#certification-and-complete-verification).
Reports distinguish whole-domain certification, unresolved cells and witnessed
violations; a negative relaxation bound alone does not establish a violation.

## Certification

Explicit evaluation `properties` use fresh graphs and original boxes without
changing the training specifications or stored partition. Lazy property datasets
support full-box certification without refinement or complete solving; explicit
properties are required for those operations.

::: CTRAIN.eval.properties.certify_properties

::: CTRAIN.eval.properties.certify_property_dataset

## Counterexample search

`find_counterexample` searches one declared box for an input that violates the
formula's Boolean semantics, including strictness and ties. It returns the
checked input as a CPU Python list, or `None` if no violation was found. An
unsuccessful search is not a certificate. Model modes, buffers and existing
parameter gradients are preserved.

::: CTRAIN.eval.properties.find_counterexample
