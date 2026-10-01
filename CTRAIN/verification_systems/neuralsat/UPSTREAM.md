# NeuralSAT vendored source

Source: https://github.com/dynaroars/neuralsat
Branch: develop
Commit: c5097698a32a6dc04d176dcf72b61a931b89cdac

Runtime source copied from src/, excluding example/,
train/, and Python caches. Upstream LICENSE, README.md and requirements.txt
are retained. The runtime includes NeuralSAT's own auto_LiRPA and onnx2pytorch
forks; these run in a subprocess and do not replace CTRAIN's dependencies.
Upstream benchmark assets and training utilities are not distributed.

CTRAIN adds beartype==0.16.4 and coloredlogs>=15.0.1. Do not install the
upstream requirements.txt into CTRAIN: its exact pins replace existing packages.

Local compatibility patch: src/setting.py initialises use_mip_verify to
USE_GUROBI rather than True. Upstream disables MIP tightening on license failure
but still invokes MIP presolving, which crashes on small models without a
license. This gates presolving using the same upstream license probe. No bound
propagation or solver algorithms are changed.
