# Controller safety

A two-dimensional controller learns from a waypoint or safety teacher. The
verification model adds the current state and bounded disturbance to the action.
The property requires actions within ±0.1, a successor inside [-1, 1]², and a
successor outside the central obstacle (-0.3, 0.3)². Disturbance is ±0.01 per
coordinate. The input region is the union of four boxes around the obstacle.
This is a one-step invariant property; goal reaching is measured empirically.

See [setup](../README.md) and the [notebook](../../docs/examples/case_studies/controller.ipynb).

```bash
python -m case_studies.controller.run --smoke --device cpu --output runs/case_studies/controller_smoke
python -m case_studies.controller.run --teacher safety --epochs 100 --device cpu --output runs/case_studies/controller_safety
python -m case_studies.controller.run --method SABR --training-options '{"subselection_ratio": 0.1, "pgd_steps": 8}' --output runs/case_studies/controller_sabr
```

`--dry-run` prints configuration without training. `--complete` enables optional
complete verification; the bundled verifier and its dependencies are required.
`--max-cells` sets the refinement budget. Neither an unknown bound nor a safe
sample establishes a certificate. The saved `config.json`, `results.json`,
`model.pt` and epoch checkpoints retain the declared property and code provenance.
The smoke uses synthetic data and two epochs; it is not a controller benchmark score.
