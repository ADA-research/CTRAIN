# Published Pareto-front Models

The final, completely verified per-method Pareto fronts from the ICML 2026
paper **“Rethinking Evaluation Paradigms in IBP-based Certified Training”** are
available in the
[`kkaulen/ctrain_pareto_fronts`](https://huggingface.co/kkaulen/ctrain_pareto_fronts)
Hugging Face model repository.

The repository manifest records each model's dataset, architecture, training
method, perturbation radius, reported complete-verification accuracies, and
SHA-256 checksum. CTRAIN's paper utility uses this metadata to select the
correct architecture and wrapper before loading a checkpoint.

## MTL-IBP CIFAR-10 Example

The
[runnable example script](https://github.com/ADA-research/CTRAIN/blob/main/papers/rethinking_evaluation_paradigms/examples/evaluate_mtl_front.py)
prints the canonical MTL-IBP CNN7 front at epsilon 2/255, selects its
highest-certified member, downloads it, and evaluates it on the CIFAR-10 test
set:

```bash
python papers/rethinking_evaluation_paradigms/examples/evaluate_mtl_front.py
```

By default, the script evaluates all 10,000 test examples with IBP
certification and CTRAIN's PGD attack. CUDA is strongly recommended. Common
variants are:

```bash
# Inspect the manifest front without downloading weights.
python papers/rethinking_evaluation_paradigms/examples/evaluate_mtl_front.py \
  --list-only

# Evaluate a shorter test-set prefix.
python papers/rethinking_evaluation_paradigms/examples/evaluate_mtl_front.py \
  --test-samples 100

# Select another displayed front member.
python papers/rethinking_evaluation_paradigms/examples/evaluate_mtl_front.py \
  --index 2
```

The script prints the paper's complete-verification metrics separately from
the fresh CTRAIN evaluation. The latter uses incomplete certification through
`CTRAINWrapper.evaluate`; selecting `ADAPTIVE` combines IBP, CROWN-IBP, and
CROWN rather than reproducing the paper's complete alpha-beta-CROWN run.

For publication details and full reproduction instructions, see the
[paper artifacts](https://github.com/ADA-research/CTRAIN/tree/main/papers/rethinking_evaluation_paradigms).
