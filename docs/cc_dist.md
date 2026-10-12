# CC-Dist

`CCDistModelWrapper` implements Algorithm 1 and equations (8)–(9) from
[Learning Better Certified Models from Empirically-Robust Teachers](https://arxiv.org/pdf/2602.02626v1).
It reuses CTRAIN's CC-IBP attack, true-class margins and auto_LiRPA bounds.
Student features are the inputs to its final affine classifier. The distillation
term mixes adversarial features with their lower/upper IBP endpoints using the
same `cc_ibp_alpha` as the classification loss, then sums the maximum squared
distance to clean teacher features over feature coordinates and averages samples.
`distillation_weight=None` means beta=5/w; beta=0 is the paper's CC-IBP control.

```python
from CTRAIN import PGDModelWrapper, CCDistModelWrapper
from CTRAIN.model_definitions import CNN7_Shi

teacher = PGDModelWrapper(
    CNN7_Shi((3, 32, 32)), (3, 32, 32), eps=2/255, num_epochs=30,
    l1_reg_weight=1e-5, device='cuda',
)
teacher.train_model(train_loader)
student = CCDistModelWrapper(
    CNN7_Shi((3, 32, 32)), (3, 32, 32), eps=2/255, num_epochs=160,
    teacher=teacher.original_model, cc_ibp_alpha=.01,
    warm_up_epochs=1, ramp_up_epochs=80,
    lr_decay_kwargs=dict(milestones=(120, 140), gamma=.2),
    pgd_steps=8, pgd_alpha=.25, pgd_eps_factor=2.1,
    l1_reg_weight=3e-6, device='cuda',
)
student.train_model(train_loader)
student.eval()
student.evaluate(test_loader, eval_method='CROWN')
# Existing full-box verification, using the same budgets/protocol as other suites:
# student.evaluate_complete(test_loader, timeout=300, engine='abcrown')
```

Teachers and students accept native Sequential classifiers (or CTRAIN `.layers`
models) ending in Linear, with matching feature widths and class counts. The
supported student/teacher feature layers are Conv, Linear, BatchNorm, ReLU,
Flatten and Identity; the existing sequential graph validator rejects mismatched
`.layers`/forward behavior before conversion. The
student makes a private frozen copy of the teacher features; caller weights,
buffers, modes and gradients remain unchanged. Teacher features use evaluation
mode and clean inputs. They are excluded from student inference weights,
optimizers, exports and certificates. Certification evaluates only the student
over the full nominal input box. The teacher is required again when resuming
training; use the same pretrained weights and training options.

Student initialization, epsilon schedule, Shi/L1 regularization, LR updates and
trajectory checkpoints use the shared expressive trainer. Warm-up uses clean CE;
positive-radius training uses one adversarial forward to establish BN statistics
for both the student features and IBP. No clean forward updates running BN
statistics after warm-up. Models with BatchNorm require LiRPA's default
`forward_before_compute_bounds=False` and `bn.mode=forward`; incompatible custom
settings are rejected before training because they replace adversarial statistics.
`population_bn`, `relu_upper_retention` and
`pgd_eps_factor` propagate through training and HPO. Shrunk intervals are unsound
training surrogates and never used for ordinary certification. `train_model`
returns final weights and rejects validation selection; HPO evaluates its supplied
holdout through the existing evaluation APIs. HPO reuses the `cc_ibp:*` search
parameters, preserves the teacher, and searches `cc_dist:distillation_weight`
logarithmically over `[0.1/w, 50/w]`, where `w` is the student's final Linear
input width. Its default is `5/w`; `defaults={'distillation_weight': ...}` can
override that default within the range. This is a proposed search range around
the paper's default, not a published optimum. Run beta=0 separately as the
no-distillation control. Older HPO configurations without this parameter retain
the wrapper's configured beta.

`PGDModelWrapper` and `pgd_train_model` train native PyTorch classifiers without
constructing a LiRPA graph, so larger/residual teachers can train too. Default
teacher training uses SGD with momentum .9, a per-batch triangular learning rate
0 -> .2 -> 0, ten PGD steps of relative size .25, no attack early stopping,
Shi initialization, and global gradient clipping at 10. Attacks run in evaluation
mode and loss evaluation uses the caller's modes. `get_pgd_loss` also supports
native models, CPU inputs, optional error statistics, and exception-safe mode
restoration. Wrapper evaluation builds a temporary LiRPA copy. Teacher trajectory
checkpoints save native weights, optimizer and batch LR scheduler state;
`resume_from_checkpoint` restores them. Neither teacher nor student checkpoints
save RNG or loader state, so stochastic replay requires restoring those separately.

The supplementary implementation was not present in the downloaded arXiv source;
this is an independent implementation of the published algorithm. CPU formula,
gradient, BN, teacher isolation, HPO and checkpoint checks are in
`tests/test_cc_dist.py`. `--cuda` checks real CNN7 architectures at nonzero radii
and actual parameter updates; it remains a synthetic execution check.
`--verifier` trains a toy student and checks real ONNX/VNN-LIB export and the
bundled abCROWN solver, bypassing incomplete screening to exercise that path.
