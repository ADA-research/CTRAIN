# CC-IBP and Exp-IBP

CTRAIN implements the three expressive objectives from [De Palma et al., Expressive Losses for Verified Robustness via Convex Combinations (ICLR 2024)](https://arxiv.org/abs/2305.13991). The formulas follow the [authors' training implementation](https://github.com/alessandrodepalma/expressive-losses/blob/main/train.py). They are sibling subclasses of `ExpressiveIBPModelWrapper` and share
`expressive_ibp_train_model`, the PGD attack, and auto_LiRPA IBP bounds.
Each specialized class selects its own loss and coefficient name; CC-IBP and
Exp-IBP do not depend on the MTL-IBP wrapper, trainer, or loss.

Let `b` be the lower bounds on true-class margins and `a` the margins at the adversarial examples. For a coefficient `alpha` in `[0, 1]`:

- **CC-IBP:** cross-entropy on margins `alpha * b + (1 - alpha) * a`.
- **Exp-IBP:** `IBP_CE ** alpha * adversarial_CE ** (1 - alpha)`, using the two **batch-mean** losses, as in the reference implementation.
- **MTL-IBP:** `alpha * IBP_CE + (1 - alpha) * adversarial_CE`.

All three use `alpha=1` for pure IBP and `alpha=0` for pure adversarial loss. Exp-IBP evaluates the geometric mean in log space, flooring zero cross-entropies at the dtype's smallest positive normal value to avoid undefined gradients. Endpoints use the exact component loss.

```python
from CTRAIN import CCIBPModelWrapper, ExpIBPModelWrapper

wrapper = CCIBPModelWrapper(
    model=model,
    input_shape=(1, 3, 32, 32),
    eps=2 / 255,
    num_epochs=100,
    cc_ibp_alpha=0.5,
    device="cuda",
)
wrapper.train_model(train_loader)

# ExpIBPModelWrapper accepts exp_ibp_alpha instead of cc_ibp_alpha.
```

Loaders use the same metadata as other CTRAIN wrappers (`normalised`, `std`, `min`, `max`). All common expressive IBP options apply, including `population_bn`, `pgd_eps_factor`, and `relu_upper_retention`. PGD runs in evaluation mode; losses run in the caller's mode. A scaled attack does not enlarge the nominal IBP box. ReLU retention below one supplies an unsound training surrogate; subsequent ordinary bound computations remain sound.

The wrappers expose `cert_train_method="cc_ibp"` and `"exp_ibp"`, and support existing HPO entry points. The shared PGD factor is named `pgd_eps_factor`, e.g. `cc_ibp:pgd_eps_factor`. Configuration-space keys include `cc_ibp:cc_ibp_alpha` or `exp_ibp:exp_ibp_alpha`. The shared HPO search uses the existing MTL-IBP coefficient range `[0.001, 0.5]`; direct wrapper construction permits `[0, 1]`.

Public trainers are `CTRAIN.train.certified.cc_ibp_train_model` and `exp_ibp_train_model`; losses are `CTRAIN.train.certified.losses.get_cc_ibp_loss` and `get_exp_ibp_loss`. The method-specific trainer functions remain compatibility entry points to
`expressive_ibp_train_model`; their existing arguments and returns are preserved.
The neutral `get_expressive_ibp_loss` owns attack/bound computation, while
method-specific loss functions supply the combination formulas. Returned bounds are the underlying IBP margin bounds, rather than mixed CC margins. When retention is below one, those training bounds are approximate.

The shared expressive IBP trainer owns CTRAIN's initialization, epsilon schedule, regularization and checkpoint policy. `val_loader` does not select weights in this trainer; training returns the final model. Checkpoints contain the model, optimizer and epoch, rather than a complete resumable trajectory. Implementing the paper objectives does not establish reproduction of the paper's benchmark scores; full GPU training and benchmark certification have not been run for these additions.
