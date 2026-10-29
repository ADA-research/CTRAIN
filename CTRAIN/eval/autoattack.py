"""Optional official AutoAttack evaluation for image inputs in [0, 1]."""
import math
import time

import numpy as np
import torch

from CTRAIN.util.util import preserve_model_state, validate_test_samples


def eval_autoattack(model, data_loader, eps, return_adv_indices=False,
                    device='cuda', test_samples=np.inf, version='standard',
                    seed=0, verbose=False):
    """Evaluate empirical robustness using the official AutoAttack ensemble.

    Inputs are mapped from loader coordinates to pixel space for the attack,
    then transformed back before model evaluation. This gives an empirical
    upper bound on robust accuracy, not a certificate.

    ``eps`` is in the loader's input coordinates, like ``eval_adversarial``.
    Loader min/max describe raw pixel endpoints 0/1 after normalisation. Attacks
    run in pixel space and the model receives loader-normalised inputs. Unequal
    pixel-space radii are rejected: official AutoAttack requires a scalar radius.
    Accuracy is an attack upper bound, not a certificate. Samples are processed
    in loader batches without materialising the full dataset on the GPU.

    Args:
        model (torch.nn.Module): Classifier receiving inputs in loader coordinates.
        data_loader (DataLoader): Image loader with min and max metadata mapping raw pixels
            to model coordinates.
        eps (float or torch.Tensor): Perturbation radius in model input coordinates.
        return_adv_indices (bool): Whether to return a Boolean failure mask and per-input
            result records.
        device (str or torch.device): Attack device.
        test_samples (int or float): Positive sample budget; infinity evaluates all inputs.
        version (str): Official AutoAttack version: 'standard', 'plus' or 'rand'.
        seed (int): Attack random seed.
        verbose (bool): Whether to print AutoAttack progress.

    Returns:
        (float or tuple): Empirical adversarial accuracy, or (accuracy, failure_mask, results).
            The Boolean mask marks inputs with a found misclassification.
    """
    if version not in ('standard', 'plus', 'rand'):
        raise ValueError('AutoAttack version must be standard, plus, or rand')
    test_samples = validate_test_samples(test_samples)
    lower = torch.as_tensor(data_loader.min, device=device)
    width = torch.as_tensor(data_loader.max, device=device) - lower
    radius = torch.as_tensor(eps, device=device) / width
    if (not torch.isfinite(width).all() or not (width > 0).all()
            or not torch.isfinite(radius).all() or not (radius >= 0).all()
            or not torch.allclose(radius, radius.flatten()[0].expand_as(radius))):
        raise ValueError('AutoAttack requires a finite, nonnegative uniform pixel-space radius')
    try:
        from autoattack import AutoAttack
    except ImportError as error:
        raise ImportError('Install the official AutoAttack implementation with '
                          '`ctrain-install-git-deps`.') from error

    class PixelModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model

        def forward(self, pixels):
            return self.model(pixels * width.to(pixels) + lower.to(pixels))

    failures, results = [], {}
    seen = 0
    with preserve_model_state(model), torch.enable_grad():
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        attack = AutoAttack(PixelModel(), norm='Linf', eps=radius.flatten()[0].item(),
                            version=version, seed=seed, verbose=verbose, device=device)
        for data, targets in data_loader:
            if seen >= test_samples:
                break
            count = int(min(len(targets), test_samples - seen))
            data, targets = data[:count].to(device), targets[:count].to(device)
            pixels = (data - lower.to(data)) / width.to(data)
            if not torch.isfinite(pixels).all() or (pixels < -1e-5).any() or (pixels > 1+1e-5).any():
                raise ValueError('AutoAttack inputs must map to pixels in [0, 1]')
            pixels = pixels.clamp(0, 1)
            start = time.monotonic()
            adversarial = attack.run_standard_evaluation(pixels, targets, bs=count)
            with torch.no_grad():
                failed = attack.model(adversarial).argmax(1).ne(targets).cpu()
            elapsed = (time.monotonic() - start) / count
            failures.append(failed)
            for i, found in enumerate(failed.tolist()):
                results[seen+i] = dict(result='sat' if found else None,
                                       method='AUTOATTACK', running_time=elapsed)
            seen += count
    if not seen:
        raise ValueError('Evaluation loader must contain samples')
    mask = torch.cat(failures).numpy()
    accuracy = float(1 - mask.mean())
    return (accuracy, mask, results) if return_adv_indices else accuracy
