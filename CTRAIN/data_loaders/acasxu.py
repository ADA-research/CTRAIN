"""Pinned ACAS Xu property-7/8 artifacts and checked repair inputs.

This is a deliberately narrow benchmark loader, not a general VNNLIB parser.
Released files are checked before conversion or training. Downloads are explicit.
"""
import hashlib
import json
import re
from pathlib import Path
from urllib.request import urlopen
import numpy as np
import torch
from CTRAIN.properties import Outputs, And, Or, Box, PropertySpec
from CTRAIN.util.onnx import load_onnx_model
from CTRAIN.properties.vnnlib import _parse

CASES = {'7': 'ACASXU_run2a_1_9_batch_2000.onnx', '8': 'ACASXU_run2a_2_9_batch_2000.onnx'}
SOURCE = 'https://github.com/VNN-COMP/vnncomp2021/tree/main/benchmarks/acasxu'
FILES = (*CASES.values(), 'prop_7.vnnlib', 'prop_8.vnnlib')


def acasxu_property(case, domain):
    """Create ACAS Xu safe property 7 or 8 over an explicit input box.

    Args:
        case (str): Either ``'7'`` or ``'8'``. Lower output scores select actions.
        domain (Box): Released input coordinates, without extra normalization.

    Returns:
        PropertySpec: Property 7 excludes globally minimal strong-left/right
        scores. Property 8 requires one allowed advisory (clear of conflict or
        weak-left) to beat all forbidden ones. Both exclude unsafe ties.
    """
    if not isinstance(domain, Box) or domain.lower.shape != (5,):
        raise ValueError('ACAS Xu properties require a five-coordinate input box')
    y = Outputs()
    if case == '7':
        formula = And(*(Or(*(y[i] > y[j] for j in range(5) if i != j)) for i in (3, 4)))
    elif case == '8':
        formula = Or(*(And(*(y[i] < y[j] for j in (2, 3, 4))) for i in (0, 1)))
    else:
        raise ValueError("Supported ACAS Xu cases are '7' and '8'")
    return PropertySpec(f'acasxu_{case}', formula, [domain])


def _read_source_spec(path):
    """Read the bounded five-input/five-output subset used by these presets."""
    lower, upper, unsafe = np.full(5, -np.inf), np.full(5, np.inf), []
    declared = set()
    for statement in _parse(path.read_text()):
        if statement[0] == 'declare-const':
            if len(statement) != 3 or statement[2] != 'Real' or statement[1] in declared:
                raise ValueError('Invalid or duplicate ACAS Xu variable declaration')
            declared.add(statement[1])
            continue
        if statement[0] != 'assert' or len(statement) != 2:
            raise ValueError('Unsupported ACAS Xu statement')
        predicate = statement[1]
        if predicate[0] in ('<=', '>=') and isinstance(predicate[1], str) and predicate[1].startswith('X_'):
            index = int(predicate[1][2:]); value = float(predicate[2])
            if not 0 <= index < 5 or not np.isfinite(value):
                raise ValueError('Unsupported ACAS Xu input bound')
            if predicate[0] == '>=':
                lower[index] = max(lower[index], value)
            else:
                upper[index] = min(upper[index], value)
        else:
            unsafe.append(predicate)
    if declared != {f'{prefix}_{i}' for prefix in ('X', 'Y') for i in range(5)}:
        raise ValueError('ACAS Xu presets require exactly five inputs and five outputs')
    box = Box(lower, upper)
    if not unsafe:
        raise ValueError('No unsafe output predicate in specification')
    return box, unsafe


def _source_unsafe(predicates, output):
    def value(term):
        if isinstance(term, str) and term.startswith('Y_'):
            index = int(term[2:])
            if not 0 <= index < 5:
                raise ValueError('Unsupported ACAS Xu output index')
            return output[:, index]
        return float(term)
    def evaluate(predicate):
        op = predicate[0]
        if op in ('and', 'or'):
            terms = torch.stack([evaluate(p) for p in predicate[1:]])
            return terms.all(dim=0) if op == 'and' else terms.any(dim=0)
        if op == '<=':
            return value(predicate[1]) <= value(predicate[2])
        if op == '>=':
            return value(predicate[1]) >= value(predicate[2])
        raise ValueError('Unsupported output predicate')
    return torch.stack([evaluate(p) for p in predicates]).all(dim=0)


def _crosscheck(number, predicates):
    generator = torch.Generator().manual_seed(17)
    output = torch.cat((torch.randn(1024, 5, generator=generator), torch.zeros(1, 5),
                        torch.eye(5), -torch.eye(5)))
    spec = acasxu_property(number, Box([-1] * 5, [1] * 5))
    if not torch.equal(spec.formula.evaluate(output).flatten(), ~_source_unsafe(predicates, output)):
        raise ValueError('Released unsafe predicate differs from this preset; do not merge conventions')


def download_acasxu(directory, revision):
    """Explicitly download both ACAS Xu cases from an immutable upstream commit.

    Args:
        directory (str | pathlib.Path): New artifact directory. Existing paths
            are rejected; downloads do not overwrite experiments or artifacts.
        revision (str): Full lowercase 40-character VNN-COMP 2021 commit SHA.

    Returns:
        dict: Manifest recording source, revision and all four file SHA256 hashes.
        The same manifest is saved as `manifest.json` in the artifact directory.

    No download occurs during import or `load_acasxu`. A failed download may
    leave partial files, but no completed manifest is written until all succeed.
    """
    if not re.fullmatch(r'[0-9a-f]{40}', revision or ''):
        raise ValueError('Download requires a full upstream commit SHA')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for filename in FILES:
        url = f'https://raw.githubusercontent.com/VNN-COMP/vnncomp2021/{revision}/benchmarks/acasxu/{filename}'
        with urlopen(url, timeout=60) as response:
            data = response.read()
        (directory / filename).write_bytes(data)
        hashes[filename] = hashlib.sha256(data).hexdigest()
    manifest = dict(source=SOURCE, revision=revision, sha256=hashes)
    (directory / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    return manifest


def load_acasxu(directory, case='7'):
    """Load a trainable model, checked safe property and artifact manifest.

    The loader verifies the pinned source and all required file hashes, extracts
    the input box from the original VNNLIB, and cross-checks the preset's safe
    formula against the original unsafe predicate on random scores and ties.
    It then uses `load_onnx_model` to check converted predictions against ONNX
    Runtime on domain endpoints, midpoint and 128 deterministic random probes.

    Args:
        directory (str | pathlib.Path): Local artifacts with `manifest.json`,
            usually created by `download_acasxu`. No automatic download occurs.
        case (str): Property `'7'` (network 1_9) or `'8'` (network 2_9).

    Returns:
        tuple: `(model, property_spec, manifest)`. The model is trainable on CPU
        in eval mode. The property uses released coordinates without additional
        normalization and excludes unsafe score ties. The manifest is suitable
        for recording experiment provenance.

    Raises:
        ValueError: The case, manifest, hashes or source specification mismatch.
        AssertionError: ONNX conversion fails the numerical parity check.

    These checks establish neither full graph equivalence nor certification;
    train and certify using `PropertyModelWrapper` afterwards.
    For caller-selected networks or properties, use `load_vnnlib_problem`
    instead; this function supplies only the two original experiment presets.
    """
    if case not in CASES:
        raise ValueError("Supported ACAS Xu cases are '7' and '8'")
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest.get('source') != SOURCE or not re.fullmatch(r'[0-9a-f]{40}', manifest.get('revision', '')):
        raise ValueError('Artifact manifest requires the expected source and pinned commit SHA')
    hashes = manifest.get('sha256', {})
    if set(hashes) != set(FILES):
        raise ValueError('Artifact manifest must contain all four required files')
    for filename in FILES:
        if hashlib.sha256((directory / filename).read_bytes()).hexdigest() != hashes[filename]:
            raise ValueError(f'Artifact hash mismatch: {filename}')
    box, unsafe = _read_source_spec(directory / f'prop_{case}.vnnlib')
    _crosscheck(case, unsafe)
    generator = torch.Generator().manual_seed(17)
    random_probes = box.lower + torch.rand(128, 5, generator=generator,
                                          dtype=torch.double) * (box.upper - box.lower)
    probes = torch.cat((torch.stack((box.lower, box.upper, (box.lower + box.upper) / 2)),
                        random_probes)).float()
    model = load_onnx_model(directory / CASES[case], probes)
    with torch.no_grad():
        if model(probes).shape != (len(probes), 5):
            raise ValueError('ACAS Xu model must return five advisory scores per input')
    return model, acasxu_property(case, box), manifest


__all__ = ['acasxu_property', 'download_acasxu', 'load_acasxu']
