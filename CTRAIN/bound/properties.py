"""Compute lower satisfaction margins for general input-output specifications using auto_LiRPA."""
import copy
from contextlib import nullcontext
import torch
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
from CTRAIN.properties import Comparison, PropertyGraph, obligations, _AffineFormula
from CTRAIN.util.util import preserve_model_state
from CTRAIN.bound.util import compute_ibp_bounds

from CTRAIN.properties.training import BOUND_METHODS as METHODS
CERTIFICATION_METHODS = (*METHODS, 'ALPHA-CROWN')


def box_endpoints(boxes, reference):
    """Convert stored box endpoints to the model device and dtype with outward rounding."""
    lower64 = torch.stack([b.lower for b in boxes]).to(reference.device)
    upper64 = torch.stack([b.upper for b in boxes]).to(reference.device)
    lower, upper = lower64.to(reference), upper64.to(reference)
    lower = torch.where(lower.double() > lower64,
                        torch.nextafter(lower, torch.full_like(lower, -float('inf'))), lower)
    upper = torch.where(upper.double() < upper64,
                        torch.nextafter(upper, torch.full_like(upper, float('inf'))), upper)
    return lower, upper


class PropertyBounds:
    """Compute property satisfaction bounds while sharing the original network weights.

    Affine comparisons are bounded through specification matrices. Boolean
    formulas are appended to the network as fixed satisfaction-margin graphs;
    property_bound='atomic' instead bounds and combines their affine leaves.
    Outer conjunctions are treated as separate requirements.

    Args:
        model (torch.nn.Module): Verification network referenced by the specifications.
        input_shape (tuple): Verification input shape without the batch dimension.
        specs (Iterable[PropertySpec]): Specifications defining formulas and input boxes.
        bound_opts (dict, optional): auto_LiRPA options and optional property_bound='graph'
            or 'atomic'.
    """
    def __init__(self, model, input_shape, specs, bound_opts=None):
        self.model, self.specs = model, tuple(specs)
        self.reference = next(model.parameters(), None)
        if self.reference is None:
            self.reference = next(model.buffers(), torch.zeros(()))
        self.input_shape = tuple(input_shape)
        example = self.reference.new_zeros((1, *self.input_shape))
        bounded_example = BoundedTensor(example, PerturbationLpNorm(norm=float('inf'), eps=1.))
        self.graphs, self.affine_formulas = {}, {}
        with preserve_model_state(model), torch.no_grad():
            model.eval()
            output = model(example)
            if output.ndim != 2:
                raise ValueError('Verification model must return [batch, output_width]')
            self.width = output.shape[1]
            for spec in specs:
                spec.formula.margin(output)  # validate output indices before LiRPA tracing
                for domain in spec.domains:
                    if tuple(domain.lower.shape) != self.input_shape:
                        raise ValueError('Property domain shape differs from verification input shape')
            options = dict(conv_mode='matrix', relu='adaptive') if bound_opts is None else copy.deepcopy(bound_opts)
            self.property_bound = options.pop('property_bound', 'graph')
            if self.property_bound not in ('graph', 'atomic'):
                raise ValueError('bound_opts.property_bound must be graph or atomic')
            # LiRPA otherwise toggles parameter gradients off in evaluation mode.
            # Fixed BN statistics and weight gradients are independently required.
            options.setdefault('param', {})['auto_requires_grad'] = False
            self.raw = BoundedModule(model, bounded_example, bound_opts=options,
                                     device=self.reference.device)
            for spec_index, spec in enumerate(specs):
                for index, formula in enumerate(obligations(spec.formula)):
                    if not isinstance(formula, Comparison):
                        if self.property_bound == 'atomic':
                            self.affine_formulas[spec_index, index] = _AffineFormula(formula, self.width, self.reference)
                        else:
                            graph = PropertyGraph(model, formula, output_width=self.width)
                            self.graphs[spec_index, index] = BoundedModule(
                                graph, bounded_example, bound_opts=options, device=self.reference.device)
        source = {id(p) for p in model.parameters()}
        for bounded in self.modules:
            if any(id(p) not in source for p in bounded.parameters()):
                raise RuntimeError('LiRPA conversion copied trainable parameters; shared weights required')

    @property
    def modules(self):
        """Return the bounded network and composed formula graphs sharing its parameters."""
        return (self.raw, *self.graphs.values())

    def sync_buffers(self):
        """Synchronise verification-graph buffers with the current network buffers."""
        buffers = dict(self.model.named_buffers())
        with torch.no_grad():
            for name, value in self.raw.named_buffers():
                # ONNX constants can be shared across several bound graphs.
                # Even copying identical values invalidates saved autograd views.
                if not torch.equal(value, buffers[name]):
                    value.copy_(buffers[name])
            for bounded in self.graphs.values():
                for name, value in bounded.named_buffers():
                    if name.startswith('model.'):
                        source = buffers[name.removeprefix('model.')]
                        if not torch.equal(value, source):
                            value.copy_(source)

    def point_margins(self, spec_index, output, *, boolean=False):
        """Evaluate each top-level requirement on a batch of network outputs.

        With boolean=True, return exact pointwise satisfaction tests, including strict
        comparison ties. Otherwise return differentiable satisfaction margins.
        """
        results = []
        for index, formula in enumerate(obligations(self.specs[spec_index].formula)):
            key = (spec_index, index)
            if key not in self.affine_formulas:
                self.affine_formulas[key] = _AffineFormula(formula, self.width, self.reference)
            results.append(self.affine_formulas[key](output, boolean=boolean))
        return torch.cat(results, dim=1)

    def __call__(self, spec_index, boxes, method='CROWN-IBP', *, relu_upper_retention=1.):
        """Bound every top-level requirement over the supplied boxes.

        Args:
            spec_index (int): Index of the specification in the constructor's specs.
            boxes (Sequence[Box]): Nonempty batch of regions in verification coordinates.
            method (str): IBP, CROWN-IBP, CROWN or ALPHA-CROWN.
            relu_upper_retention (float): Training-only IBP retention in (0, 1]. Values below
                one are unsound surrogates.

        Returns:
            (torch.Tensor): Lower margins of shape (number of boxes, number of requirements).
                Positive finite bounds establish satisfaction when retention is one.

        The computation uses evaluation-mode buffers while preserving caller modes
        and parameter gradient flags. Gradients through model weights remain
        available for training except during alpha-CROWN slope optimisation.
        """
        if method not in CERTIFICATION_METHODS:
            raise ValueError(f'bound_method must be one of {CERTIFICATION_METHODS}')
        if not boxes:
            raise ValueError('Cannot bound an empty region batch')
        if method != 'IBP' and relu_upper_retention != 1.:
            raise ValueError('ReLU retention is only available for training IBP')
        def compute(model, **kwargs):
            if method == 'IBP':
                return compute_ibp_bounds(model, relu_upper_retention, **kwargs)
            return model.compute_bounds(**kwargs)
        lower, upper = box_endpoints(boxes, self.reference)
        nominal = (lower + upper) / 2
        formulas = obligations(self.specs[spec_index].formula)
        results = {}
        optimized = method == 'ALPHA-CROWN'
        lirpa_method = 'CROWN-Optimized' if optimized else method
        with preserve_model_state(self.model, *self.modules), (torch.enable_grad() if optimized else nullcontext()):
            self.model.eval()
            for bounded in self.modules:
                bounded.eval()
            if optimized:
                # Optimize relaxation slopes without accumulating model gradients.
                for parameter in self.model.parameters():
                    parameter.requires_grad_(False)
            self.sync_buffers()
            atomic = [(i, f) for i, f in enumerate(formulas) if isinstance(f, Comparison)]
            if atomic:
                dummy = nominal.new_zeros((len(boxes), self.width))
                rows, offsets = [], []
                for _, formula in atomic:
                    row, offset = (formula.expression * (1 / formula.scale)).tensors(dummy)
                    rows.append(row)
                    offsets.append(offset)
                C = torch.stack(rows).unsqueeze(0).expand(len(boxes), -1, -1)
                ptb = PerturbationLpNorm(norm=float('inf'), x_L=lower, x_U=upper)
                lb, _ = compute(self.raw, x=(BoundedTensor(nominal, ptb),),
                                               C=C, method=lirpa_method, bound_upper=False)
                lb = lb + torch.stack(offsets)
                for column, (index, _) in enumerate(atomic):
                    results[index] = lb[:, column:column + 1]
            for index, formula in enumerate(formulas):
                if not isinstance(formula, Comparison):
                    ptb = PerturbationLpNorm(norm=float('inf'), x_L=lower, x_U=upper)
                    if self.property_bound == 'atomic':
                        compiled = self.affine_formulas[spec_index, index]
                        C = compiled.weight.unsqueeze(0).expand(len(boxes), -1, -1)
                        atomic_lb, _ = compute(self.raw,
                            x=(BoundedTensor(nominal, ptb),), C=C, method=lirpa_method, bound_upper=False)
                        lb = compiled.reduce(atomic_lb + compiled.bias)
                    else:
                        lb, _ = compute(self.graphs[spec_index, index],
                            x=(BoundedTensor(nominal, ptb),), method=lirpa_method, bound_upper=False)
                    results[index] = lb
        return torch.cat([results[i] for i in range(len(formulas))], dim=1)
