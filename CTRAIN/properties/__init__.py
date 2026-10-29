"""Represent input-output specifications through affine comparisons and Boolean formulas.

A PropertySpec associates an output formula with closed input boxes. Affine
comparisons define satisfaction margins; conjunction and disjunction combine
them using minimum and maximum. These formulas can be compiled into fixed
computation graphs for auto_LiRPA bound propagation.
"""
from dataclasses import dataclass
from numbers import Real
import math
import torch
from torch import nn


def _number(value):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError('Expected a finite real constant')
    return float(value)


class _Symbolic:
    def __bool__(self):
        raise TypeError('Use And/Or; Python and/or and chained comparisons are unsupported')

    def __eq__(self, other):
        raise TypeError('Exact equality is unsupported; express an explicit tolerance with And')

    def __ne__(self, other):
        raise TypeError('Disequality is unsupported')


@dataclass(frozen=True, eq=False)
class Affine(_Symbolic):
    """Represent an affine expression over network outputs.

    The expression is the sum of coefficient-weighted outputs plus an offset.
    Arithmetic and comparison operators construct formulas without evaluating a
    network. Use Outputs()[i] to select an output coordinate.

    Args:
        coefficients (tuple): Unique nonnegative output indices paired with finite coefficients.
        offset (float): Finite constant added to the weighted outputs.
    """
    coefficients: tuple = ()
    offset: float = 0.

    def __post_init__(self):
        coefficients = tuple(self.coefficients)
        if any(type(i) is not int or i < 0 for i, _ in coefficients) or len(
                {i for i, _ in coefficients}) != len(coefficients):
            raise ValueError('Affine coefficients require unique nonnegative output indices')
        object.__setattr__(self, 'coefficients', tuple((i, _number(v)) for i, v in coefficients))
        object.__setattr__(self, 'offset', _number(self.offset))

    def __add__(self, other):
        other = _affine(other)
        coefficients = dict(self.coefficients)
        for index, value in other.coefficients:
            coefficients[index] = coefficients.get(index, 0.) + value
        return Affine(tuple((i, _number(v)) for i, v in sorted(coefficients.items()) if v),
                      _number(self.offset + other.offset))

    __radd__ = __add__

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + -_affine(other)

    def __rsub__(self, other):
        return _affine(other) - self

    def __mul__(self, constant):
        constant = _number(constant)
        return Affine(tuple((i, _number(v * constant)) for i, v in self.coefficients),
                      _number(self.offset * constant))

    __rmul__ = __mul__

    def __ge__(self, other):
        return Ge(self, other)

    def __le__(self, other):
        return Le(self, other)

    def __gt__(self, other):
        return Gt(self, other)

    def __lt__(self, other):
        return Lt(self, other)

    def tensors(self, output):
        """Return dense affine coefficients and offset in the output tensor's device and dtype."""
        n = output.shape[1]
        values = [0.] * n
        for index, value in self.coefficients:
            if not torch.jit.is_tracing() and index >= n:
                raise ValueError(f'Output index {index} is outside output width {n}')
            values[index] = value
        return output.new_tensor(values), output.new_tensor(self.offset)

    def evaluate(self, output):
        """Evaluate the affine expression on a batch of network outputs."""
        coefficients, offset = self.tensors(output)
        return torch.nn.functional.linear(output, coefficients.unsqueeze(0), offset.reshape(1))


def _affine(value):
    return value if isinstance(value, Affine) else Affine(offset=_number(value))


class Outputs:
    """Construct symbolic references to network output coordinates.

    For example, y = Outputs() followed by And(y[0] >= -0.75, y[0] <= 0.75)
    defines an output-range property. Use And and Or to combine comparisons;
    Python's and, or and chained comparisons do not construct symbolic formulas.
    """
    def __getitem__(self, index):
        return OutputIndex(self, index)


def OutputIndex(output, index):
    """Return an affine expression selecting one network output.

    Args:
        output (Outputs): Symbolic output collection.
        index (int): Nonnegative output coordinate.

    Returns:
        (Affine): Expression with coefficient one at the selected coordinate.
    """
    if not isinstance(output, Outputs) or type(index) is not int or index < 0:
        raise ValueError('OutputIndex requires Outputs and a nonnegative integer index')
    return Affine(((index, 1.),))


@dataclass(frozen=True, eq=False)
class Comparison(_Symbolic):
    """Represent a signed affine output comparison.

    The satisfaction margin is expression / scale. A non-strict comparison is
    satisfied at zero; a strict comparison requires a positive margin.

    Args:
        expression (Affine): Left-hand side minus right-hand side of the comparison.
        strict (bool): Whether equality violates the comparison.
        scale (float): Positive finite margin-normalisation factor.
    """
    expression: Affine
    strict: bool = False
    scale: float = 1.

    def __post_init__(self):
        if not isinstance(self.expression, Affine) or _number(self.scale) <= 0:
            raise ValueError('Comparison requires an affine expression and positive finite scale')
        _number(1 / self.scale)
        if type(self.strict) is not bool:
            raise ValueError('Comparison strictness must be bool')

    def margin(self, output, relu=False):
        # Fold constant scaling into coefficients. An explicit division causes
        # the installed LiRPA graph optimizer to lose dependency flags when it
        # replaces division, and then unnecessarily lower native min/max.
        """Return the scaled affine satisfaction margin for each output."""
        return (self.expression * (1 / self.scale)).evaluate(output)

    def evaluate(self, output):
        """Test each output against the comparison, preserving strict inequalities."""
        margin = self.margin(output)
        return margin > 0 if self.strict else margin >= 0

    def to_dict(self):
        """Return the comparison coefficients, strictness and scale as serialisable data."""
        return dict(op='comparison', coefficients=list(self.expression.coefficients),
                    offset=self.expression.offset, strict=self.strict, scale=self.scale)


def Ge(lhs, rhs, scale=1.):
    """Construct lhs >= rhs with satisfaction margin (lhs - rhs) / scale."""
    return Comparison(_affine(lhs) - rhs, scale=scale)


def Le(lhs, rhs, scale=1.):
    """Construct lhs <= rhs with satisfaction margin (rhs - lhs) / scale."""
    return Ge(rhs, lhs, scale)


def Gt(lhs, rhs, scale=1.):
    """Construct lhs > rhs with satisfaction margin (lhs - rhs) / scale."""
    return Comparison(_affine(lhs) - rhs, strict=True, scale=scale)


def Lt(lhs, rhs, scale=1.):
    """Construct lhs < rhs with satisfaction margin (rhs - lhs) / scale."""
    return Gt(rhs, lhs, scale)


@dataclass(frozen=True, eq=False)
class Junction(_Symbolic):
    """Combine output formulas through conjunction or disjunction.

    Conjunction takes the minimum child satisfaction margin, while disjunction
    takes the maximum. Pointwise Boolean evaluation also preserves each affine
    comparison's strictness. Construct these formulas with And or Or.

    Args:
        op (str): 'and' or 'or'.
        children (tuple): At least two affine comparisons or Boolean formulas.
    """
    op: str
    children: tuple

    def __post_init__(self):
        if self.op not in ('and', 'or') or len(self.children) < 2 or any(
                not isinstance(c, (Comparison, Junction)) for c in self.children):
            raise ValueError('And/Or require at least two property expressions')

    def margin(self, output, relu=False):
        """Combine child satisfaction margins using minimum for And and maximum for Or."""
        result = self.children[0].margin(output, relu)
        for child in self.children[1:]:
            other = child.margin(output, relu)
            if relu:
                result = (result - torch.relu(result - other) if self.op == 'and'
                          else other + torch.relu(result - other))
            else:
                result = (torch.minimum(result, other) if self.op == 'and'
                          else torch.maximum(result, other))
        return result

    def evaluate(self, output):
        """Evaluate the Boolean formula on each network output."""
        result = self.children[0].evaluate(output)
        for child in self.children[1:]:
            other = child.evaluate(output)
            result = result & other if self.op == 'and' else result | other
        return result

    def to_dict(self):
        """Return the Boolean operator and its child formulas as serialisable data."""
        return dict(op=self.op, children=[c.to_dict() for c in self.children])


def And(*children):
    """Construct a conjunction of at least two output formulas."""
    return Junction('and', tuple(children))


def Or(*children):
    """Construct a disjunction of at least two output formulas."""
    return Junction('or', tuple(children))


def obligations(formula):
    """Extract outer conjunctive requirements while retaining nested disjunctions."""
    if isinstance(formula, Junction) and formula.op == 'and':
        return tuple(item for child in formula.children for item in obligations(child))
    return (formula,)


@dataclass(frozen=True, eq=False)
class Box:
    """Define a closed input region through coordinatewise lower and upper endpoints.

    Endpoints must be finite, ordered and have matching nonempty shapes. They
    are stored as detached float64 CPU tensors; bounding routines convert them
    to model coordinates with outward rounding.

    Args:
        lower (Tensor or sequence): Lower endpoint, excluding the batch dimension.
        upper (Tensor or sequence): Upper endpoint with the same shape.
    """
    lower: torch.Tensor
    upper: torch.Tensor

    def __post_init__(self):
        lower = torch.as_tensor(self.lower, dtype=torch.float64).detach().cpu().clone()
        upper = torch.as_tensor(self.upper, dtype=torch.float64).detach().cpu().clone()
        if lower.shape != upper.shape or lower.ndim < 1 or lower.numel() == 0 or not (
                torch.isfinite(lower).all() and torch.isfinite(upper).all() and (lower <= upper).all()):
            raise ValueError('Box endpoints must be finite, ordered, nonempty, and have matching shapes')
        object.__setattr__(self, 'lower', lower)
        object.__setattr__(self, 'upper', upper)

    def to_dict(self):
        """Return the input endpoints as lists for manifests and checkpoints."""
        return dict(lower=self.lower.tolist(), upper=self.upper.tolist())


@dataclass(frozen=True, eq=False)
class PropertySpec:
    """Associate an output formula with the input regions where it must hold.

    The specification requires the formula to hold for every input in every
    supplied box. Task examples and property regions are independent inputs to
    training.

    Args:
        name (str): Nonempty specification name, unique within a wrapper.
        formula (Comparison or Junction): Affine or Boolean output formula.
        domains (Iterable[Box]): Nonempty collection of closed input regions.
    """
    name: str
    formula: object
    domains: tuple

    def __post_init__(self):
        domains = tuple(self.domains)
        if not isinstance(self.name, str) or not self.name or not isinstance(
                self.formula, (Comparison, Junction)) or not domains or any(
                not isinstance(d, Box) for d in domains):
            raise ValueError('PropertySpec requires a name, formula, and nonempty Box domains')
        object.__setattr__(self, 'domains', domains)

    def to_dict(self):
        """Return the specification name, formula and input boxes as serialisable data."""
        return dict(name=self.name, formula=self.formula.to_dict(),
                    domains=[d.to_dict() for d in self.domains])


class _AffineFormula(nn.Module):
    """Compile shared affine leaves and their Boolean satisfaction-margin reduction.

    Bounding the leaves separately allows inexpensive monotone aggregation of
    their lower margins. Disjunctions can be weaker than bounds obtained on the
    complete composed formula graph.
    """
    def __init__(self, formula, width, reference):
        super().__init__()
        self.nodes = []
        rows, offsets, strict, memo = [], [], [], {}

        def compile_node(node):
            if isinstance(node, Comparison):
                expression = node.expression * (1 / node.scale)
                key = ('atom', expression.coefficients, expression.offset, node.strict)
                if key in memo:
                    return memo[key]
                row, offset = expression.tensors(reference.new_zeros((1, width)))
                record = ('atom', len(rows))
                rows.append(row)
                offsets.append(offset)
                strict.append(node.strict)
            else:
                children = tuple(compile_node(child) for child in node.children)
                key = (node.op, children)
                if key in memo:
                    return memo[key]
                record = (node.op, children)
            memo[key] = len(self.nodes)
            self.nodes.append(record)
            return memo[key]

        self.root = compile_node(formula)
        self.register_buffer('weight', torch.stack(rows))
        self.register_buffer('bias', torch.stack(offsets))
        self.register_buffer('strict', torch.tensor(strict, device=reference.device))

    def forward(self, output, *, boolean=False):
        """Evaluate all affine leaves together, preserving strict comparison ties."""
        atomic = torch.nn.functional.linear(output, self.weight, self.bias)
        if boolean:
            atomic = torch.where(self.strict, atomic > 0, atomic >= 0)
        return self.reduce(atomic, boolean=boolean)

    def reduce(self, atomic, *, boolean=False):
        """Reduce atomic values using the compiled Boolean structure or min/max margins."""
        values = []
        for op, children in self.nodes:
            if op == 'atom':
                result = atomic[:, children:children + 1]
            else:
                result = torch.cat([values[i] for i in children], dim=1)
                if boolean:
                    result = (result.all(dim=1, keepdim=True) if op == 'and'
                              else result.any(dim=1, keepdim=True))
                else:
                    result = (result.amin(dim=1, keepdim=True) if op == 'and'
                              else result.amax(dim=1, keepdim=True))
            values.append(result)
        return values[self.root]


class _CompiledFormula(nn.Module):
    """Compile affine comparisons and Boolean operations into a fixed PyTorch graph."""
    def __init__(self, formula, width, reference, relu):
        super().__init__()
        self.op, self.relu = None, relu
        if isinstance(formula, Comparison):
            row, offset = (formula.expression * (1 / formula.scale)).tensors(reference.new_zeros((1, width)))
            self.register_buffer('weight', row.unsqueeze(0))
            self.register_buffer('bias', offset.reshape(1))
        else:
            self.op = formula.op
            self.children_nodes = nn.ModuleList([_CompiledFormula(c, width, reference, relu)
                                                 for c in formula.children])

    def forward(self, output):
        if self.op is None:
            return torch.nn.functional.linear(output, self.weight, self.bias)
        result = self.children_nodes[0](output)
        for child in self.children_nodes[1:]:
            other = child(output)
            if self.relu:
                result = (result - torch.relu(result - other) if self.op == 'and'
                          else other + torch.relu(result - other))
            else:
                result = (torch.minimum(result, other) if self.op == 'and'
                          else torch.maximum(result, other))
        return result


class PropertyGraph(nn.Module):
    """Append a fixed satisfaction-margin computation to a network.

    The composed graph shares the original network's trainable weights and adds
    no trainable parameters. Bounding its output gives a lower satisfaction
    margin over an input region.

    Args:
        model (torch.nn.Module): Network whose outputs the formula references.
        formula (Comparison or Junction): Output specification to compile.
        relu (bool): Whether to express minimum and maximum through ReLU operations.
        output_width (int, optional): Number of network outputs; required when it cannot be
            inferred from the last layer.
    """
    def __init__(self, model, formula, relu=False, output_width=None):
        super().__init__()
        self.model, self.formula, self.relu = model, formula, relu
        if output_width is None:
            last = model[-1] if isinstance(model, nn.Sequential) else model
            output_width = getattr(last, 'out_features', None)
        if type(output_width) is not int or output_width < 1:
            raise ValueError('Supply output_width for a custom verification graph')
        reference = next(model.parameters(), None)
        if reference is None:
            reference = next(model.buffers(), torch.zeros(()))
        self.compiled = _CompiledFormula(formula, output_width, reference, relu)

    def forward(self, inputs):
        """Evaluate the network and return its compiled satisfaction margin."""
        return self.compiled(self.model(inputs))


from .vnnlib import load_vnnlib_property

__all__ = ['Outputs', 'OutputIndex', 'And', 'Or', 'Ge', 'Le', 'Gt', 'Lt',
           'Box', 'PropertySpec', 'PropertyGraph', 'load_vnnlib_property']
