"""Import bounded-box specifications into the output-property DSL.

The supported subset preserves strict comparisons and unsafe/safe polarity;
unsupported input regions or operators are rejected rather than approximated.
"""
import math
from pathlib import Path
import re
import torch
from CTRAIN.properties import Affine, And, Or, Ge, Le, Gt, Lt, Box, PropertySpec


def _parse(text):
    """Read parenthesized statements without interpreting their semantics."""
    tokens = re.findall(r'\(|\)|[^\s()]+', re.sub(r';[^\n]*', '', text))
    stack, result = [], []
    for token in tokens:
        if token == '(':
            stack.append([])
        elif token == ')':
            if not stack:
                raise ValueError('Unbalanced specification')
            item = stack.pop()
            (stack[-1] if stack else result).append(item)
        elif stack:
            stack[-1].append(token)
        else:
            raise ValueError('Unexpected specification token')
    if stack:
        raise ValueError('Unbalanced specification')
    return result


def _join(constructor, children):
    if not children:
        raise ValueError('Empty Boolean expressions are unsupported')
    return children[0] if len(children) == 1 else constructor(*children)


def _affine(term, declared):
    if isinstance(term, str):
        if term in declared:
            return Affine(((int(term[2:]), 1.),))
        try:
            return Affine(offset=float(term))
        except (ValueError, TypeError) as error:
            raise ValueError(f'Unknown or non-finite affine term: {term}') from error
    if not isinstance(term, list) or len(term) < 2:
        raise ValueError('Expected an affine expression')
    op, *args = term
    values = [_affine(arg, declared) for arg in args]
    if op == '+':
        return sum(values, Affine())
    if op == '-':
        return -values[0] if len(values) == 1 else values[0] - sum(values[1:], Affine())
    if op == '*' and len(values) == 2:
        left, right = values
        if not left.coefficients:
            return right * left.offset
        if not right.coefficients:
            return left * right.offset
    if op == '/' and len(values) == 2 and not values[1].coefficients and values[1].offset != 0:
        return values[0] * (1 / values[1].offset)
    raise ValueError(f'Unsupported/nonlinear affine operator: {op}')


def _output_formula(term, declared, negate=False):
    if not isinstance(term, list) or not term:
        raise ValueError('Expected an output predicate')
    op, *args = term
    if op == 'not' and len(args) == 1:
        return _output_formula(args[0], declared, not negate)
    if op in ('and', 'or'):
        conjunction = (op == 'and') != negate
        return _join(And if conjunction else Or,
                     [_output_formula(arg, declared, negate) for arg in args])
    if op in ('<=', '>=', '<', '>') and len(args) == 2:
        comparison = {'<=': Le, '>=': Ge, '<': Lt, '>': Gt}
        if negate:
            op = {'<=': '>', '>=': '<', '<': '>=', '>': '<='}[op]
        return comparison[op](*[_affine(arg, declared) for arg in args])
    raise ValueError(f'Unsupported output predicate: {op}; use inequalities and And/Or/Not')


def _conjuncts(term):
    if isinstance(term, list) and term and term[0] == 'and':
        return [child for arg in term[1:] for child in _conjuncts(arg)]
    return [term]


def _variables(term):
    if isinstance(term, str):
        return {term} if re.fullmatch(r'[XY]_\d+', term) else set()
    return set().union(*(_variables(arg) for arg in term))


def _input_clauses(term):
    """Distribute input And/Or into closed-box conjunctions."""
    if isinstance(term, list) and term and term[0] in ('and', 'or'):
        if len(term) < 2:
            raise ValueError('Empty input Boolean expression')
        children = [_input_clauses(arg) for arg in term[1:]]
        if term[0] == 'or':
            if sum(len(child) for child in children) > 1024:
                raise ValueError('Input region expansion exceeds 1024 boxes')
            return [clause for child in children for clause in child]
        clauses = [[]]
        for child in children:
            # ponytail: cap materialized DNF; use symbolic regions for larger unions.
            if len(clauses) * len(child) > 1024:
                raise ValueError('Input region expansion exceeds 1024 boxes')
            clauses = [left + right for left in clauses for right in child]
        return clauses
    return [[term]]


def _input_box(clause, inputs, shape):
    lower, upper = [-math.inf] * len(inputs), [math.inf] * len(inputs)
    for term in clause:
        if not isinstance(term, list) or len(term) != 3 or term[0] not in ('<=', '>='):
            raise ValueError('Input predicates must be closed axis-aligned box bounds')
        difference = _affine(term[1], inputs) - _affine(term[2], inputs)
        if len(difference.coefficients) != 1:
            raise ValueError('Input predicates must constrain exactly one coordinate')
        index, coefficient = difference.coefficients[0]
        if coefficient == 0:
            raise ValueError('Input predicates must have a nonzero coordinate coefficient')
        value = -difference.offset / coefficient
        if not math.isfinite(value):
            raise ValueError('Non-finite input bound')
        if (term[0] == '<=') == (coefficient > 0):
            upper[index] = min(upper[index], value)
        else:
            lower[index] = max(lower[index], value)
    return Box(torch.tensor(lower, dtype=torch.double).reshape(shape),
               torch.tensor(upper, dtype=torch.double).reshape(shape))


def _read_vnnlib(path, name=None, input_shape=None, output_semantics='unsafe'):
    if output_semantics not in ('unsafe', 'safe'):
        raise ValueError("output_semantics must be 'unsafe' or 'safe'")
    path = Path(path)
    declarations, assertions = set(), []
    for statement in _parse(path.read_text()):
        if not statement:
            raise ValueError('Empty VNNLIB statement')
        op, *args = statement
        if op == 'declare-const' and len(args) == 2 and args[1] == 'Real':
            variable = args[0]
            if not isinstance(variable, str) or not re.fullmatch(r'[XY]_\d+', variable) or variable in declarations:
                raise ValueError('Declarations require unique X_i/Y_i real variables')
            declarations.add(variable)
        elif op == 'assert' and len(args) == 1:
            assertions.extend(_conjuncts(args[0]))
        elif op == 'set-logic' and args == ['QF_LRA']:
            continue
        else:
            raise ValueError(f'Unsupported VNNLIB statement: {op}')
    inputs = {v for v in declarations if v.startswith('X_')}
    outputs = declarations - inputs
    if not inputs or not outputs or inputs != {f'X_{i}' for i in range(len(inputs))} or outputs != {
            f'Y_{i}' for i in range(len(outputs))}:
        raise ValueError('Declare contiguous input/output indices starting at zero')
    predicates, input_predicates = [], []
    for term in assertions:
        variables = _variables(term)
        if not variables <= declarations:
            raise ValueError('Predicate references undeclared variables')
        if variables & inputs:
            if variables & outputs:
                raise ValueError('Mixed input/output predicates are unsupported')
            input_predicates.append(term)
        else:
            predicates.append(term)
    shape = (len(inputs),) if input_shape is None else tuple(input_shape)
    if not shape or any(type(n) is not int or n < 1 for n in shape) or math.prod(shape) != len(inputs):
        raise ValueError('input_shape must contain one entry per declared input coordinate')
    domains = [_input_box(clause, inputs, shape)
               for clause in _input_clauses(['and', *input_predicates])]
    if not predicates:
        raise ValueError('VNNLIB must contain an output predicate')
    combined = predicates[0] if len(predicates) == 1 else ['and', *predicates]
    formula = _output_formula(combined, outputs, output_semantics == 'unsafe')
    return PropertySpec(path.stem if name is None else name, formula, domains), len(outputs)


def load_vnnlib_property(path, *, name=None, input_shape=None, output_semantics='unsafe'):
    """Import a user-supplied VNNLIB property without benchmark presets.

    Args:
        path (str | pathlib.Path): Local VNNLIB file with contiguous `X_i` and
            `Y_i` real declarations, finite closed input bounds and output logic.
        name (str | None): Property name; defaults to the source filename stem.
        input_shape (tuple[int, ...] | None): Shape excluding batch. Defaults to
            a flat vector; specify this to reshape image coordinates in index order.
        output_semantics (str): `'unsafe'` (default) complements the conjunction
            of output assertions, as verification benchmarks describe violations.
            `'safe'` imports the output assertions as desired conditions directly.

    Returns:
        PropertySpec: Desired formula over the declared input boxes. Preserves
        strictness and ties when complementing unsafe inequalities.

    Raises:
        ValueError: Unsupported syntax, undeclared variables, nonlinear terms,
            non-box/mixed input conditions, unbounded or inconsistent inputs.

    Supports affine output expressions, `<`, `<=`, `>` and `>=`, nested `and`,
    `or` and `not`. Exact equality and general SMT-LIB are unsupported. Input
    constraints support And/Or combinations of closed coordinate bounds, yielding
    a finite union of at most 1024 bounded boxes. Mixed input/output logic and coupled input
    inequalities are rejected. No normalization is added. This
    small adapter avoids the bundled verifier parser's strict-comparison rewrite
    and compiled pickle cache, preserving source semantics without cache reads.
    Numeric constants and affine arithmetic use floating-point values; this is
    not an exact-rational or directed-rounding translation proof.
    """
    return _read_vnnlib(path, name, input_shape, output_semantics)[0]
