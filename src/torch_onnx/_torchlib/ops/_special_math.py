# PyTorch-derived algorithms: LICENSE-PYTORCH, LICENSE-CEPHES, LICENSE-SPECIAL.
"""Portable ONNX arithmetic for special functions without standard ONNX kernels."""

from __future__ import annotations

import math

import onnx_ir as ir
import torch

from torch_onnx._ir_builder import TranslationContext
from torch_onnx._torchlib.ops import _promotion
from torch_onnx._torchlib.ops import _special_coefficients as coef


def prepare(op, *values, kind="INT_TO_FLOAT"):
    computation, result = _promotion.dtypes(op, *values, kind=kind)
    return tuple(
        _promotion.cast(op, value, computation) for value in values
    ), _promotion._ONNX_DTYPES[result]


def finish(op, value, dtype):
    return value if value.dtype == dtype else op.Cast(value, to=dtype)


def constant(op, value, like):
    return op.constant(value, dtype=like.dtype)


def polynomial(op, x, coefficients):
    result = constant(op, coefficients[0], x)
    for coefficient in coefficients[1:]:
        result = op.Add(op.Mul(result, x), coefficient)
    return result


def chebyshev(op, x, coefficients):
    current, previous = constant(op, coefficients[0], x), constant(op, 0, x)
    older = previous
    for coefficient in coefficients[1:]:
        older, previous, current = (
            previous,
            current,
            op.Add(op.Sub(op.Mul(x, current), previous), coefficient),
        )
    return op.Mul(op.Sub(current, older), 0.5)


def hypot(op, x, y):
    x, y = op.Abs(x), op.Abs(y)
    largest, smallest = op.Max(x, y), op.Min(x, y)
    ratio = op.Div(
        smallest, op.Where(op.Equal(largest, 0), constant(op, 1, x), largest)
    )
    result = op.Mul(largest, op.Sqrt(op.Add(op.Mul(ratio, ratio), 1)))
    return op.Where(op.Or(op.IsInf(x), op.IsInf(y)), constant(op, math.inf, x), result)


def bessel_i(op, x, order=0):
    absolute = op.Abs(x)
    precision = "DOUBLE" if x.dtype == ir.DataType.DOUBLE else "FLOAT"
    a = getattr(coef, f"I{order}_A_{precision}")
    b = getattr(coef, f"I{order}_B_{precision}")
    small_x = op.Min(absolute, 8)
    large_x = op.Max(absolute, 8)
    small = chebyshev(op, op.Sub(op.Div(small_x, 2), 2), a)
    if order:
        small = op.Mul(small, small_x)
    large = op.Div(
        chebyshev(op, op.Sub(op.Div(constant(op, 32, x), large_x), 2), b),
        op.Sqrt(large_x),
    )
    result = op.Where(op.LessOrEqual(absolute, 8), small, large)
    result = op.Mul(op.Exp(absolute), result)
    if order:
        result = op.Where(op.Less(x, 0), op.Neg(result), result)
    return result


def bessel_j(op, x, order=0):
    absolute = op.Abs(x)
    small_x, large_x = op.Min(absolute, 5), op.Max(absolute, 5)
    square = op.Mul(small_x, small_x)
    roots = (
        (5.78318596294678452118, 30.4712623436620863991)
        if not order
        else (14.6819706421238932572, 49.2184563216946036703)
    )
    small = op.Mul(
        op.Mul(op.Sub(square, roots[0]), op.Sub(square, roots[1])),
        op.Div(
            polynomial(op, square, getattr(coef, f"J{order}_RP")),
            polynomial(op, square, getattr(coef, f"J{order}_RQ")),
        ),
    )
    if order:
        small = op.Mul(small, small_x)
    else:
        small = op.Where(
            op.Less(small_x, 1e-5), op.Sub(constant(op, 1, x), op.Div(square, 4)), small
        )
    inverse = op.Div(constant(op, 5, x), large_x)
    squared_inverse = op.Mul(inverse, inverse)
    p = op.Div(
        polynomial(op, squared_inverse, getattr(coef, f"J{order}_PP")),
        polynomial(op, squared_inverse, getattr(coef, f"J{order}_PQ")),
    )
    q = op.Div(
        polynomial(op, squared_inverse, getattr(coef, f"J{order}_QP")),
        polynomial(op, squared_inverse, getattr(coef, f"J{order}_QQ")),
    )
    phase = op.Sub(large_x, math.pi * (0.75 if order else 0.25))
    large = op.Div(
        op.Mul(
            op.Sub(op.Mul(p, op.Cos(phase)), op.Mul(op.Mul(inverse, q), op.Sin(phase))),
            math.sqrt(2 / math.pi),
        ),
        op.Sqrt(large_x),
    )
    result = op.Where(op.LessOrEqual(absolute, 5), small, large)
    return op.Where(op.Less(x, 0), op.Neg(result), result) if order else result


def ndtri(op, x):
    reflected = op.Greater(x, 1 - math.exp(-2))
    y = op.Where(reflected, op.Sub(constant(op, 1, x), x), x)
    centered = op.Sub(y, 0.5)
    square = op.Mul(centered, centered)
    central = op.Mul(
        op.Add(
            centered,
            op.Mul(
                centered,
                op.Mul(
                    square,
                    op.Div(
                        polynomial(op, square, coef.NDTRI_P0),
                        polynomial(op, square, coef.NDTRI_Q0),
                    ),
                ),
            ),
        ),
        math.sqrt(2 * math.pi),
    )
    # Clipping only protects the unused branch and endpoint intermediates.
    tail_y = op.Min(
        op.Max(
            y,
            constant(
                op,
                torch.finfo(_promotion._DTYPES[x.dtype]).tiny
                * torch.finfo(_promotion._DTYPES[x.dtype]).eps,
                x,
            ),
        ),
        math.exp(-2),
    )
    root = op.Sqrt(op.Mul(op.Log(tail_y), -2))
    inverse = op.Reciprocal(root)
    p1 = op.Div(
        polynomial(op, inverse, coef.NDTRI_P1), polynomial(op, inverse, coef.NDTRI_Q1)
    )
    p2 = op.Div(
        polynomial(op, inverse, coef.NDTRI_P2), polynomial(op, inverse, coef.NDTRI_Q2)
    )
    tail = op.Sub(
        op.Sub(root, op.Div(op.Log(root), root)),
        op.Mul(inverse, op.Where(op.Less(root, 8), p1, p2)),
    )
    result = op.Where(
        op.Greater(y, math.exp(-2)), central, op.Where(reflected, tail, op.Neg(tail))
    )
    result = op.Where(op.Equal(x, 0), constant(op, -math.inf, x), result)
    result = op.Where(op.Equal(x, 1), constant(op, math.inf, x), result)
    return op.Where(
        op.Or(op.Less(x, 0), op.Greater(x, 1)), constant(op, math.nan, x), result
    )


def erfinv(op, x):
    probability = op.Mul(op.Sub(constant(op, 1, x), op.Abs(x)), 0.5)
    result = op.Mul(
        op.Mul(op.Neg(op.Sign(x)), ndtri(op, probability)), 1 / math.sqrt(2)
    )
    return op.Where(op.Equal(x, 0), x, result)


def spherical_j0(op, x):
    square = op.Mul(x, x)
    small = op.Add(
        op.Mul(
            square,
            polynomial(
                op,
                square,
                (1 / 6227020800, -1 / 39916800, 1 / 362880, -1 / 5040, 1 / 120, -1 / 6),
            ),
        ),
        1,
    )
    result = op.Where(op.Less(op.Abs(x), 0.5), small, op.Div(op.Sin(x), x))
    return op.Where(op.IsInf(x), constant(op, 0, x), result)


def lgamma(op, x):
    reflected = op.Less(x, 0)
    small_positive = op.And(op.Greater(x, 0), op.Less(x, 0.5))
    y = op.Where(reflected, op.Sub(constant(op, 1, x), x), x)
    y = op.Where(small_positive, op.Add(x, 1), y)
    inverse = op.Reciprocal(y)
    lanczos = op.Div(
        polynomial(op, inverse, coef.LANCZOS_NUM[::-1]),
        polynomial(op, inverse, coef.LANCZOS_DENOM[::-1]),
    )
    result = op.Add(
        op.Mul(
            op.Sub(y, 0.5), op.Sub(op.Log(op.Add(y, 5.524680040776729583740234375)), 1)
        ),
        op.Log(lanczos),
    )
    fraction = op.Sub(x, op.Floor(x))
    reflection = op.Sub(
        op.Sub(
            constant(op, math.log(math.pi), x),
            op.Log(op.Abs(op.Sin(op.Mul(fraction, math.pi)))),
        ),
        result,
    )
    result = op.Where(reflected, reflection, result)
    result = op.Where(small_positive, op.Sub(result, op.Log(x)), result)
    poles = op.And(op.LessOrEqual(x, 0), op.Equal(x, op.Floor(x)))
    return op.Where(op.Or(poles, op.IsInf(x)), constant(op, math.inf, x), result)


def digamma(op, x):
    reflected = op.Less(x, 0)
    y = op.Where(reflected, op.Sub(constant(op, 1, x), x), x)
    shift = op.Less(y, 10)
    correction = constant(op, 0, x)
    for k in range(10):
        correction = op.Sub(correction, op.Reciprocal(op.Add(y, k)))
    large = op.Where(shift, op.Add(y, 10), y)
    inverse = op.Reciprocal(large)
    squared_inverse = op.Mul(inverse, inverse)
    asymptotic = op.Mul(
        squared_inverse,
        polynomial(
            op,
            squared_inverse,
            (1 / 12, -691 / 32760, 1 / 132, -1 / 240, 1 / 252, -1 / 120, 1 / 12),
        ),
    )
    result = op.Sub(op.Sub(op.Log(large), op.Mul(inverse, 0.5)), asymptotic)
    result = op.Add(result, op.Where(shift, correction, constant(op, 0, x)))
    fraction = op.Sub(x, op.Ceil(x))
    result = op.Where(
        reflected,
        op.Sub(
            result, op.Div(constant(op, math.pi, x), op.Tan(op.Mul(fraction, math.pi)))
        ),
        result,
    )
    result = op.Where(
        op.And(reflected, op.Equal(x, op.Floor(x))), constant(op, math.nan, x), result
    )
    return op.Where(op.Equal(x, 0), op.Div(constant(op, -1, x), x), result)


def nextafter(op, x, y, source_dtype):
    bits = {
        ir.DataType.FLOAT16: (10, -14, 15),
        ir.DataType.BFLOAT16: (7, -126, 127),
        ir.DataType.FLOAT: (23, -126, 127),
        ir.DataType.DOUBLE: (52, -1022, 1023),
    }
    mantissa, low, high = bits[source_dtype]
    absolute = op.Abs(x)
    exponent = op.Floor(
        op.Div(op.Log(op.Max(absolute, constant(op, 2.0**low, x))), math.log(2))
    )
    exponent = op.Clip(exponent, constant(op, low, x), constant(op, high, x))
    power = op.Pow(constant(op, 2, x), exponent)
    exponent = op.Where(op.Less(absolute, power), op.Sub(exponent, 1), exponent)
    exponent = op.Where(
        op.GreaterOrEqual(absolute, op.Mul(power, 2)), op.Add(exponent, 1), exponent
    )
    exponent = op.Clip(exponent, constant(op, low, x), constant(op, high, x))
    power = op.Pow(constant(op, 2, x), exponent)
    upward = op.Greater(y, x)
    toward_zero = op.Xor(op.Greater(x, 0), upward)
    boundary = op.And(
        op.And(toward_zero, op.Equal(absolute, power)), op.Greater(exponent, low)
    )
    exponent = op.Sub(op.Sub(exponent, mantissa), op.cast_like(boundary, x))
    step = op.Pow(constant(op, 2, x), exponent)
    result = op.Add(x, op.Where(upward, step, op.Neg(step)))
    result = op.Where(op.Equal(result, 0), op.Mul(x, constant(op, 0, x)), result)
    max_value = torch.finfo(_promotion._DTYPES[source_dtype]).max
    result = op.Where(
        op.IsInf(x),
        op.Where(
            op.Greater(x, 0), constant(op, max_value, x), constant(op, -max_value, x)
        ),
        result,
    )
    result = op.Where(op.Equal(x, y), y, result)
    return op.Where(op.Or(op.IsNaN(x), op.IsNaN(y)), op.Add(x, y), result)


def erfc(op, x):
    absolute = op.Abs(x)
    small_x = op.Min(absolute, 8)
    large_x = op.Max(absolute, 8)
    medium = op.Div(
        polynomial(op, small_x, coef.ERFC_P), polynomial(op, small_x, coef.ERFC_Q)
    )
    large = op.Div(
        polynomial(op, large_x, coef.ERFC_R), polynomial(op, large_x, coef.ERFC_S)
    )
    tail = op.Mul(
        op.Exp(op.Neg(op.Mul(absolute, absolute))),
        op.Where(op.Less(absolute, 8), medium, large),
    )
    result = op.Where(
        op.Less(absolute, 1), op.Sub(constant(op, 1, x), op.Erf(absolute)), tail
    )
    return op.Where(op.Less(x, 0), op.Sub(constant(op, 2, x), result), result)


def _any(op, mask):
    return op.Greater(
        op.ReduceSum(op.Cast(mask, to=ir.DataType.INT64), keepdims=False), 0
    )


def _iterate(op, states, step, *, limit=2000):
    """Converge elementwise recurrences with tensor carry, never ONNX sequences."""
    from torch_onnx._torchlib.ops import common

    name = f"special_iteration_{len(op.builder.graph)}"
    iteration = ir.Value(
        name=name + "_i", shape=ir.Shape([]), type=ir.TensorType(ir.DataType.INT64)
    )
    condition = ir.Value(
        name=name + "_condition",
        shape=ir.Shape([]),
        type=ir.TensorType(ir.DataType.BOOL),
    )
    carried = [
        ir.Value(name=f"{name}_state_{i}", shape=value.shape, type=value.type)
        for i, value in enumerate(states)
    ]
    graph = ir.Graph([iteration, condition, *carried], [], nodes=[], name=name)
    child = TranslationContext(op.builder.subgraph(graph))
    following, active = step(child, iteration, *carried)
    for previous, value in zip(states, following, strict=True):
        value.type = previous.type
        value.shape = previous.shape
    common.finish_subgraph(child, graph, [_any(child, active), *following])
    maximum = None if limit is None else op.constant(limit, dtype=ir.DataType.INT64)
    return op.builder.node(
        "Loop",
        [maximum, _any(op, states[-1]), *states],
        attributes={"body": graph},
        output_types=[value.type for value in states],
    )


def _broadcast(op, a, b):
    # Broadcast via a zero of b's shape; do not multiply possibly infinite b by 0.
    a = op.Add(a, op.cast_like(op.ConstantOfShape(op.Shape(b)), a))
    return a, op.Expand(b, op.Shape(a))


def zeta(op, s, q):
    s, q = _broadcast(op, s, q)
    nan, inf = constant(op, math.nan, s), constant(op, math.inf, s)
    valid_q = op.Or(
        op.Greater(q, 0),
        op.And(op.Not(op.Equal(q, op.Floor(q))), op.Equal(s, op.Floor(s))),
    )
    valid = op.And(op.Greater(s, 1), valid_q)
    a = op.Where(valid, q, constant(op, 1, q))
    initial = op.Pow(a, op.Neg(s))
    early = op.Expand(op.constant(False, dtype=ir.DataType.BOOL), op.Shape(a))
    epsilon = torch.finfo(_promotion._DTYPES[s.dtype]).eps / 2

    def step(body, i, position, total, last, converged, active):
        next_position = body.Add(position, 1)
        term = body.Pow(next_position, body.Neg(s))
        value = body.Add(total, term)
        small = body.Less(body.Abs(term), body.Mul(value, epsilon))
        more = body.Or(body.Less(i, 8), body.LessOrEqual(next_position, 9))
        next_active = body.And(active, body.And(body.Not(small), more))
        return [
            body.Where(active, next_position, position),
            body.Where(active, value, total),
            body.Where(active, term, last),
            body.Or(converged, body.And(active, small)),
            next_active,
        ], next_active

    a, total, last, early, _ = _iterate(
        op, [a, initial, initial, early, valid], step, limit=None
    )
    result = op.Add(total, op.Mul(last, op.Sub(op.Div(a, op.Sub(s, 1)), 0.5)))
    rising = constant(op, 1, s)
    coefficients = (
        12.0,
        -720.0,
        30240.0,
        -1209600.0,
        47900160.0,
        -1.8924375803183791606e9,
        7.47242496e10,
        -2.950130727918164224e12,
        1.1646782814350067249e14,
        -4.5979787224074726105e15,
        1.8152105401943546773e17,
        -7.1661652561756670113e18,
    )
    for i, coefficient in enumerate(coefficients):
        rising = op.Mul(rising, op.Add(s, 2 * i))
        last = op.Div(last, a)
        result = op.Add(result, op.Div(op.Mul(rising, last), coefficient))
        rising = op.Mul(rising, op.Add(s, 2 * i + 1))
        last = op.Div(last, a)
    result = op.Where(early, total, result)
    result = op.Where(valid, result, nan)
    result = op.Where(
        op.And(op.LessOrEqual(q, 0), op.Equal(q, op.Floor(q))), inf, result
    )
    result = op.Where(op.Less(s, 1), nan, result)
    return op.Where(op.Equal(s, 1), inf, result)


def _log1pmx(op, delta):
    # Avoid cancellation in log(1+delta)-delta close to zero.
    coefficients = tuple(((-1.0) ** (n + 1)) / n for n in range(14, 1, -1))
    near = op.Mul(op.Mul(delta, delta), polynomial(op, delta, coefficients))
    ordinary = op.Sub(op.Log(op.Add(delta, 1)), delta)
    return op.Where(op.Less(op.Abs(delta), 0.1), near, ordinary)


def _gamma_log_factor(op, a, x):
    direct = op.Sub(op.Sub(op.Mul(a, op.Log(x)), x), lgamma(op, a))
    inverse = op.Reciprocal(a)
    correction = op.Mul(
        inverse,
        polynomial(
            op, op.Mul(inverse, inverse), (-1 / 1680, 1 / 1260, -1 / 360, 1 / 12)
        ),
    )
    delta = op.Div(op.Sub(x, a), a)
    logarithm = op.Sub(
        op.Add(
            op.Mul(a, _log1pmx(op, delta)), op.Mul(op.Log(op.Div(a, 2 * math.pi)), 0.5)
        ),
        correction,
    )
    return op.Where(op.Greater(a, 10), logarithm, direct)


def _gamma_asymptotic(op, a, x, complement):
    delta = op.Div(op.Sub(x, a), a)
    eta = op.Mul(op.Sign(delta), op.Sqrt(op.Max(op.Mul(_log1pmx(op, delta), -2), 0)))
    z = op.Mul(eta, op.Sqrt(op.Div(a, 2)))
    result = op.Mul(erfc(op, z if complement else op.Neg(z)), 0.5)
    inverse = op.Reciprocal(a)
    total = constant(op, 0, a)
    count, degree = (4, 12) if a.dtype == ir.DataType.DOUBLE else (2, 7)
    for k in reversed(range(count)):
        row = getattr(coef, f"IGAMMA_D{k}")[:degree]
        total = op.Add(op.Mul(total, inverse), polynomial(op, eta, row[::-1]))
    correction = op.Div(
        op.Mul(op.Exp(op.Mul(op.Mul(eta, eta), op.Mul(a, -0.5))), total),
        op.Sqrt(op.Mul(a, 2 * math.pi)),
    )
    return op.Add(result, correction) if complement else op.Sub(result, correction)


def igamma(op, a, x, *, complement=False):
    a, x = _broadcast(op, a, x)
    dtype = a.dtype
    eps = torch.finfo(_promotion._DTYPES[dtype]).eps / 2
    tiny = 1e-300 if dtype == ir.DataType.DOUBLE else 1e-30
    valid = op.And(
        op.And(op.Greater(a, 0), op.Greater(x, 0)),
        op.Not(op.Or(op.IsInf(a), op.IsInf(x))),
    )
    safe_a = op.Where(valid, a, constant(op, 1, a))
    safe_x = op.Where(valid, x, constant(op, 1, x))
    delta = op.Div(op.Sub(safe_x, safe_a), safe_a)
    asymptotic = op.And(
        valid, op.And(op.GreaterOrEqual(safe_a, 10000), op.Less(op.Abs(delta), 0.1))
    )
    lower = op.Less(safe_x, op.Add(safe_a, 1))
    active = op.And(valid, op.Not(asymptotic))
    series_active = op.And(active, lower)
    one = op.Expand(constant(op, 1, a), op.Shape(a))

    def series(body, i, term, total, running):
        denominator = body.Add(safe_a, body.Add(body.cast_like(i, safe_a), 1))
        next_term = body.Mul(term, body.Div(safe_x, denominator))
        next_total = body.Add(total, next_term)
        more = body.And(
            running,
            body.Greater(body.Abs(next_term), body.Mul(body.Abs(next_total), eps)),
        )
        return [
            body.Where(running, next_term, term),
            body.Where(running, next_total, total),
            more,
        ], more

    _, summed, _ = _iterate(op, [one, one, series_active], series)
    log_factor = _gamma_log_factor(op, safe_a, safe_x)
    factor = op.Exp(log_factor)
    p = op.Mul(summed, op.Exp(op.Sub(log_factor, op.Log(safe_a))))
    fraction_active = op.And(active, op.Not(lower))
    b = op.Where(fraction_active, op.Sub(op.Add(safe_x, 1), safe_a), one)
    c = op.Expand(constant(op, 1 / tiny, a), op.Shape(a))
    d = op.Reciprocal(b)

    def fraction(body, i, b, c, d, h, running):
        iteration = body.Add(body.cast_like(i, safe_a), 1)
        coefficient = body.Neg(body.Mul(iteration, body.Sub(iteration, safe_a)))
        b_next = body.Add(b, 2)
        d_next = body.Add(body.Mul(coefficient, d), b_next)
        c_next = body.Add(b_next, body.Div(coefficient, c))
        d_next = body.Reciprocal(
            body.Where(
                body.Less(body.Abs(d_next), tiny), constant(body, tiny, a), d_next
            )
        )
        c_next = body.Where(
            body.Less(body.Abs(c_next), tiny), constant(body, tiny, a), c_next
        )
        change = body.Mul(c_next, d_next)
        h_next = body.Mul(h, change)
        more = body.And(running, body.Greater(body.Abs(body.Sub(change, 1)), eps))
        return [
            body.Where(running, new, old)
            for new, old in zip(
                [b_next, c_next, d_next, h_next], [b, c, d, h], strict=True
            )
        ] + [more], more

    _, _, _, h, _ = _iterate(op, [b, c, d, d, fraction_active], fraction)
    q = op.Mul(factor, h)
    result = op.Where(
        lower, op.Sub(one, p) if complement else p, q if complement else op.Sub(one, q)
    )
    if complement:
        small_active = op.And(
            active,
            op.And(lower, op.And(op.Less(safe_a, 1), op.LessOrEqual(safe_x, 1.1))),
        )
        zero = op.Expand(constant(op, 0, a), op.Shape(a))

        def complementary_series(body, i, fac, total, running):
            n = body.Add(body.cast_like(i, safe_a), 1)
            next_fac = body.Mul(fac, body.Div(body.Neg(safe_x), n))
            term = body.Div(next_fac, body.Add(safe_a, n))
            next_total = body.Add(total, term)
            more = body.And(
                running,
                body.Greater(body.Abs(term), body.Mul(body.Abs(next_total), eps)),
            )
            return [
                body.Where(running, next_fac, fac),
                body.Where(running, next_total, total),
                more,
            ], more

        _, alt_sum, _ = _iterate(op, [one, zero, small_active], complementary_series)
        t = op.Sub(op.Mul(safe_a, op.Log(safe_x)), lgamma1p(op, safe_a))
        small_q = op.Sub(
            op.Neg(expm1(op, t)), op.Mul(op.Exp(op.Add(log_factor, safe_x)), alt_sum)
        )
        result = op.Where(small_active, small_q, result)
    result = op.Where(
        asymptotic, _gamma_asymptotic(op, safe_a, safe_x, complement), result
    )
    nan = constant(op, math.nan, a)
    low, high = (
        constant(op, 0 if complement else 1, a),
        constant(op, 1 if complement else 0, a),
    )
    result = op.Where(op.Or(op.IsNaN(a), op.IsNaN(x)), nan, result)
    result = op.Where(op.IsInf(x), low, result)
    result = op.Where(op.IsInf(a), op.Where(op.IsInf(x), nan, high), result)
    result = op.Where(op.Equal(x, 0), high, result)
    result = op.Where(op.Equal(a, 0), op.Where(op.Greater(x, 0), low, nan), result)
    return op.Where(op.Or(op.Less(a, 0), op.Less(x, 0)), nan, result)


def expm1(op, x):
    # Taylor evaluation preserves tiny residuals that Exp(x)-1 would round away.
    coefficients = tuple(1 / math.factorial(n) for n in reversed(range(1, 16)))
    near = op.Mul(x, polynomial(op, x, coefficients))
    return op.Where(op.Less(op.Abs(x), 0.5), near, op.Sub(op.Exp(x), 1))


def lgamma1p(op, x):
    # log Gamma(1+x) = -EulerGamma*x + sum((-1)^n*zeta(n)*x^n/n).
    coefficients = (
        0.07143294629536133,
        -0.0769325164113522,
        0.08335384054610899,
        -0.09095401714582904,
        0.10009945751278182,
        -0.11133426586956473,
        0.12550966952474307,
        -0.14404989676884614,
        0.16955717699740813,
        -0.2073855510286741,
        0.27058080842778454,
        -0.4006856343865313,
        0.8224670334241133,
    )
    near = op.Mul(
        x, op.Add(op.Mul(x, polynomial(op, x, coefficients)), -0.5772156649015328606)
    )
    return op.Where(op.Less(op.Abs(x), 0.1), near, lgamma(op, op.Add(x, 1)))
