# -*- coding: utf-8 -*-

import numpy as np

from .operators import Operator, Adjoint, Inverse, Scaled, Sum, Product

def inv(x):
    """
    Return the inverse of `x`. If `x` is a scalar of an array, `1/x` is returned. Otherwise,
    `x` must be an operator.
    """
    if isinstance(x, Operator):
        # Call dedicated method.
        return x.inv()
    else:
        # Assume scalar or array.
        return 1/x

def adj(x):
    """
    Return the adjoint of `x`. If `x` is a scalar of an array, the element-wise conjugate of
    `x` is returned. Otherwise, `x` must be an operator.
    """
    if isinstance(x, Operator):
        # Call dedicated method.
        return x.adj()
    else:
        # Assume scalar or array.
        return np.conj(x)

def mul(x, y):
    """
    Return the "dot" product of `x` and `y`. If `x` and `y` are both operators, their
    composition is returned. If one of `x` or `y` is an operator while the other is a
    scalar, a scaled operator is returned. If `x` is an operator and `y` an array, the
    result of applying the operator is returned.If `x` and `y` are scalars or arrays, their
    element-wise multiplication is returned.
    """
    if isinstance(x, Operator):
        if isinstance(y, Operator):
            # Product of two operators.
            alpha, A = x.descale()
            beta, B = y.descale()
            return Product(A, B).scale_by(alpha * beta)
        elif np.isscalar(y):
            alpha, A = x.descale()
            return A.scale_by(alpha * y)
        else:
            return x.apply(np.asarray(y))
    elif isinstance(y, Operator):
        if np.isscalar(x):
            beta, B = y.descale()
            return B.scale_by(x * beta)
        else:
            raise ValueError('invalid argument types for `mul`')
    else:
        return x * y

def inner(x, y):
    """
    Return the "inner" product of `x` by `y`.
    """
    return sum(x.ravel() * y.ravel())
