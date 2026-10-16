# -*- coding: utf-8 -*-

import numpy as np
from .operators import Operator, Identity, add, compose
from .utils import adapt_multiplier, scale_array

def conj(x):
    if np.iscomplexobj(x):
        return np.conj(x)
    else:
        return x

class Diag(Operator):
    """
    Diagonal operator.

    Calling:
        D = Diag(wgt)

    """
    def __init__(self, coefs):
        self.coefs = np.asarray(coefs)

    def __repr__(self):
        return 'Diag(…)'

    def __neg__(self):
        return Diag(-self.coefs)

    # self + other
    def __add__(self, other):
        if not isinstance(other, Operator):
            raise ValueError('only addition of operators is implemented')
        if isinstance(other, Diag):
            return Diag(self.coefs + other.coefs)
        if isinstance(other, Identity):
            return Diag(self.coefs + 1)
        if isinstance(other, Scaled):
            alpha = other.multiplier
            if alpha == 0:
                return self
            A = other.operator
            if isinstance(A, Identity):
                coefs = self.coefs
                alpha = adapt_multiplier(coefs.dtype, alpha)
                return Diag(coefs + alpha)
            elif isinstance(A, Diag):
                return Diag(self.coefs + scale_array(alpha, A.coefs))
        return add(self, other)

    # other + self
    def __radd__(self, other):
        # Addition of operators is symmetric.
        return __add__(self, other)

    def scaled_by(self, alpha):
        assert np.isscalar(alpha)
        if alpha == 1:
            return self
        else:
            return Diag(scale_array(alpha, self.coefs))

    def apply(self, x):
        return self.coefs * x

    def apply_adjoint(self, x):
        return conj(self.coefs) * x

    def apply_inverse(self, x):
        return x / self.coefs

    def apply_inverse_adjoint(self, x):
        return x / conj(self.coefs)

    def try_simplify_prod(self, terms, i):
        for j in (i - 1, i + 1):
            if not (0 <= j < len(terms)): continue
            other = terms[j]
            if isinstance(other, Diag):
                # The predecessor or successor of this operator is also a
                # diagonal operator.
                coefs = self.coefs * other.coefs
                return simplify_prod(min(i, j), 2, Diag(coefs))
        return None
