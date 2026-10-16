# -*- coding: utf-8 -*-

import numpy as np
from .operators import Operator, Identity, add, compose

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

    def __neg__(self):
        return Diag(-self.coefs)

    def scaled_by(self, alpha):
        assert np.isscalar(alpha)
        if alpha == 1:
            return self
        else:
            return Diag(alpha * self.coefs)

    def apply(self, x):
        return self.coefs * x

    def apply_adjoint(self, x):
        return conj(self.coefs) * x

    def apply_inverse(self, x):
        return x / self.coefs

    def apply_inverse_adjoint(self, x):
        return x / conj(self.coefs)

    def simplify_add(self, other):
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
                # FIXME adapt alpha
                return Diag(self.coefs + alpha)
            elif isinstance(A, Diag):
                return Diag(self.coefs + alpha * A.coefs)
        return None

    def simplify_radd(self, other):
        # Addition of operators is symmetric.
        return simplify_add(self, other)

    def simplify_mul(self, other):
        pass
