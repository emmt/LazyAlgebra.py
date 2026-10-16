# -*- coding: utf-8 -*-

import numpy as np
from numpy import isscalar
from copy import copy, deepcopy
from .utils import adapt_multiplier, scale_array

# TODO adapt multiplier precision
# TODO overload __pow__
# TODO attributes key and bits are invariant and shall be computed at construction time
# TODO make attributes private (by pre-pending an underscore)
#
# NOTE Class constructors always return an instance of the class, so
#      simplifications and digestion of arguments can only be done by functions
#      or methods that eventually call constructors. For these reasons, the
#      end-users shall not directly call the constructors of non-bare operators
#      like `Scaled`, `Adjoint`, `Inverse`, `Sum`, and `Product`.
#
class Operator(object):
    """
    Operators are special objects that represent linear mappings of variables
    specified as multi-dimensional arrays.
    """

    # Unary +
    def __pos__(self):
        return self

    # Unary -
    def __neg__(self):
        return self.scaled_by(-1)

    # self(x)
    def __call__(self, x):
        return self.apply(x)

    # self == other
    def __eq__(self, other):
        return self is other

    # self + other
    def __add__(self, other):
        if not isinstance(other, Operator):
            raise ValueError('only addition of operators is implemented')
        if isinstance(other, Scaled) and other.multiplier == 0:
            return self
        else:
            return add(self, other)

    # self - other
    def __sub__(self, other):
        if not isinstance(other, Operator):
            raise ValueError('only subtraction of operators is implemented')
        if isinstance(other, Scaled) and other.multiplier == 0:
            return self
        else:
            return add(self, other.scaled_by(-1))

    # self * other
    def __mul__(self, other):
        if isinstance(other, Operator):
            return compose(self, other)
        elif np.isscalar(other):
            return self.scaled_by(other)
        else:
            return self.apply(other)

    # other * self
    def __rmul__(self, other):
        if isinstance(other, Operator):
            return compose(other, self)
        elif np.isscalar(other):
            return self.scaled_by(other)
        raise ValueError('invalid left operand for right multiplication of an operator')

    # self / other
    def __truediv__(self, other):
        if isinstance(other, Operator):
            return compose(self, other.inv())
        elif np.isscalar(other):
            return self.scaled_by(1/other)
        raise ValueError('invalid right operand to left-division of an operator')

    # other/self
    def __rtruediv__(self, other):
        if isinstance(other, Operator):
            return compose(other, self.inv())
        elif np.isscalar(other):
            return self.inv().scaled_by(other)
        raise ValueError('invalid right operand to left-division of an operator')

    def bits(self):
        """
        Return the bit mask of operator `self` representing whether the operator
        is the inverse (bit 1) and/or the adjoint (bit 2) of some bare operator.
        """
        return 0

    def key(self):
        """
        Return the ordering key of the bare operator in `self`. Two bare operators
        having different keys shall not be equal.
        """
        return id(self)

    def adj(self):
        """
        Return the adjoint of operator `self`.
        """
        return Adjoint(self)

    def inv(self):
        """
        Return the inverse of operator `self`.
        """
        return Inverse(self)

    def descale(self):
        """
        Return the 2-tuple `(alpha, A)` with `alpha` a scalar and `A` and
        operator such that `alpha*A` is equivalent to `self`. If `self` is not
        a scaled operator, then `(1, self)` is returned.
        """
        return (1, self)

    def scaled_by(self, alpha):
        """
        Return operator multiplied by scalar multiplier `alpha`. The same
        result is given by the expression `alpha * self`.
        """
        assert np.isscalar(alpha)
        if alpha == 1:
            return self
        else:
            return Scaled(alpha, self)

    def apply(self, x):
        """
        Return the result of applying the operator to array-like object `x`.
        """
        raise NotImplementedError('`apply` method not implemented for this operator')

    def apply_adjoint(self, x):
        """
        Return the result of applying the adjoint of the operator to array-like
        object `x`.
        """
        raise NotImplementedError('`apply_adjoint` method not implemented for this operator')

    def apply_inverse(self, arg):
        """
        Return the result of applying the inverse of the operator to array-like
        object `x`.
        """
        raise NotImplementedError('`apply_inverse` method not implemented for this operator')

    def apply_inverse_adjoint(self, arg):
        """
        Return the result of applying the inverse-adjoint of the operator to
        array-like object `x`.
        """
        raise NotImplementedError('`apply_inverse_adjoint` method not implemented for this operator')

    def simplify_add(self, other):
        """
        Simplify the addition `self + other`. Return `None` if no
        simplification is possible.
        """
        if isinstance(other, Scaled) and other.multiplier == 0:
            return self
        else:
            return None

    def simplify_radd(self, other):
        """
        Simplify the addition `other + self`. Return `None` if no
        simplification is possible.
        """
        if isinstance(other, Scaled) and other.multiplier == 0:
            return self
        else:
            return None

    def simplify_mul(self, other):
        """
        Simplify the composition `self * other`. Return `None` if no
        simplification is possible.
        """
        if isinstance(other, Identity):
            return self
        elif isinstance(other, Scaled) and isinstance(other.operator, Identity):
            return self.scaled_by(other.multiplier)
        else:
            return None

    def simplify_rmul(self, other):
        """
        Simplify the composition of `other * self`. Return `None` if no
        simplification is possible.
        """
        if isinstance(other, Identity):
            return self
        elif isinstance(other, Scaled) and isinstance(other.operator, Identity):
            return self.scaled_by(other.multiplier)
        else:
            return None

class Adjoint(Operator):
    """
    Adjoint operator `Adjoint(A)` represents the inverse of `A`. The end-users
    shall never directly call the constructor but rather call `adj(A)` or
    `A.adj()`.
    """
    def __init__(self, A):
        assert isinstance(A, Operator)
        assert not isinstance(A, Adjoint)
        assert not isinstance(A, Inverse)
        assert not isinstance(A, Scaled)
        self.parent = A

    def __eq__(self, other):
        return (self is other) or (isinstance(other, Adjoint)
                                   and self.parent == other.parent)

    def bits(self):
        return self.parent.bits() ^ 2

    def key(self):
        return self.parent.key()

    def adj(self):
        # Simplify adjoint of adjoint.
        return self.parent

    def inv(self):
        if isinstance(self.parent, Inverse):
            # Simplify inverse of adjoint inverse.
            return self.parent.parent.adj()
        else:
            return Inverse(self)

    def apply(self, x):
        return self.parent.apply_adjoint(x)

    def apply_adjoint(self, x):
        return self.parent.apply(x)

    def apply_inverse(self, x):
        return self.parent.apply_inverse_adjoint(x)

    def apply_inverse_adjoint(self, x):
        return self.parent.apply_inverse(x)

class Inverse(Operator):
    """
    Inverse operator `Inverse(A)` represents the inverse of `A`. The end-users
    shall never directly call the constructor but rather call `inv(A)` or
    `A.inv()`.
    """
    def __init__(self, A):
        assert isinstance(A, Operator)
        assert not isinstance(A, Inverse)
        assert not isinstance(A, Scaled)
        self.parent = A

    def __eq__(self, other):
        return (self is other) or (isinstance(other, Inverse)
                                   and self.parent == other.parent)

    def bits(self):
        return self.parent.bits() ^ 1

    def key(self):
        return self.parent.key()

    def adj(self):
        if isinstance(self.parent, Adjoint):
            # Simplify adjoint of inverse adjoint.
            return self.parent.parent.inv()
        else:
            return Adjoint(self)

    def inv(self):
        # Simplify inverse of inverse.
        return self.parent

    def apply(self, x):
        return self.parent.apply_inverse(x)

    def apply_adjoint(self, x):
        return self.parent.apply_inverse_adjoint(x)

    def apply_inverse(self, x):
        return self.parent.apply(x)

    def apply_inverse_adjoint(self, x):
        return self.parent.apply_adjoint(x)

class Scaled(Operator):
    """
    Scaled operator `Scaled(alpha, A)` represents `alpha*A` where `alpha` shall
    be a non-unit scalar and `A` a non-scaled operator. The end-users shall
    never directly call the constructor but use expressions `alpha*A` or
    `A.scaled_by(alpha)` instead.
    """
    def __init__(self, alpha, A):
        assert np.isscalar(alpha) and alpha != 1
        assert isinstance(A, Operator)
        assert not isinstance(A, Scaled)
        self.multiplier = alpha
        self.operator = A

    def __eq__(self, other):
        if self is other:
            return True
        elif not isinstance(other, Operator):
            return False
        elif isinstance(other, Scaled):
            return (self.multiplier == other.multiplier
                    and self.operator == other.operator)
        else:
            return (self.multiplier == 1 and self.operator == other)

    def bits(self):
        return self.operator.bits()

    def key(self):
        return self.operator.key()

    def descale(self):
        return (self.multiplier, self.operator)

    def scaled_by(self, alpha):
        assert np.isscalar(alpha)
        return self.operator.scaled_by(alpha * self.multiplier)

    def adj(self):
        return self.operator.adj().scaled_by(np.conj(self.multiplier))

    def inv(self):
        return self.operator.inv().scaled_by(1/self.multiplier)

    def apply(self, arg):
        return scale_array(self.multiplier, self.operator.apply(arg))

    def apply_adjoint(self, arg):
        return scale_array(np.conj(self.multiplier),
                           self.operator.apply_adjoint(arg))

    def apply_inverse(self, arg):
        return scale_array(1/self.multiplier,
                           self.operator.apply_inverse(arg))

    def apply_inverse_adjoint(self, arg):
        return scale_array(1/np.conj(self.multiplier),
                           self.operator.apply_inverse_adjoint(arg))

class Identity(Operator):
    """
    Identity operator.
    """

    def __call__(self, other):
        return np.asarray(other)

    def __mul__(self, other):
        if isinstance(other, Operator):
            return other
        elif np.isscalar(other):
            return self.scaled_by(other)
        else:
            return np.asarray(other)

    def __eq__(self, other):
        return (self is other) or isinstance(other, Identity)

    def __repr__(self):
        return 'Id'

    def key(self):
        return -1 # to make sure identity is always ordered first

    def adj(self):
        return self

    def inv(self):
        return self

    def apply(self, x):
        return np.asarray(x)

    def apply_adjoint(self, arg):
        return np.asarray(x)

    def apply_inverse(self, arg):
        return np.asarray(x)

    def apply_inverse_adjoint(self, arg):
        return np.asarray(x)

Id = Identity()

class Symbolic(Operator):
    """
    Symbolic operator. Identified by a name given at creation time.
    """
    def __init__(self, name):
        assert isinstance(name, str)
        self.name = name

    def __eq__(self, other):
        return (self is other) or (isinstance(other, Symbolic)
                                   and self.name == other.name)

    def __repr__(self):
        return self.name

    def key(self):
        return id(self.name)

#--------------------------------------------------------------- Sum of terms -

def add(A, B):
    """
    Add operators `A` and `B`.
    """
    # Collect the terms of the sum.
    terms = []
    collect_sum_terms(terms, A)
    collect_sum_terms(terms, B)
    # Sort terms in-place.
    terms.sort()
    # Group terms that can be represented by a single scaled operator.
    n = len(terms)
    for i in range(n):
        A = terms[i]
        if A is None:
            continue
        for j in range(i + 1, n):
            B = terms[j]
            if B.key != A.key or B.bits != A.bits:
                break # subsequent terms cannot be grouped with A
            if B.bare_operator == A.bare_operator:
                A.multiplier += B.multiplier # update A in-place
                terms[j] = None # delete B
    # Delete terms that have a zero multiplier (keep at least one).
    n = 0
    X = None # first term to keep
    for i in range(len(terms)):
        A = terms[i]
        if A is None:
            continue
        if A.multiplier == 0:
            terms[i] = None
            if X is None:
                X = A
        else:
            n += 1
            if n == 1:
                X = A
    if n  > 1:
        # Return the sum of non deleted terms converted back to operators.
        return Sum(list(A.as_operator() for A in terms if A is not None))
    else:
        return X.as_operator()

def collect_sum_terms(terms, A):
    if isinstance(A, Sum):
        for B in A.terms:
            collect_sum_terms(terms, SumTerm(B))
    else:
        terms.append(SumTerm(A))
    return None

class Sum(Operator):
    """
    Operator `Sum([A, B, ...])` represents the sum of operators `A`, `B`, etc.
    The end-users shall never directly call the constructor but rather call `A
    + B + ...`.
    """
    def __init__(self, terms):
        assert len(terms) >= 2
        self.terms = terms

    def __eq__(self, other):
        return (self is other) or (isinstance(other, Sum)
                                   and self.terms == other.terms)

    def adj(self):
        # Distribute adjoint over the terms of a sum.
        # FIXME sort terms
        return Sum(list(A.adj() for A in self.terms))

    def apply(self, x):
        # TODO Write a function like BLAS `axpy(a, x, y=None)`.
        y = None
        for A in self.terms:
            if y is None:
                y = A.apply(x)
            else:
                y += A.apply(x)
        return y

    def apply_adjoint(self, arg):
        y = None
        for A in self.terms:
            if y is None:
                y = A.apply_adjoint(x)
            else:
                y += A.apply_adjoint(x)
        return y

class SumTerm(object):
    def __init__(self, A):
        bits = 0
        alpha = 1
        while True:
            if isinstance(A, Scaled):
                beta = A.multiplier
                if (bits & 2) == 2:
                    beta = np.conj(beta)
                if (bits & 1) == 1:
                    beta = 1/beta
                alpha *= beta
                A = A.operator
            elif isinstance(A, Adjoint):
                bits ^= 2
                A = A.parent
            elif isinstance(A, Inverse):
                bits ^= 1
                A = A.parent
            else:
                break
        self.multiplier = alpha
        self.bare_operator = A
        self.bits = bits
        self.key = A.key()

    def scaled_by(self, alpha):
        assert np.isscalar(alpha)
        if alpha == 1:
            return self
        else:
            other = copy(self)
            other.multiplier *= alpha
            return other

    def as_operator(self):
        """
        Return operator corresponding to sum term `self`.
        """
        A = self.bare_operator
        if (self.bits & 2) == 2:
            A = Adjoint(A)
        if (self.bits & 1) == 1:
            A = Inverse(A)
        if self.multiplier == 1:
            return A
        else:
            return Scaled(self.multiplier, A)

    # The following is needed to sort the terms of a sum.
    def __lt__(self, other):
        return (self.key < other.key) or (self.key == other.key
                                          and self.bits < other.bits)

#----------------------------------------------------------- Product of terms -

# TODO Simplify some products on the fly can be done by sub-class methods.
# Helper function for compose operators `A` and `B`. It is not checked that
# `A` and `B` are operators.
def compose(A, B):
    """
    Compose operators `A` and `B`.
    """
    # Collect the terms of the sum updating the multiplier in the process.
    terms = []
    alpha = collect_prod_terms(1, terms, A)
    alpha = collect_prod_terms(alpha, terms, B)
    # Discard identities and build the operator resulting form the composition.
    terms = list(A for A in terms if not isinstance(A, Identity))
    n = len(terms)
    if n == 0:
        A = Id
    elif n == 1:
        A = terms[0]
    else:
        A = Product(terms)
    # Return the resulting operator scaled by the factorized multiplier.
    return A.scaled_by(alpha)

def collect_prod_terms(alpha, terms, A):
    """
    Append operator `A` in-place to the list of `terms` and return the updated
    scalar multiplier `alpha`. If `A` is a scaled operator, its operator is
    appended to `terms` and its multiplier times `alpha` is returned. If `A` is
    a product, its terms are appended to `terms` and `alpha` is returned.
    """
    if isinstance(A, Scaled):
        return collect_prod_terms(alpha * A.multiplier, terms, A.operator)
    elif isinstance(A, Product):
        for B in A.terms:
            alpha = collect_prod_terms(alpha, terms, B)
    else:
        terms.append(A)
    return alpha


class Product(Operator):
    """
    Operator `Product([A, B, ...])` represents the composition of operators
    `A`, `B`, etc. The end-users shall never directly call the constructor but
    rather call `A * B * ...`.
    """
    def __init__(self, terms):
        assert len(terms) >= 2
        self.terms = terms

    def __eq__(self, other):
        return (self is other) or (isinstance(other, Product)
                                   and self.terms == other.terms)

    def adj(self):
        # Distribute adjoint over the terms of a product.
        terms = []
        alpha = 1
        for A in reversed(self.terms):
            beta, B = A.descale()
            alpha *= np.conj(beta)
            terms.append(B.adj())
        return Product(terms).scaled_by(alpha)

    def inv(self):
        # Distribute inverse over the terms of a product.
        terms = []
        alpha = 1
        for A in reversed(self.terms):
            beta, B = A.descale()
            alpha /= beta
            terms.append(B.inv())
        return Product(terms).scaled_by(alpha)

    def apply(self, x):
        for A in reversed(self.terms):
            x = A.apply(x)
        return x

    def apply_adjoint(self, x):
        for A in self.terms:
            x = A.apply_adjoint(x)
        return x

    def apply_inverse(self, x):
        for A in self.terms:
            x = A.apply_inverse(x)
        return x

    def apply_inverse_adjoint(self, x):
        for A in reversed(self.terms):
            x = A.apply_inverse_adjoint(x)
