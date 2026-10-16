# -*- coding: utf-8 -*-

import numpy as np
from .operators import Operator, Identity, add, compose

# NOTE Diff is a singleton, hence its key is a constant given by the id of
# its name (for unicity).
_D_name = 'D'
_D_key = id(_D_name)
_DtD_name = 'DtD'
_DtD_key = id(_DtD_name)

class Diff(Operator):
    """
    Finite difference operator. The effect of this operator is to compute the
    finite differences along all dimensions of the input array.
    """
    def __init__(self):
        pass

    def __repr__(self):
        return _D_name

    # Finite difference operator is a singleton so two operator of this kind
    # are equal if they have the same type.
    def __eq__(self, other):
        return isinstance(other, Diff)

    def key(self):
        return _D_key

    # NOTE The differences along each dimension are stored in a single array.
    #      As a result, the output has an extra dimension of length the number
    #      of dimensions and the differences along each dimension must be
    #      padded to the same dimensions as the input. These extra values are
    #      set to zero so this amounts to having the finite differences
    #      computed along a given dimension of length `n` by a `n × n` matrix
    #      whose last line is filled with zeros. For example, for `n = 5`:
    #
    #              ( -1  1  0  0  0 )              ( -1  0  0  0  0 )
    #              (  0 -1  1  0  0 )              (  1 -1  0  0  0 )
    #          D = (  0  0 -1  1  0 )   ==>   D' = (  0  1 -1  0  0 )
    #              (  0  0  0 -1  1 )              (  0  0  1 -1  0 )
    #              (  0  0  0  0  0 )              (  0  0  0  1  0 )
    #
    #     Hence, the last column of the transpose operator is filled with
    #     zeros which amounts to ignoring the last entry along each dimension
    #     when applying the adjoint operator.
    def apply(self, x):
        x = np.asarray(x)
        dims = x.shape
        rank = len(dims) # number of dimensions, also x.ndim
        y = np.zeros(dims + (rank,), dtype=x.dtype) # to store the result
        if rank == 0:
            return y
        # NOTE Surprisingly `x[1:,...] - x[0:-1,...]` is a bit faster than
        #      `np.diff(x, axis=0)` and the same holds for the second axis.
        if dims[0] >= 2:
            y[0:-1,...,0] = x[1:,...] - x[0:-1,...]
        if rank == 1:
            return y
        if dims[1] >= 2:
            y[:,0:-1,...,1] = x[:,1:,...] - x[:,0:-1,...]
        if rank == 2:
            return y
        if dims[2] >= 2:
            y[:,:,0:-1,...,2] = x[:,:,1:,...] - x[:,:,0:-1,...]
        if rank == 3:
            return y
        if dims[3] >= 2:
            y[:,:,:,0:-1,...,3] = x[:,:,:,1:,...] - x[:,:,:,0:-1,...]
        if rank == 4:
            return y
        raise ValueError('too many dimensions')

    def apply_adjoint(self, x):
        x = np.asarray(x)
        dims = x.shape
        rank = len(dims) - 1 # number of dimensions of the result
        if rank < 1 or dims[-1] != rank:
            raise ValueError('invalid shape')
        y = np.zeros(dims[0:-1], dtype=x.dtype) # to store the result
        if rank == 0:
            return y
        # NOTE Index range `0:-1` taken in the input `x` for each dimension
        #      amounts to discard the last entry as explained in the
        #      description of the operator above.
        if dims[0] >= 2:
            d = x[0:-1,...,0]
            y[1:,...] += d
            y[0:-1,...] -= d
        if rank == 1:
            return y
        if dims[1] >= 2:
            d = x[:,0:-1,...,1]
            y[:,1:,...] += d
            y[:,0:-1,...] -= d
        if rank == 2:
            return y
        if dims[2] >= 2:
            d = x[:,:,0:-1,...,2]
            y[:,:,1:,...] += d
            y[:,:,0:-1,...] -= d
        if rank == 3:
            return y
        if dims[3] >= 2:
            d = x[:,:,:,0:-1,...,3]
            y[:,:,:,1:,...] += d
            y[:,:,:,0:-1,...] -= d
        if rank == 4:
            return y
        raise ValueError('too many dimensions')

class GramDiff(Operator):
    """
    Operator `D'⋅D` where `D = Diff()` is the finite dimensional operator and
    `D' = adj(D)` is its transpose.
    """
    def __init__(self):
        pass

    def __repr__(self):
        return _DtD_name

    # Two operators of this kind are equal if they have the same type.
    def __eq__(self, other):
        return isinstance(other, GramDiff)

    def key(self):
        return _DtD_key

    # DtD is symmetric.
    def adj(self):
        return self

    def apply(self, x):
        return DtD(x)

    def apply_adjoint(self, x):
        return DtD(x)

def DtD(x):
    """
    Return the result of D'⋅D⋅x where D is a (multi-dimensional) finite
    difference operator and D' is its transpose.
    """
    dims = x.shape
    y = np.zeros(dims, dtype=x.dtype) # to store the result
    rank = x.ndim # number of dimensions
    if rank == 0: return y
    if dims[0] >= 2:
        d = x[1:,...] - x[0:-1,...]
        y[1:,...] += d
        y[0:-1,...] -= d
    if rank == 1: return y
    if dims[1] >= 2:
        d = x[:,1:,...] - x[:,0:-1,...]
        y[:,1:,...] += d
        y[:,0:-1,...] -= d
    if rank == 2: return y
    if dims[2] >= 2:
        d = x[:,:,1:,...] - x[:,:,0:-1,...]
        y[:,:,1:,...] += d
        y[:,:,0:-1,...] -= d
    if rank == 3: return y
    if dims[3] >= 2:
        d = x[:,:,:,1:,...] - x[:,:,:,0:-1,...]
        y[:,:,:,1:,...] += d
        y[:,:,:,0:-1,...] -= d
    if rank == 4: return y
    if dims[4] >= 2:
        d = x[:,:,:,:,1:,...] - x[:,:,:,:,0:-1,...]
        y[:,:,:,:,1:,...] += d
        y[:,:,:,:,0:-1,...] -= d
    if rank == 5: return y
    raise ValueError('too many dimensions')
