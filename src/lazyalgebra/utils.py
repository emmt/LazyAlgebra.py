# -*- coding: utf-8 -*-

import numpy as np

def adapt_multiplier(T, alpha):
    """
    Return multiplier `alpha` with the numerical precision of type `T`.
    """
    if T == np.float32 or T == np.complex64:
        if np.iscomplexobj(alpha):
            return np.complex64(alpha)
        else:
            return np.float32(alpha)
    elif T == np.float128 or T == np.complex256:
        if np.iscomplexobj(alpha):
            return np.complex256(alpha)
        else:
            return np.float128(alpha)
    else:
        if np.iscomplexobj(alpha):
            return np.complex128(alpha)
        else:
            return np.float64(alpha)

def scale_array(alpha, x):
    """
    Compute `alpha*x` efficiently and taking care of preserving the floating-point type of
    `x`. `alpha` must be a scalar and `x` must be an array.
    """
    if not isinstance(x, np.ndarray):
        x = np.asarray(x)
    if alpha == 0:
        # Assume a strong zero.
        return np.zeros(x.shape, x.dtype)
    elif alpha == 1:
        return x
    elif alpha == -1:
        return -x
    else:
        return np.multiply(adapt_multiplier(x.dtype, alpha), x)
