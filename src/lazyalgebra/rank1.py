from .operators import *
from .operations import inner
from .utils import scale_array
import numpy as np

class Rank1(Operator):
    """Rank one operator..

        A = Rank1(u, v=u)

    Return a rank one operator `A = u⋅v'` such that `A*x -> (v'⋅x)⋅u` where
    `v'⋅x` is the scalar product of `v` by `x`.

    """
    def __init__(self, u, v=None):
        self.u = np.asarray(u)
        self.v = v is None and self.u or np.asarray(v)

    def __repr__(self):
        return 'Rank1(…)'

    def apply(self, x):
        return scale_array(inner(self.v, x), self.u)

    def apply_adjoint(self, x):
        # FIXME u may be complex
        return scale_array(inner(self.u, x), self.v)
