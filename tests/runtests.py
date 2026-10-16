# -*- coding: utf-8 -*-

import numpy as np
import os.path
import sys
import unittest

SRCDIR = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'src'))
#print(SRCDIR)
sys.path.append(SRCDIR)

import lazyalgebra
from lazyalgebra import *

def inner(x, y):
    """
    Return inner product of `x` and `y` considered as vectors. Arguments must have the same
    shape and must be both real or both complex.
    """
    if x.shape != y.shape:
        raise Exception('arguments must have the same shape')
    x_is_complex = np.iscomplexobj(x)
    if np.iscomplexobj(y) ^ x_is_complex:
        raise Exception('arguments must be both real or both complex')
    if x_is_complex:
        # NOTE `np.inner` does take the conjugate.
        return np.inner(np.conj(x.ravel()), y.ravel())
    else:
        return np.inner(x.ravel(), y.ravel())

class TestLazyAlgebra(unittest.TestCase):

    def test_diff(self):
        D = Diff()
        DtD = lazyalgebra.diff.GramDiff()
        self.assertEqual(D, Diff())
        self.assertTrue(D == Diff())
        self.assertEqual(repr(D), 'D')
        # Generate integer valued random array for exact computations.
        x = np.rint(200*np.random.random_sample((3,4,5)) - 100)
        Dx = D*x
        self.assertTrue((np.diff(x, axis=0) == Dx[0:-1,...,0]).all())
        self.assertTrue((np.diff(x, axis=1) == Dx[:,0:-1,...,1]).all())
        self.assertTrue((np.diff(x, axis=2) == Dx[:,:,0:-1,...,2]).all())
        DtDx = D.adj()*Dx
        w = lazyalgebra.diff.DtD(x)
        self.assertTrue((DtDx == w).all())
        # Check __call__
        self.assertTrue((Dx == D(x)).all())
        self.assertTrue((DtDx == DtD(x)).all())
        y = np.rint(200*np.random.random_sample(Dx.shape) - 100)
        ytDx = inner(y, D*x)
        xtDty = inner(x, D.adj()*y)
        self.assertTrue(ytDx == xtDty)

    def test_identity(self):
        self.assertEqual(Id, lazyalgebra.operators.Identity())

    #def test_split(self):
    #    s = 'hello world'
    #    self.assertEqual(s.split(), ['hello', 'world'])
    #    # check that s.split fails when the separator is not a string
    #    with self.assertRaises(TypeError):
    #        s.split(2)

if __name__ == '__main__':
    unittest.main()
