from .operators import *
import numpy as np

class Conv(Operator):
    """
    Circular convolution operator.

    Calling:
        H = Conv(psf)

    """
    def __init__(self, psf, real=True):
        self.real = real
        self.mtf = np.fft.fftn(psf)

    def apply(self, x):
        z = np.fft.ifftn(self.mtf * np.fft.fftn(x))
        if self.real:
            return np.real(z)
        else:
            return z

    def apply_adjoint(self, x):
        z = np.fft.ifftn(np.conj(self.mtf) * np.fft.fftn(x))
        if self.real:
            return np.real(z)
        else:
            return z
