from .operators import *
import numpy as np

class Conv(Operator):
    """Circular convolution operator.

        H = Conv(psf)

    Return a linear operator whose effect is to convolve its argument by the
    point spread function `psf`.

    """
    def __init__(self, psf, real=True):
        assert isinstance(real, bool)
        self.real = real
        self.mtf = np.fft.fftn(psf)

    def __repr__(self):
        return 'Conv(…)'

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
