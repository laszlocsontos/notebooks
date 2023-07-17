import numpy as np
import scipy.stats as stats
import scipy.fft as fft


def spectral_entropy(y) -> float:
    """Computes the spectral entropy of a time series."""
    n = len(y)
    yf = fft.fft(y)
    ps = np.abs(yf[0:n//2])**2
    ps_norm = ps / ps.sum()
    return stats.entropy(ps_norm)
