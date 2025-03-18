import numpy as np
import warnings


def kernel_function(x, j, p, q):
    """
    Function to create decaying, oscillatory kernels

    Parameters
    __________
    x: numpy.ndarray
        location of the nodes of our mesh

    j: float
        kernel index

    p: float
        how quickly the function decays (if negative) or grows (if positive)

    q: float
        how oscillatory our function in

    Returns
    _______
    numpy.ndarray
        kernel function

    """
    if p > 0:
        warnings.warn(
            "The value of p is positive, {p}, this will cause exponential growth of the kernel"
        )
    return np.exp(j * p * x) * np.cos(2 * np.pi * j * q * x)
