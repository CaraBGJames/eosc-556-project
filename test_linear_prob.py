import numpy as np
from utils import kernel_function

n_cells = 100
n_nodes = n_cells + 1
x_nodes = np.linspace(0, 1, n_nodes)

n_kernels = 5
p = -0.05
q = 0.1
j = 1


# make little functions, if true just keeps running, if false just throws an error but keeps going
def test_kernel_size():
    output = kernel_function(x_nodes, j, p, q)
    assert len(output) == len(x_nodes)


def test_kernel_decay():
    test_cosine = kernel_function(x_nodes, 1, 0, q)
    analytic_cosine = np.cos(2 * np.pi * 1 * q * x_nodes)
    print(
        f"norm numeric: {np.linalg.norm(test_cosine):1.2e}, norm analytic: {np.linalg.norm(analytic_cosine):1.2e}"
    )
    assert np.allclose(
        test_cosine, analytic_cosine
    )  # raises an assertion error if not true


def test_p_positive():
    kernel_function(x_nodes, j, p * -1, q)
