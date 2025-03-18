import numpy as np
from functions import kernel_function

n_cells = 100
n_nodes = n_cells + 1
x_nodes = np.linspace(0, 1, n_nodes)

n_kernels = 5

p = -0.05
q = 0.1

j0 = 1
jN = 30

j_values = np.linspace(j0, jN, n_kernels)

output = kernel_function(x_nodes, j0, p, q)

# testing
len(output) == len(x_nodes)

test_cosine = kernel_function(x_nodes, 1, 0, q)
analytic_cosine = np.cos(2 * np.pi * 1 * q * x_nodes)

np.all(test_cosine == analytic_cosine)

np.allclose(test_cosine, analytic_cosine)

# if numerical, won't be exactly the same, need to account for small changes with a threshold value
# np.linalg.norm(test_cosine - analytic_cosine) < threshold
