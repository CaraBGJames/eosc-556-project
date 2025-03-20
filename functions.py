import numpy as np
import warnings


def extract_resistivity_from_res2dinv(file_path):
    """
    Extracts electrode positions, resistivity and error data from a RES2DINV .dat file.

    This function reads a resistivity survey file in RES2DINV format and extracts:
    - Electrode coordinates (X, Z) for each measurement
    - Apparent resistivity values (Ωm)
    - Measurement error percentages

    Chargeability data (if present) is ignored.

    Parameters:
    ----------
    file_path : str
        Path to the .dat file containing the resistivity survey data.

    Returns:
    -------
    list of dict
        A list where each entry is a dictionary with:
        - 'coordinates': List of 4 tuples [(x1, z1), (x2, z2), (x3, z3), (x4, z4)]
        - 'resistivity': float (Apparent resistivity in Ωm)
        - 'error': float (Measurement error percentage)

    Example:
    -------
    >>> data = extract_resistivity_data("survey.dat")
    >>> print(data[0])
    {'electrodes': [(10.0, 0.0), (40.0, 0.0), (20.0, 0.0), (30.0, 0.0)],
     'resistivity': 82.437,
     'error': 12.183}
    """
    with open(file_path, "r") as file:
        lines = file.readlines()

    num_data_points = int(lines[6].strip())  # Number of data points

    # Read the data section
    electrode_positions = []
    resistivity_data = []
    data_start_line = 12
    for line in lines[data_start_line : data_start_line + num_data_points]:
        values = line.strip().split()
        if len(values) >= 11:
            x1, z1 = float(values[1]), float(values[2])
            x2, z2 = float(values[3]), float(values[4])
            x3, z3 = float(values[5]), float(values[6])
            x4, z4 = float(values[7]), float(values[8])
            resistivity = float(values[9])
            error = float(values[10])

            electrode_positions.extend([(x1, z1), (x2, z2), (x3, z3), (x4, z4)])

            resistivity_data.append(
                {
                    "coordinates": [(x1, z1), (x2, z2), (x3, z3), (x4, z4)],
                    "resistivity": resistivity,
                    "error": error,
                }
            )

    return resistivity_data


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
