import numpy as np
import warnings
from simpeg.electromagnetics.static.utils import generate_survey_from_abmn_locations
from simpeg import data
import matplotlib.pyplot as plt
from discretize import TreeMesh
from discretize.utils import active_from_xyz
from simpeg.utils import model_builder
from simpeg import maps
from matplotlib.colors import LogNorm
import matplotlib as mpl
from simpeg.electromagnetics.static.utils.static_utils import (
    plot_pseudosection,
    apparent_resistivity_from_voltage,
)
from scipy.interpolate import interp1d


def haversine(lat1, lon1, lat2, lon2):
    """Compute Haversine distance between two lat/lon points."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * 6371000 * np.arcsin(np.sqrt(a))


def create_topo(file_path, topo_from_dat, n_points, flat=False):
    """
    Reads in the topography from the Google earth exported file, ignores lat and long.
    """
    x_vals = topo_from_dat[:, 0]
    x_min = np.min(x_vals)
    x_max = np.max(x_vals)
    topo_x = np.linspace(x_min, x_max, n_points)

    if flat is False:
        # Load data from txt file (assuming tab or space delimiter, skipping header)
        data = np.loadtxt(file_path, skiprows=1, usecols=(1, 2, 3))

        # Extract latitude, longitude, and altitude
        lat, lon, alt = data[:, 0], data[:, 1], data[:, 2]

        # Compute cumulative distance along the track (approximate using Haversine formula)
        distances = np.zeros(len(lat))
        for i in range(1, len(lat)):
            distances[i] = distances[i - 1] + haversine(
                lat[i - 1], lon[i - 1], lat[i], lon[i]
            )

        # Interpolate elevation profile at n_points equally spaced distances
        interp_func = interp1d(distances, alt, kind="cubic")  # Smooth interpolation
        topo_z = interp_func(topo_x)

    else:
        topo_z = np.zeros_like(topo_x)

    topo_2d = np.c_((topo_x, topo_z))

    return topo_2d


def plot_3_pseudosections(voltage_data):

    # Plot voltages pseudo-section
    fig = plt.figure(figsize=(8, 2.75))
    ax1 = fig.add_axes([0.1, 0.15, 0.75, 0.78])
    plot_pseudosection(
        voltage_data,
        plot_type="scatter",
        ax=ax1,
        scale="log",
        cbar_label="V/A",
        scatter_opts={"cmap": mpl.cm.viridis},
    )
    ax1.set_title("Normalized Voltages")

    # Plot errors pseudo-section
    fig = plt.figure(figsize=(8, 2.75))
    ax1 = fig.add_axes([0.1, 0.15, 0.75, 0.78])
    plot_pseudosection(
        voltage_data.survey,
        voltage_data.standard_deviation,
        plot_type="scatter",
        ax=ax1,
        scale="log",
        cbar_label="V/A",
        scatter_opts={"cmap": mpl.cm.viridis},
    )
    ax1.set_title("Errors")

    # Get apparent resistivities from volts and survey geometry
    apparent_resistivities = apparent_resistivity_from_voltage(
        voltage_data.survey, voltage_data.dobs
    )

    # Plot apparent resistivity pseudo-section
    fig = plt.figure(figsize=(8, 2.75))
    ax1 = fig.add_axes([0.1, 0.15, 0.75, 0.78])
    plot_pseudosection(
        voltage_data.survey,
        apparent_resistivities,
        plot_type="contourf",
        ax=ax1,
        scale="log",
        cbar_label="$\Omega m$",
        mask_topography=True,
        contourf_opts={"levels": 20, "cmap": mpl.cm.RdYlBu},
    )
    ax1.set_title("Apparent Resistivity")


def make_permafrost_model(mesh, topo_2d, plot=True):
    # Indices of the active mesh cells from topography (e.g. cells below surface)
    active_cells = active_from_xyz(mesh, topo_2d)

    # define the model
    air_cond = 1e-8
    active_layer_cond = 1 / 50
    perma_layer_cond = 1 / 600
    bedrock_cond = 1 / 10000

    conductivity_model = model_builder.create_layers_model(
        mesh.cell_centers,
        layer_tops=np.array([0, -1, -18]),
        layer_values=np.array([active_layer_cond, perma_layer_cond, bedrock_cond]),
    )

    # Conductivity map. Model parameters are conductivities for all active cells.
    conductivity_map = maps.InjectActiveCells(mesh, active_cells, air_cond)

    # Generate a mapping to ignore inactice cells in plot
    plotting_map = maps.InjectActiveCells(mesh, active_cells, np.nan)

    if plot:
        fig = plt.figure(figsize=(9, 4))

        norm = LogNorm(vmin=1e-3, vmax=1e-1)

        ax1 = fig.add_axes([0.14, 0.17, 0.68, 0.7])
        mesh.plot_image(
            plotting_map * conductivity_model,
            ax=ax1,
            grid=False,
            pcolor_opts={"norm": norm, "cmap": mpl.cm.RdYlBu_r},
        )
        ax1.set_xlim(-500, 500)
        ax1.set_ylim(-50, 10)
        ax1.set_title("Conductivity Model")
        ax1.set_xlabel("x (m)")
        ax1.set_ylabel("z (m)")

        ax2 = fig.add_axes([0.84, 0.17, 0.03, 0.7])
        cbar = mpl.colorbar.ColorbarBase(
            ax2, norm=norm, orientation="vertical", cmap=mpl.cm.RdYlBu_r
        )
        cbar.set_label(r"$\sigma$ (S/m)", rotation=270, labelpad=15, size=12)
    return active_cells, conductivity_map, conductivity_model


def generate_dcdata_from_res2dinv_with_topo(
    file_path,
    n_points=401,
):
    """
    Extracts electrode positions, resistivity and error data from a RES2DINV .dat file,
    and generates a survey using SimPEG's generate_survey_from_abmn_locations function.

    This function reads a resistivity survey file in RES2DINV format and extracts:
    - Electrode coordinates (X, Z) for each measurement
    - Apparent resistivity values (Ωm)
    - Measurement error percentages

    Chargeability data (if present) is ignored.

    Parameters:
    ----------
    file_path : str
        Path to the .dat file containing the resistivity survey data.

    Returns
    -------
    dc_data : SimPEG `data.Data` object containing the survey and resistivity values.
    topo_profile : 2d array of x and z of topography spread out 4x as wide as the electrode points

    Example:
    -------
    >>> data = generate_dcdata_from_res2dinv("survey.dat")
    >>> print(data[0])
    {'electrodes': [(10.0, 0.0), (40.0, 0.0), (20.0, 0.0), (30.0, 0.0)],
     'resistivity': 82.437,
     'error': 12.183}
    """
    with open(file_path, "r") as file:
        lines = file.readlines()

    num_data_points = int(lines[6].strip())  # Number of data points

    # Read the data section
    A = np.zeros((num_data_points, 2))
    B = np.zeros((num_data_points, 2))
    M = np.zeros((num_data_points, 2))
    N = np.zeros((num_data_points, 2))
    dobs = np.zeros(num_data_points)
    error = np.zeros(num_data_points)

    data_start_line = 12
    for i, line in enumerate(
        lines[data_start_line : data_start_line + num_data_points]
    ):
        values = line.strip().split()
        if len(values) >= 11:
            A[i] = [float(values[1]), float(values[2])]
            B[i] = [float(values[3]), float(values[4])]
            M[i] = [float(values[5]), float(values[6])]
            N[i] = [float(values[7]), float(values[8])]
            dobs[i] = float(values[9])  # Apparent resistivity
            error[i] = float(values[10])  # Error percentage

    # Store all arrays in a list for easy iteration
    arrays = [A, B, M, N]

    # Compute the midpoint of all x-coordinates
    all_x_coords = np.concatenate([arr[:, 0] for arr in arrays])
    midpoint_electrodes = np.mean(
        [np.min(all_x_coords), np.max(all_x_coords)]
    )  # Equivalent to (min + max) / 2

    # Adjust x-coordinates
    for arr in arrays:
        arr[:, 0] -= midpoint_electrodes

    # Combine all electrodes into a single array and remove duplicates
    electrodes = np.unique(np.vstack(arrays), axis=0)

    # Generate the survey
    survey = generate_survey_from_abmn_locations(
        locations_a=A,
        locations_b=B,
        locations_m=M,
        locations_n=N,
        data_type="volt",
    )

    std = 1e-7 + 0.05 * np.abs(dobs)  # 5% constant error
    dc_data = data.Data(survey, dobs=dobs, standard_deviation=std)

    topo_profile = np.zeros((n_points, 2))
    topo_profile[:, 0] = np.linspace(
        -midpoint_electrodes * 4, midpoint_electrodes * 4, n_points
    )

    # Plot electrode positions
    plot_electrode_positions(topo_profile, electrodes)

    return dc_data, topo_profile


def build_tree_mesh(voltage_data, topo_2d, plot=True):
    """
    Constructs a TreeMesh for DC resistivity or IP inversion.

    This function generates an adaptive octree mesh based on electrode locations
    and topography data. The mesh is refined around electrodes and the surface
    to ensure accurate modeling.

    Parameters
    ----------
    voltage_data : SimPEG.data.Data
        SimPEG data object containing the survey with electrode locations.

    topo_2d : ndarray, shape (N, 2)
        Array of topography points [(x1, z1), (x2, z2), ...], where z represents elevation.

    Returns
    -------
    TreeMesh
        A refined SimPEG TreeMesh for forward modeling and inversion.

    Notes
    -----
    - The base cell size is set to 1/10 of the smallest electrode spacing.
    - The mesh extends ~8x the survey width in x and ~6x in depth.
    - Refinement is applied near electrodes and along the topography.
    """
    # Compute electrode spacing parameters
    electrode_x = voltage_data.survey.unique_electrode_locations[:, 0]
    max_spacing = np.max(electrode_x) - np.min(electrode_x)
    min_spacing = np.min(np.diff(np.sort(electrode_x)))

    cell_size = min_spacing / 2  # base cell width
    dom_width_x = max_spacing * 8  # domain width x
    dom_width_z = max_spacing * 8  # domain width z

    # Compute number of base cells (rounded to power of 2)
    nbcx = 2 ** int(np.round(np.log(dom_width_x / cell_size) / np.log(2.0)))
    nbcz = 2 ** int(np.round(np.log(dom_width_z / cell_size) / np.log(2.0)))

    # Define the base mesh with top at z = 0 m.
    hx = [(cell_size, nbcx)]
    hz = [(cell_size, nbcz)]
    mesh = TreeMesh([hx, hz], x0="CN", diagonal_balance=True)

    # Shift top to maximum topography
    mesh.origin = mesh.origin + np.r_[0.0, topo_2d[:, 1].max()]

    # Mesh refinement based on topography
    mesh.refine_surface(
        topo_2d,
        padding_cells_by_level=[0, 0, 6, 8],
        finalize=False,
    )

    # Extract unique electrode locations.
    unique_locations = voltage_data.survey.unique_electrode_locations

    # Mesh refinement near electrodes.
    mesh.refine_points(
        unique_locations, padding_cells_by_level=[10, 15, 8, 8], finalize=False
    )

    mesh.finalize()

    if plot:
        # plot mesh
        fig = plt.figure(figsize=(10, 4))
        ax1 = fig.add_axes([0.14, 0.17, 0.8, 0.7])
        mesh.plot_grid(ax=ax1, linewidth=1)
        ax1.grid(False)
        ax1.set_xlim(-1500, 1500)
        ax1.set_ylim(np.max(topo_2d[:, 1]) - 1000, np.max(topo_2d[:, 1]))
        ax1.set_title("Mesh")
        ax1.set_xlabel("x (m)")
        ax1.set_ylabel("z (m)")
        ax1.plot(topo_2d[:, 0], topo_2d[:, 1], "k-")

    return mesh


def build_new_tree_mesh(voltage_data, topo_2d, plot=True):
    dh = 5  # base cell width
    dom_width_x = 3200.0  # domain width x
    dom_width_z = 2400.0  # domain width z
    nbcx = 2 ** int(
        np.round(np.log(dom_width_x / dh) / np.log(2.0))
    )  # num. base cells x
    nbcz = 2 ** int(
        np.round(np.log(dom_width_z / dh) / np.log(2.0))
    )  # num. base cells z

    # Define the base mesh with top at z = 0 m.
    hx = [(dh, nbcx)]
    hz = [(dh, nbcz)]
    mesh = TreeMesh([hx, hz], x0="CN", diagonal_balance=True)

    # Shift top to maximum topography
    mesh.origin = mesh.origin + np.r_[0.0, topo_2d[:, 1].max()]

    # Mesh refinement based on topography
    mesh.refine_surface(
        topo_2d,
        padding_cells_by_level=[0, 0, 4, 4],
        finalize=False,
    )

    # Extract unique electrode locations.
    unique_locations = voltage_data.survey.unique_electrode_locations

    # Mesh refinement near electrodes.
    mesh.refine_points(
        unique_locations, padding_cells_by_level=[8, 12, 6, 6], finalize=False
    )

    mesh.finalize()

    if plot:
        # plot mesh
        fig = plt.figure(figsize=(10, 4))
        ax1 = fig.add_axes([0.14, 0.17, 0.8, 0.7])
        mesh.plot_grid(ax=ax1, linewidth=1)
        ax1.grid(False)
        ax1.set_xlim(-1500, 1500)
        ax1.set_ylim(np.max(topo_2d[:, 1]) - 1000, np.max(topo_2d[:, 1]))
        ax1.set_title("Mesh")
        ax1.set_xlabel("x (m)")
        ax1.set_ylabel("z (m)")
        ax1.plot(topo_2d[:, 0], topo_2d[:, 1], "k-")

    return mesh


def plot_electrode_positions(topo, electrodes):
    """
    Plots all electrode positions as black dots along with the survey topography as a line.

    Parameters
    ----------
    topo : np.ndarray, shape (N, 2)
        Array of unique survey topography points [(x1, z1), (x2, z2), ...].
    electrodes : np.ndarray, shape (M, 2)
        Array of all electrode positions [(x1, z1), (x2, z2), ...].

    Returns
    -------
    None
    """
    plt.figure(figsize=(8, 5))

    # Plot topography line
    if topo is not None:
        sorted_topo = topo[np.argsort(topo[:, 0])]  # Sort by x-coordinates
        plt.plot(
            sorted_topo[:, 0],
            sorted_topo[:, 1],
            color="black",
            linestyle="-",
            linewidth=1.5,
            label="Topography",
        )

    # Plot electrodes as black dots
    plt.scatter(
        electrodes[:, 0],
        electrodes[:, 1],
        color="red",
        marker="o",
        s=20,
        label="Electrodes",
    )

    # Labels and formatting
    plt.xlabel("X Position (m)")
    plt.ylabel("Z Position (m)")
    plt.title("Electrode Positions with Topography")
    plt.legend()
    plt.grid(True)
    plt.gca().invert_yaxis()  # Invert Z-axis to show depth correctly

    # plt.show()


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
