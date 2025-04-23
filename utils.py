import numpy as np
import warnings
from simpeg.utils import model_builder
from discretize import TensorMesh
from simpeg.electromagnetics.static.utils.static_utils import (
    generate_survey_from_abmn_locations,
    apparent_resistivity_from_voltage,
    plot_pseudosection,
)
from discretize.utils import active_from_xyz
from simpeg import maps, data
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import glob
import os

mpl.rcParams.update({"font.size": 14})  # default font size


def load_all_models(directory, searchstring):
    """
    Load all .npy inversion model files from a directory into a list.

    Parameters:
    -----------
    directory : str
        Directory containing .npy inversion model files.
    searchstring : str
        Filename pattern to match (e.g., '*.npy').

    Returns:
    --------
    models : list of np.ndarray
        List of 1D numpy arrays, one per model file.
    """
    files = sorted(glob.glob(os.path.join(directory, searchstring)))

    if not files:
        print(f"No model files found in {directory}")
        return []

    models = [np.load(f) for f in files]
    return models


def plot_tikhonov_curves_from_file(file_pattern, target_misfit=None):
    """
    Plot Tikhonov trade-off curves (beta vs phi_d, phi_m, and phi_m vs phi_d) from a data file.

    Parameters:
    -----------
    file_pattern : str
        Glob pattern to find the target data file (e.g., 'tikhonov_output_*.txt').
    target_misfit : float, optional
        If provided, draws a horizontal line at this misfit value.

    Returns:
    --------
    beta : np.ndarray
        Regularization parameter values.
    phi_d : np.ndarray
        Data misfit values.
    phi_m : np.ndarray
        Model roughness (regularization) values.
    """
    # Read the data from the file
    similar_files = sorted(
        glob.glob(os.path.join(".", file_pattern))
    )  # , key=lambda f: os.path.getmtime(f)) # can sort by time instead

    if not similar_files:
        print("No model data files found")
        return []

    # get last file (youngest)
    file_path = similar_files[-1]
    data = np.loadtxt(file_path, comments="#")

    # Extract columns: Assuming beta, phi_d, phi_m are in the first 3 columns
    beta = data[:, 1]  # 1st column is beta
    phi_d = data[:, 2]  # 2nd column is phi_d
    phi_m = data[:, 3]  # 3rd column is phi_m

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6, 10))

    ax1.plot(beta, phi_d, "o-k")
    ax2.plot(beta, phi_m, "o-k")
    ax3.plot(phi_m, phi_d, "o-k")

    if target_misfit is not None:
        ax1.axhline(target_misfit, color="red", linestyle="--")
        ax3.axhline(target_misfit, color="red", linestyle="--")

    ax1.set_xscale("log")
    ax1.set_xlim(ax1.get_xlim()[::-1])
    ax1.set_xlabel("β")
    ax2.set_xscale("log")
    ax2.set_xlim(ax2.get_xlim()[::-1])
    ax2.set_xlabel("β")

    ax1.set_ylabel("ϕd")
    ax2.set_ylabel("ϕm")
    ax3.set_ylabel("ϕd")
    ax3.set_xlabel("ϕm")

    ax1.set_title("Tikhonov Trade-off Curves")
    for ax in (ax1, ax2, ax3):
        ax.grid(True)

    plt.tight_layout()
    plt.show()

    return beta, phi_d, phi_m


def plot_results(
    mesh,
    plotting_map,
    topo_2d,
    model_to_plot,
    title="Resistivity model",
    cmap=mpl.cm.jet,
    vmin=1e2,
    vmax=1e5,
    xlim=None,
    ylim=None,
):
    """
    Plot a 2D resistivity model on a mesh.

    Parameters:
    -----------
    mesh : TreeMesh
        The discretized mesh used for modeling.
    plotting_map : Map
        Map to apply (e.g., to mask inactive cells).
    topo_2d : np.ndarray
        2D array of surface topography points (x, z).
    model_to_plot : np.ndarray
        Resistivity model values to visualize.
    title : str, optional
        Title for the plot.
    cmap : matplotlib Colormap, optional
        Colormap used to display resistivity values.
    vmin : float, optional
        Minimum value for color scaling (log scale).
    vmax : float, optional
        Maximum value for color scaling (log scale).
    xlim : tuple, optional
        Limits for the x-axis.
    ylim : tuple, optional
        Limits for the y-axis.

    Returns:
    --------
    None
    """
    # Plot results
    fig = plt.figure(figsize=(9, 2))

    norm = LogNorm(vmin=vmin, vmax=vmax)

    ax1 = fig.add_axes([0.14, 0.17, 0.68, 0.7])
    mesh.plot_image(
        plotting_map * model_to_plot,
        ax=ax1,
        grid=False,
        pcolor_opts={"norm": norm, "cmap": cmap},
    )

    if xlim is None:
        ax1.set_xlim(np.min(topo_2d[:, 0]), np.max(topo_2d[:, 0]))
        ax1.set_ylim(np.max(topo_2d[:, 1]) - 100, np.max(topo_2d[:, 1] + 5))
    else:
        ax1.set_xlim(xlim)
        ax1.set_ylim(ylim)

    ax1.set_title(title)
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("z (m)")

    ax2 = fig.add_axes([0.84, 0.17, 0.03, 0.7])
    cbar = mpl.colorbar.ColorbarBase(ax2, norm=norm, orientation="vertical", cmap=cmap)
    cbar.set_label(
        r"resistivity ($\Omega \cdot m$)", rotation=270, labelpad=15, size=12
    )


def build_starting_model(survey, dpred, active_cells):
    """
    Create a homogeneous starting resistivity model from the average apparent resistivity.

    Parameters:
    -----------
    survey : Survey
        DC resistivity survey object.
    dpred : np.ndarray
        Predicted voltage data from a forward model.
    active_cells : np.ndarray
        Boolean array indicating which cells are active in the inversion.

    Returns:
    --------
    starting_resistivity_model : np.ndarray
        1D array of starting resistivity values for active cells.
    """
    apparent_resistivities = apparent_resistivity_from_voltage(survey, dpred)
    average_resistivity = np.mean(apparent_resistivities)
    nC = int(active_cells.sum())
    starting_resistivity_model = average_resistivity * np.ones(nC)
    return starting_resistivity_model


def make_data(survey, clean_data, std):
    """
    Generate synthetic observed data by adding Gaussian noise to clean voltage data.

    Parameters:
    -----------
    survey : Survey
        DC resistivity survey object.
    clean_data : np.ndarray
        Clean synthetic voltage data (noiseless).
    std : float or np.ndarray
        Relative noise level (e.g., 0.05 for 5%) or array of standard deviations.

    Returns:
    --------
    data_dc : Data
        SimPEG Data object containing noisy observations and associated uncertainties.
    """
    # Add 5% Gaussian noise to each datum
    rng = np.random.default_rng(seed=225)
    error = 1e-5 + std * np.abs(clean_data)
    dc_noise = rng.normal(scale=error, size=len(clean_data))
    dobs = clean_data + dc_noise

    data_dc = data.Data(survey, dobs=dobs, standard_deviation=error)

    return data_dc


def build_big_model(mesh, active_cells):
    """
    Build a synthetic resistivity model with ice lens and wedge (or edit accordingly from within utils.py file)

    Parameters:
        mesh (TreeMesh): The discretized mesh used for modeling.
        active_cells (ndarray): Boolean array indicating active cells.

    Returns:
        tuple: (resistivity_model, resistivity_map, plotting_map)
            - log_resistivity_model (ndarray): 1D array of resistivity values.
            - log_resistivity_map (Map): Map to inject air resistivity into inactive cells.
            - plotting_map (Map): Map to inject NaN into inactive cells for plotting.
    """
    n_active = np.sum(active_cells)

    # Make model
    air_resistivity = 1e8
    active_resistivity = 10
    permafrost_resistivity = 1000
    ice_resistivity = 5000
    unfrozen_resistivity = 100

    # Define conductivity model

    # # with active layer
    # resistivity_model = model_builder.create_layers_model(
    #     mesh,
    #     layer_tops=np.array([0, -2.5, -50]),
    #     layer_values=np.array(
    #         [active_resistivity, permafrost_resistivity, ice_resistivity]
    #     ),
    # )

    # without active layer
    resistivity_model = model_builder.create_layers_model(
        mesh,
        layer_tops=np.array([0, -50]),
        layer_values=np.array([permafrost_resistivity, unfrozen_resistivity]),
    )

    ind_ice = model_builder.get_indices_block(
        np.r_[30, 00], np.r_[60, -40], mesh.cell_centers[active_cells, :]
    )
    resistivity_model[ind_ice] = ice_resistivity

    ind_ice = model_builder.get_indices_block(
        np.r_[70, -10], np.r_[200, -20], mesh.cell_centers[active_cells, :]
    )
    resistivity_model[ind_ice] = ice_resistivity

    ind_ice = model_builder.get_indices_block(
        np.r_[180, -10], np.r_[190, -20], mesh.cell_centers[active_cells, :]
    )
    resistivity_model[ind_ice] = ice_resistivity

    ind_ice = model_builder.get_indices_block(
        np.r_[350, -30], np.r_[370, -40], mesh.cell_centers[active_cells, :]
    )
    resistivity_model[ind_ice] = ice_resistivity

    ind_ice = model_builder.get_indices_block(
        np.r_[150, -30], np.r_[310, -35], mesh.cell_centers[active_cells, :]
    )
    resistivity_model[ind_ice] = ice_resistivity

    ind_ice = model_builder.get_indices_block(
        np.r_[370, -5], np.r_[400, -20], mesh.cell_centers[active_cells, :]
    )
    resistivity_model[ind_ice] = ice_resistivity

    ind_ice = model_builder.get_indices_block(
        np.r_[340, -5], np.r_[365, -20], mesh.cell_centers[active_cells, :]
    )
    resistivity_model[ind_ice] = unfrozen_resistivity

    ind_ice = model_builder.get_indices_block(
        np.r_[320, -5], np.r_[355, -10], mesh.cell_centers[active_cells, :]
    )
    resistivity_model[ind_ice] = active_resistivity

    ind_ice = model_builder.get_indices_block(
        np.r_[80, -40], np.r_[120, -50], mesh.cell_centers[active_cells, :]
    )
    resistivity_model[ind_ice] = unfrozen_resistivity

    ind_ice = model_builder.get_indices_block(
        np.r_[170, 0], np.r_[180, -20], mesh.cell_centers[active_cells, :]
    )
    resistivity_model[ind_ice] = unfrozen_resistivity

    # convert to log space
    log_resistivity_model = np.log(resistivity_model)

    log_resistivity_map = maps.InjectActiveCells(
        mesh, active_cells, np.log(air_resistivity)
    ) * maps.ExpMap(nP=n_active)

    # Generate a mapping to ignore inactice cells in plot
    plotting_map = maps.InjectActiveCells(mesh, active_cells, np.nan)

    return log_resistivity_model, log_resistivity_map, plotting_map


def build_model(mesh, active_cells):
    """
    Build a synthetic resistivity model with ice lens and wedge (or edit accordingly from within utils.py file)

    Parameters:
        mesh (TreeMesh): The discretized mesh used for modeling.
        active_cells (ndarray): Boolean array indicating active cells.

    Returns:
        tuple: (resistivity_model, resistivity_map, plotting_map)
            - log_resistivity_model (ndarray): 1D array of resistivity values.
            - log_resistivity_map (Map): Map to inject air resistivity into inactive cells.
            - plotting_map (Map): Map to inject NaN into inactive cells for plotting.
    """
    n_active = np.sum(active_cells)

    # Make model
    air_resistivity = 1e8
    active_resistivity = 10
    permafrost_resistivity = 1000
    ice_resistivity = 5000
    unfrozen_resistivity = 100

    # Define conductivity model

    # # with active layer
    # resistivity_model = model_builder.create_layers_model(
    #     mesh,
    #     layer_tops=np.array([0, -2.5, -50]),
    #     layer_values=np.array(
    #         [active_resistivity, permafrost_resistivity, ice_resistivity]
    #     ),
    # )

    # without active layer
    resistivity_model = model_builder.create_layers_model(
        mesh,
        layer_tops=np.array([0, -50]),
        layer_values=np.array([permafrost_resistivity, unfrozen_resistivity]),
    )

    # add block
    ind_ice = model_builder.get_indices_block(
        np.r_[240, 00], np.r_[260, -40], mesh.cell_centers[active_cells, :]
    )
    resistivity_model[ind_ice] = ice_resistivity

    # convert to log space
    log_resistivity_model = np.log(resistivity_model)

    log_resistivity_map = maps.InjectActiveCells(
        mesh, active_cells, np.log(air_resistivity)
    ) * maps.ExpMap(nP=n_active)

    # Generate a mapping to ignore inactice cells in plot
    plotting_map = maps.InjectActiveCells(mesh, active_cells, np.nan)

    return log_resistivity_model, log_resistivity_map, plotting_map


def build_tensor_mesh_padded(
    topo_2d,
    dx=5,
    dz=5,
    width=470,
    depth=100,
    pad_dist_x=125,
    pad_dist_z=125,
    pad_growth=1.3,
):
    """
    Create a 2D TensorMesh with roughly 5m cell sizes,
    extending 470m wide and 100m deep.

    Parameters:
    -----------
    topo_2d : (n, 2) array_like
        2D array of topography points [x, y] in the same coordinate system as the mesh.
    dx : float
        Horizontal cell size (default 5m)
    dz : float
        Vertical cell size (default 5m)
    width : float
        Total width of the mesh in meters
    depth : float
        Total depth of the mesh in meters

    Returns:
    --------
    mesh : discretize.TensorMesh
        The generated 2D TensorMesh.
    active_cells : np.ndarray
        Boolean array indicating which mesh cells are below topography.
    """
    ncx = int(width / dx)
    ncz = int(depth / dz)

    # Helper to compute padding cells
    def compute_padding_cells(cell_size, distance, factor):
        n = 0
        total = 0
        while total < distance:
            total += cell_size * (factor**n)
            n += 1
        return n

    # --- Padding cell counts ---
    npad_x = compute_padding_cells(dx, pad_dist_x, pad_growth)
    npad_z = compute_padding_cells(dz, pad_dist_z, pad_growth)

    # X direction: [left padding | core | right padding]
    hx = (
        (dx * pad_growth ** np.arange(npad_x)[::-1]).tolist()
        + [dx] * ncx
        + (dx * pad_growth ** np.arange(npad_x)).tolist()
    )

    # Y direction: [padding | core]
    hz = (dz * pad_growth ** np.arange(npad_z)[::-1]).tolist() + [dz] * ncz

    # --- Create mesh ---
    # x0: [origin_x, origin_y]
    # Since we want core to start at x=0, y=0 (top), we shift accordingly
    origin_x = -sum(hx[:npad_x])  # start of left padding
    # Place top of core at y = 0
    origin_z = -sum(hz)

    mesh = TensorMesh([hx, hz], x0=[origin_x, origin_z])

    # Indices of the active mesh cells from topography (e.g. cells below surface)
    active_cells = active_from_xyz(mesh, topo_2d)

    return mesh, active_cells


def res2dinv_to_survey(file_path):
    """
    Parse a Res2DInv file and generate a SimPEG DC survey and 2D topography.

    Parameters:
        file_path (str): Path to the Res2DInv .dat file.

    Returns:
        tuple: (survey, topo_2d)
            - survey (Survey): Generated DC resistivity survey.
            - topo_2d (ndarray): 2D array of electrode positions used as topography.
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
            dobs[i] = float(values[9])  # Voltage
            error[i] = float(values[10])  # Error percentage

    # Generate the survey
    survey = generate_survey_from_abmn_locations(
        locations_a=A, locations_b=B, locations_m=M, locations_n=N, data_type="volt"
    )

    # Topo from electrode positions (flat)
    topo_2d = np.unique(np.vstack([A, B, M, N]), axis=0)

    return survey, topo_2d


def res2dinv_to_real_data(file_path):
    """
    Parse a Res2DInv file and generate a SimPEG DC survey, 2D topography, and observed data.

    Parameters:
        file_path (str): Path to the Res2DInv .dat file.

    Returns:
        tuple: (survey, topo_2d, dobs, error)
            - survey (Survey): Generated DC resistivity survey.
            - topo_2d (ndarray): 2D array of electrode positions used as topography.
            - dobs (ndarray): Observed voltage data.
            - error (ndarray): Standard deviation or percentage error of the measurements.
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
            dobs[i] = float(values[9])  # Voltage
            error[i] = float(values[10])  # Error percentage

    # Generate the survey
    survey = generate_survey_from_abmn_locations(
        locations_a=A, locations_b=B, locations_m=M, locations_n=N, data_type="volt"
    )

    # Topo from electrode positions (flat)
    topo_2d = np.unique(np.vstack([A, B, M, N]), axis=0)

    return survey, topo_2d, dobs, error


def plot_2_pseudosections(survey, dobs):
    """
    Plot two pseudosections: one for voltages and another for apparent resistivities.

    Parameters:
    -----------
    survey : Survey
        The DC resistivity survey object containing the survey geometry.
    dobs : ndarray
        The observed voltage data (V/A) corresponding to the survey.

    Returns:
    --------
    None
        This function generates and displays the plots of the voltage pseudosection and the apparent resistivity pseudosection.
    """
    # Plot voltages pseudo-section
    fig = plt.figure(figsize=(8, 2.75))
    ax1 = fig.add_axes([0.1, 0.15, 0.75, 0.78])
    plot_pseudosection(
        survey,
        dobs,
        plot_type="scatter",
        ax=ax1,
        scale="log",
        cbar_label="V/A",
        scatter_opts={"cmap": mpl.cm.viridis},
    )
    ax1.set_title("Voltages")

    # Get apparent resistivities from volts and survey geometry
    apparent_resistivities = apparent_resistivity_from_voltage(survey, dobs)

    # Plot apparent resistivity pseudo-section
    fig = plt.figure(figsize=(8, 2.75))
    ax1 = fig.add_axes([0.1, 0.15, 0.75, 0.78])
    plot_pseudosection(
        survey,
        apparent_resistivities,
        plot_type="contourf",
        ax=ax1,
        scale="log",
        cbar_label="$\Omega m$",
        mask_topography=True,
        contourf_opts={"levels": 20, "cmap": mpl.cm.RdYlBu},
    )
    ax1.set_title("Apparent Resistivity")

    plt.tight_layout()
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
