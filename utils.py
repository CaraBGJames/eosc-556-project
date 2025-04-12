import numpy as np
from simpeg.utils import model_builder
from discretize import TreeMesh
from simpeg.electromagnetics.static.utils.static_utils import (
    generate_survey_from_abmn_locations,
    apparent_resistivity_from_voltage,
)
from discretize.utils import active_from_xyz
from simpeg import maps, data
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

mpl.rcParams.update({"font.size": 14})  # default font size


def plot_results(
    mesh,
    plotting_map,
    topo_2d,
    model_to_plot,
    title="Resistivity model",
    cmap=mpl.cm.jet,
):
    """
    Plot a 2D resistivity model on a mesh.

    Parameters:
        mesh (TreeMesh): The discretized mesh used for modeling.
        plotting_map (Map): A map to inject inactive cells as NaN for plotting.
        topo_2d (ndarray): 2D array of topography points.
        model_to_plot (ndarray): Resistivity model to be visualized.
        title (str): Title of the plot.
        cmap (Colormap): Matplotlib colormap to use.

    Returns:
        None
    """
    # Plot results
    fig = plt.figure(figsize=(9, 4))

    norm = LogNorm(vmin=1e2, vmax=1e5)

    ax1 = fig.add_axes([0.14, 0.17, 0.68, 0.7])
    mesh.plot_image(
        plotting_map * model_to_plot,
        ax=ax1,
        grid=False,
        pcolor_opts={"norm": norm, "cmap": cmap},
    )
    ax1.set_xlim(np.min(topo_2d[:, 0]), np.max(topo_2d[:, 0]))
    ax1.set_ylim(np.max(topo_2d[:, 1]) - 200, np.max(topo_2d[:, 1] + 5))
    ax1.set_title(title)
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("z (m)")

    ax2 = fig.add_axes([0.84, 0.17, 0.03, 0.7])
    cbar = mpl.colorbar.ColorbarBase(ax2, norm=norm, orientation="vertical", cmap=cmap)
    cbar.set_label(
        r"resistivity ($\Omega \cdot m$)", rotation=270, labelpad=15, size=12
    )

    plt.show()


def build_starting_model(survey, dpred, active_cells):
    """
    Create a starting resistivity model from an average of the predicted voltage data apparent resistivities.

    Parameters:
        survey (Survey): DC resistivity survey object.
        dpred (ndarray): Predicted voltages.
        active_cells (ndarray): Boolean array indicating active cells.

    Returns:
        ndarray: Homogeneous starting resistivity model for active cells.
    """
    apparent_resistivities = apparent_resistivity_from_voltage(survey, dpred)
    average_resistivity = np.mean(apparent_resistivities)
    nC = int(active_cells.sum())
    starting_resistivity_model = average_resistivity * np.ones(nC)
    return starting_resistivity_model


def make_data(survey, clean_data, std):
    """
    Generate observed data with added Gaussian noise.

    Parameters:
        survey (Survey): DC resistivity survey object.
        clean_data (ndarray): Clean synthetic voltage data.
        std (float or ndarray): Standard deviation for noise.

    Returns:
        Data: SimPEG data object containing noisy observations.
    """
    # Add 5% Gaussian noise to each datum
    rng = np.random.default_rng(seed=225)
    std = 0.05 * np.abs(clean_data)
    dc_noise = rng.normal(scale=std, size=len(clean_data))
    dobs = clean_data + dc_noise

    data_dc = data.Data(survey, dobs=dobs, standard_deviation=std)

    return data_dc


def build_model(mesh, active_cells):
    """
    Build a synthetic resistivity model with ice lens and wedge (or edit accordingly from within utils.py file)

    Parameters:
        mesh (TreeMesh): The discretized mesh used for modeling.
        active_cells (ndarray): Boolean array indicating active cells.

    Returns:
        tuple: (resistivity_model, resistivity_map, plotting_map)
            - resistivity_model (ndarray): 1D array of resistivity values.
            - resistivity_map (Map): Map to inject air resistivity into inactive cells.
            - plotting_map (Map): Map to inject NaN into inactive cells for plotting.
    """
    n_active = np.sum(active_cells)

    # Make model
    air_resistivity = 1e8
    permafrost_resistivity = 600
    bedrock_resistivity = 10000
    ice_resistivity = 1000
    unfrozen_resistivity = 400

    # Define conductivity model
    resistivity_model = bedrock_resistivity * np.ones(n_active)

    ind_ice = model_builder.get_indices_block(
        np.r_[80, -60], np.r_[250, -30], mesh.cell_centers[active_cells, :]
    )
    resistivity_model[ind_ice] = ice_resistivity

    ind_ice_wedge = model_builder.get_indices_block(
        np.r_[350, -80], np.r_[360, -10], mesh.cell_centers[active_cells, :]
    )
    resistivity_model[ind_ice_wedge] = ice_resistivity

    resistivity_map = maps.InjectActiveCells(mesh, active_cells, air_resistivity)

    # Generate a mapping to ignore inactice cells in plot
    plotting_map = maps.InjectActiveCells(mesh, active_cells, np.nan)

    return resistivity_model, resistivity_map, plotting_map


def build_mesh(topo_2d):
    """
    Generate a 2D TreeMesh and identify active cells based on topography.

    Parameters:
        topo_2d (ndarray): 2D array of topography points.

    Returns:
        tuple: (mesh, active_cells)
            - mesh (TreeMesh): The created tree mesh.
            - active_cells (ndarray): Boolean array indicating active cells.
    """
    # Build the tree mesh
    dh = 5  # minimum cell width
    dom_width_x = np.max(topo_2d[:, 0]) - np.min(topo_2d[:, 0])  # domain width x
    dom_width_z = np.max(topo_2d[:, 1]) - (
        np.min(topo_2d[:, 1] - 250)
    )  # domain width z
    dz = 5
    nbcx = 2 ** (
        int(np.round(np.log(dom_width_x / dh) / np.log(2.0))) + 1
    )  # num. base cells x
    nbcz = 2 ** (
        int(np.ceil(np.log(dom_width_z / dz) / np.log(2.0)))
    )  # num. base cells z

    # Define the base mesh with top at z = 0 m.
    hx = [(dh, nbcx)]
    hz = [(dz, nbcz)]
    mesh = TreeMesh([hx, hz], x0="0N", diagonal_balance=True)

    # Shift top to maximum topography
    mesh.origin = mesh.origin + np.r_[-200, np.max(topo_2d[:, 1])]

    # Mesh refinement based on topography
    mesh.refine_surface(
        topo_2d,
        padding_cells_by_level=[10, 10, 5, 5],
        finalize=False,
    )

    mesh.finalize()

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
