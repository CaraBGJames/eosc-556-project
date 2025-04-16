from functions import generate_dcdata_from_res2dinv_with_topo, build_new_tree_mesh
from discretize.utils import active_from_xyz
import numpy as np
from simpeg.utils import model_builder
from simpeg import maps
from simpeg.electromagnetics.static.utils.static_utils import (
    pseudo_locations,
    plot_pseudosection,
    apparent_resistivity_from_voltage,
)
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from simpeg.electromagnetics.static import resistivity as dc

mpl.rcParams.update({"font.size": 14})  # default font size

# import the data read in the survey locations create topography
file_path = "data/Project4_Wenner_1.dat"
n_points = 401
voltage_data, topo_2d = generate_dcdata_from_res2dinv_with_topo(
    file_path, n_points=n_points
)

# plot pseudo-locations
pseudo_locations_xz = pseudo_locations(voltage_data.survey)
fig = plt.figure(figsize=(8, 2.75))
ax = fig.add_axes([0.1, 0.1, 0.85, 0.8])
ax.scatter(pseudo_locations_xz[:, 0], pseudo_locations_xz[:, -1], 8, "r")
ax.set_xlabel("x (m)")
ax.set_ylabel("z (m)")
ax.set_title("Pseudo-locations")
plt.show()

# make mesh
mesh = build_new_tree_mesh(voltage_data, topo_2d)

# define active cells (not air)
active_cells = active_from_xyz(mesh, topo_2d)

# number of active cells
n_active = np.sum(active_cells)

# define the TEST model
air_conductivity = 1e-8
background_conductivity = 1e-2
conductor_conductivity = 1e-1
resistor_conductivity = 1e-3

# Define conductivity model
conductivity_model = background_conductivity * np.ones(n_active)
ind_conductor = model_builder.get_indices_sphere(
    np.r_[-120.0, -40.0], 60.0, mesh.cell_centers[active_cells, :]
)
conductivity_model[ind_conductor] = conductor_conductivity
ind_resistor = model_builder.get_indices_sphere(
    np.r_[120.0, -72.0], 60.0, mesh.cell_centers[active_cells, :]
)
conductivity_model[ind_resistor] = resistor_conductivity

# Conductivity map. Model parameters are conductivities for all active cells.
conductivity_map = maps.InjectActiveCells(mesh, active_cells, air_conductivity)

# Generate a mapping to ignore inactice cells in plot
plotting_map = maps.InjectActiveCells(mesh, active_cells, np.nan)

# plot model
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
ax1.set_ylim(-300, 200)
ax1.set_title("Conductivity Model")
ax1.set_xlabel("x (m)")
ax1.set_ylabel("z (m)")

ax2 = fig.add_axes([0.84, 0.17, 0.03, 0.7])
cbar = mpl.colorbar.ColorbarBase(
    ax2, norm=norm, orientation="vertical", cmap=mpl.cm.RdYlBu_r
)
cbar.set_label(r"$\sigma$ (S/m)", rotation=270, labelpad=15, size=12)

plt.show()

# electrodes on topography
voltage_data.survey.drape_electrodes_on_topography(mesh, active_cells, option="top")

# DC simulation for a conductivity model
simulation = dc.simulation_2d.Simulation2DNodal(
    mesh, survey=voltage_data.survey, sigmaMap=conductivity_map
)

# get predicted data
dpred = simulation.dpred(conductivity_model)

# Plot voltages pseudo-section
fig = plt.figure(figsize=(8, 2.75))
ax1 = fig.add_axes([0.1, 0.15, 0.75, 0.78])
plot_pseudosection(
    voltage_data.survey,
    dobs=np.abs(dpred),
    plot_type="scatter",
    ax=ax1,
    scale="log",
    cbar_label="V/A",
    scatter_opts={"cmap": mpl.cm.viridis},
)
ax1.set_title("Normalized Voltages")
plt.show()

# Get apparent conductivities from volts and survey geometry
apparent_conductivities = 1 / apparent_resistivity_from_voltage(
    voltage_data.survey, dpred
)

# Plot apparent conductivity pseudo-section
fig = plt.figure(figsize=(8, 2.75))
ax1 = fig.add_axes([0.1, 0.15, 0.75, 0.78])
plot_pseudosection(
    voltage_data.survey,
    dobs=apparent_conductivities,
    plot_type="contourf",
    ax=ax1,
    scale="log",
    cbar_label="S/m",
    mask_topography=True,
    contourf_opts={"levels": 20, "cmap": mpl.cm.RdYlBu_r},
)
ax1.set_title("Apparent Conductivity")
plt.show()
