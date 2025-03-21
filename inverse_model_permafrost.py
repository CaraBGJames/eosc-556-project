import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from functions import (
    generate_dcdata_from_res2dinv,
    build_tree_mesh,
    make_permafrost_model,
    plot_3_pseudosections,
)
from simpeg.electromagnetics.static import resistivity as dc
from simpeg.utils import model_builder
from simpeg import maps
from discretize.utils import active_from_xyz
from matplotlib.colors import LogNorm

mpl.rcParams.update({"font.size": 14})  # default font size
cmap = mpl.cm.RdYlBu_r  # default colormap

# extract data from file, plot survey layout
file_path = "data/Wenner_1-2024-07-30-142945.dat"
voltage_data = generate_dcdata_from_res2dinv(file_path, std_method="measured")

# for now, topo is completely flat so...
x_topo = np.linspace(-2000, 2000, 401)
z_topo = np.zeros_like(x_topo)
topo_2d = np.c_[x_topo, z_topo]

# have a pre-look at the data, errors, apparent resistivity
plot_3_pseudosections(voltage_data)

# Design a (tree) mesh
mesh = build_tree_mesh(voltage_data, topo_2d, plot=True)

# make model and plot
active_cells, conductivity_map = make_permafrost_model(mesh, topo_2d, plot=True)

voltage_data.survey.drape_electrodes_on_topography(mesh, active_cells, option="top")

# DC simulation for a conductivity model
simulation_con = dc.simulation_2d.Simulation2DNodal(
    mesh, survey=voltage_data.survey, sigmaMap=conductivity_map
)

plt.show()
