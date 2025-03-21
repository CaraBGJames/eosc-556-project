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
from simpeg import (
    maps,
    data_misfit,
    regularization,
    optimization,
    inverse_problem,
    inversion,
    directives,
)
from discretize.utils import active_from_xyz
from simpeg.electromagnetics.static.utils.static_utils import (
    plot_pseudosection,
    apparent_resistivity_from_voltage,
)

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
# plot_3_pseudosections(voltage_data)

# Design a (tree) mesh
mesh = build_tree_mesh(voltage_data, topo_2d, plot=False)

# make a starting model and plot
active_cells, conductivity_map, conductivity_model = make_permafrost_model(
    mesh, topo_2d, plot=False
)

voltage_data.survey.drape_electrodes_on_topography(mesh, active_cells, option="top")

# Map model parameters to all cells
log_conductivity_map = maps.InjectActiveCells(mesh, active_cells, 1e-8) * maps.ExpMap(
    nP=np.sum(active_cells)
)

voltage_simulation = dc.simulation_2d.Simulation2DNodal(
    mesh, survey=voltage_data.survey, sigmaMap=log_conductivity_map, storeJ=True
)

dmis_L2 = data_misfit.L2DataMisfit(simulation=voltage_simulation, data=voltage_data)

reg_L2 = regularization.WeightedLeastSquares(
    mesh,
    active_cells=active_cells,
    alpha_s=1**-2,
    alpha_x=1,
    alpha_y=1,
    reference_model=conductivity_model,
    reference_model_in_smooth=False,
)

opt_L2 = optimization.InexactGaussNewton(
    maxIter=40, maxIterLS=20, maxIterCG=20, tolCG=1e-3
)

inv_prob_L2 = inverse_problem.BaseInvProblem(dmis_L2, reg_L2, opt_L2)

sensitivity_weights = directives.UpdateSensitivityWeights(
    every_iteration=True, threshold_value=1e-2
)
update_jacobi = directives.UpdatePreconditioner(update_every_iteration=True)
starting_beta = directives.BetaEstimate_ByEig(beta0_ratio=10)
beta_schedule = directives.BetaSchedule(coolingFactor=2.0, coolingRate=2)
target_misfit = directives.TargetMisfit(chifact=1.0)

directives_list_L2 = [
    sensitivity_weights,
    update_jacobi,
    starting_beta,
    beta_schedule,
    target_misfit,
]

# Here we combine the inverse problem and the set of directives
inv_L2 = inversion.BaseInversion(inv_prob_L2, directives_list_L2)

# Run the inversion
# recovered_model_L2 = inv_L2.run(np.log(0.01) * np.ones(n_param))
recovered_log_conductivity_model = inv_L2.run(conductivity_model)
