import numpy as np
from utils import (
    build_mesh,
    build_model,
    res2dinv_to_survey,
    make_data,
    build_starting_model,
    plot_results,
)
from simpeg.electromagnetics.static import resistivity as dc
from simpeg import (
    data_misfit,
    regularization,
    optimization,
    inverse_problem,
    directives,
    inversion,
)
import matplotlib as mpl

# Read in the electrode locations from the res2dinv data file
file_path = "data/Project4_Wenner_1.dat"
survey, topo_2d = res2dinv_to_survey(file_path)

# build mesh
mesh, active_cells = build_mesh(topo_2d)

# number of active cells
n_active = np.sum(active_cells)

# build model (change from within utils)
resistivity_model, resistivity_map, plotting_map = build_model(mesh, active_cells)

# ensure electrodes on top
survey.drape_electrodes_on_topography(mesh, active_cells, option="top")

# DC simulation for a resistivity model
simulation = dc.simulation_2d.Simulation2DNodal(
    mesh, survey=survey, rhoMap=resistivity_map, storeJ=True
)

# Forward simulated data
dpred = simulation.dpred(resistivity_model)

# add noise to simulate real situation and make it into a data object for the inversion
data_dc = make_data(survey, dpred, std=0.05)

# define starting model from average of apparent resistivities
starting_resistivity_model = build_starting_model(survey, dpred, active_cells)

# define data misfit
dmis = data_misfit.L2DataMisfit(data=data_dc, simulation=simulation)

# least-squares regularization
reg = regularization.WeightedLeastSquares(
    mesh,
    active_cells=active_cells,
    alpha_s=1e-4,
    # alpha_x=0.1,
    # alpha_y=0.1,
    reference_model=starting_resistivity_model,  # no reference model?...
    reference_model_in_smooth=True,  # no reference model?...
)

# optimization
opt = optimization.InexactGaussNewton(
    maxIter=40,
    maxIterLS=20,
    maxIterCG=20,
    tolCG=1e-3,
)

# Define inverse problem
inv_prob = inverse_problem.BaseInvProblem(dmis, reg, opt)

# Inversion directives

# Apply and update sensitivity weighting as the model updates
update_sensitivity_weighting = directives.UpdateSensitivityWeights()

# Defining a starting value for the trade-off parameter (beta) between the data
# misfit and the regularization.
starting_beta = directives.BetaEstimate_ByEig(beta0_ratio=1e1)

# Set the rate of reduction in trade-off parameter (beta) each time the
# the inverse problem is solved. And set the number of Gauss-Newton iterations
# for each trade-off paramter value.
beta_schedule = directives.BetaSchedule(coolingFactor=3, coolingRate=2)

# Options for outputting recovered models and predicted data for each beta.
save_iteration = directives.SaveOutputEveryIteration(save_txt=False)

# Setting a stopping criteria for the inversion.
target_misfit = directives.TargetMisfit(chifact=1)

# Update preconditioner
update_jacobi = directives.UpdatePreconditioner()

directives_list = [
    update_sensitivity_weighting,
    starting_beta,
    beta_schedule,
    save_iteration,
    target_misfit,
    update_jacobi,
]

# Inversion
# Here we combine the inverse problem and the set of directives
dc_inversion = inversion.BaseInversion(inv_prob, directiveList=directives_list)

# Run inversion
recovered_resistivity_model = dc_inversion.run(starting_resistivity_model)

# set colormap
cmap = mpl.cm.jet

# plot
plot_results(
    mesh, plotting_map, topo_2d, resistivity_model, title="Starting model", cmap=cmap
)

plot_results(
    mesh,
    plotting_map,
    topo_2d,
    recovered_resistivity_model,
    title="Recovered model",
    cmap=cmap,
)
