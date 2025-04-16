import numpy as np
from utils import (
    # build_mesh,
    build_tensor_mesh,
    build_model,
    res2dinv_to_survey,
    make_data,
    build_starting_model,
    plot_results,
    plot_tikhonov_curves_from_file,
    load_all_models,
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
mesh, active_cells = build_tensor_mesh(topo_2d, dx=5, dz=5, width=470, depth=100)

# number of active cells
n_active = np.sum(active_cells)

# build model (change from within utils)
log_resistivity_model, log_resistivity_map, plotting_map = build_model(
    mesh, active_cells
)

# set colormap
cmap = mpl.cm.jet_r

vmin = 30
vmax = 10 ** (4.5)

# plot
plot_results(
    mesh,
    plotting_map,
    topo_2d,
    np.exp(log_resistivity_model),  # to get resistivities
    title="Synthetic model",
    cmap=cmap,
    vmin=vmin,
    vmax=vmax,
)

# ensure electrodes on top
survey.drape_electrodes_on_topography(mesh, active_cells, option="top")

# DC simulation for a resistivity model
simulation = dc.simulation_2d.Simulation2DNodal(
    mesh, survey=survey, rhoMap=log_resistivity_map, storeJ=True
)

# Forward simulated data
dpred = simulation.dpred(log_resistivity_model)

# add noise to simulate real situation and make it into a data object for the inversion
data_dc = make_data(survey, dpred, std=0.05)

# define starting model from average of apparent resistivities
starting_log_resistivity_model = np.log(
    build_starting_model(survey, dpred, active_cells)
)

# define data misfit
dmis = data_misfit.L2DataMisfit(data=data_dc, simulation=simulation)

# least-squares regularization
reg = regularization.WeightedLeastSquares(
    mesh,
    active_cells=active_cells,
    mapping=log_resistivity_map,
    alpha_s=1e-4,
    alpha_x=1,
    alpha_y=1,
    reference_model=starting_log_resistivity_model,  # no reference model?...
    reference_model_in_smooth=True,  # no reference model?...
)

# optimization
opt = optimization.InexactGaussNewton(
    maxIter=40,
    maxIterLS=50,
    maxIterCG=20,
    tolCG=1e-3,
)

# Define inverse problem
inv_prob = inverse_problem.BaseInvProblem(dmis, reg, opt)

# Inversion directives

# Apply and update sensitivity weighting as the model updates
update_sensitivity_weighting = directives.UpdateSensitivityWeights(
    every_iteration=True, threshold_value=1e-2
)

# Defining a starting value for the trade-off parameter (beta) between the data misfit and the regularization.
starting_beta = directives.BetaEstimate_ByEig(beta0_ratio=1)

# Set the rate of reduction in trade-off parameter (beta) each time the inverse problem is solved. And set the number of Gauss-Newton iterations for each trade-off paramter value.
beta_schedule = directives.BetaSchedule(coolingFactor=2, coolingRate=2)

# Options for outputting recovered models and predicted data for each beta.
save_iteration = directives.SaveOutputEveryIteration(save_txt=True)

# Options for outputting recovered models and predicted data for each beta.
save_model = directives.SaveModelEveryIteration()

# Setting a stopping criteria for the inversion.
target_misfit = directives.TargetMisfit(chifact=1)

# Update preconditioner
update_jacobi = directives.UpdatePreconditioner()

directives_list = [
    update_sensitivity_weighting,
    starting_beta,
    beta_schedule,
    save_iteration,
    save_model,
    target_misfit,
    update_jacobi,
]

# Inversion
# Here we combine the inverse problem and the set of directives
dc_inversion = inversion.BaseInversion(inv_prob, directiveList=directives_list)

# Run inversion
recovered_log_resistivity_model = dc_inversion.run(starting_log_resistivity_model)

vmin = 30
vmax = 10 ** (4.5)

# look at the iterations
models = load_all_models(".")
for i, model in enumerate(models):
    title = f"Inversion iteration {i+1}"
    plot_results(
        mesh,
        plotting_map,
        topo_2d,
        np.exp(model),
        title=title,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

# plot

vmin = 500
vmax = 10 ** (5)

plot_results(
    mesh,
    plotting_map,
    topo_2d,
    np.exp(log_resistivity_model),
    # np.exp(recovered_resistivity_model),
    # np.exp(starting_log_resistivity_model),
    title="Synthetic model",
    cmap=cmap,
    vmin=vmin,
    vmax=vmax,
)


plot_results(
    mesh,
    plotting_map,
    topo_2d,
    # np.exp(log_resistivity_model),
    np.exp(recovered_log_resistivity_model),
    # np.exp(starting_log_resistivity_model),
    title="Recovered model",
    cmap=cmap,
    vmin=vmin,
    vmax=vmax,
)

beta, phid, phim = plot_tikhonov_curves_from_file(
    "InversionModel*.txt", target_misfit=data_dc.nD
)
