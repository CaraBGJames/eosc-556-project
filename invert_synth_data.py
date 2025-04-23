import numpy as np
from utils import (
    # build_mesh,
    build_tensor_mesh_padded,
    build_model,
    # build_big_model, #for last part of sythetic testing
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
from simpeg.electromagnetics.static.utils.static_utils import plot_pseudosection
import matplotlib.pyplot as plt

######################
# SET INVERSION PARAMETERS HERE
std = 0.05  # expected noise in the data

alpha_s = 1
alpha_x = 1
alpha_y = 1

beta0_ratio = 1
coolingFactor = 2
coolingRate = 1
chifact = 1
#######################


# Read in the electrode locations from the res2dinv data file
file_path = "data/Project4_Wenner_1.dat"
survey, topo_2d = res2dinv_to_survey(file_path)

# build mesh
mesh, active_cells = build_tensor_mesh_padded(
    topo_2d,
    dx=2.5,
    dz=2.5,
    width=470,
    depth=100,
    pad_dist_x=125,
    pad_dist_z=125,
    pad_growth=1.3,
)

# number of active cells
n_active = np.sum(active_cells)

# build model (change from within utils)
log_resistivity_model, log_resistivity_map, plotting_map = build_model(
    mesh, active_cells
)

# set colormap and bounds
cmap = mpl.cm.jet_r
vmin = 1
vmax = 10 ** (4)

# plot synthetic model
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
###################################

# ensure electrodes on top
survey.drape_electrodes_on_topography(mesh, active_cells, option="top")

# DC simulation for a resistivity model
simulation = dc.simulation_2d.Simulation2DNodal(
    mesh, survey=survey, rhoMap=log_resistivity_map, storeJ=True
)

# Forward simulated data
dpred = simulation.dpred(log_resistivity_model)

# add noise to simulate real situation and make it into a data object for the inversion
data_dc = make_data(survey, dpred, std=std)  # relative error sets percentage

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
    alpha_s=alpha_s,
    alpha_x=alpha_x,
    alpha_y=alpha_y,
    reference_model=starting_log_resistivity_model,  # no reference model?...
    reference_model_in_smooth=True,  # no reference model?...
)

# optimization
opt = optimization.InexactGaussNewton(
    maxIter=40,
)

# Define inverse problem
inv_prob = inverse_problem.BaseInvProblem(dmis, reg, opt)

# Inversion directives

# # Apply and update sensitivity weighting as the model updates
update_sensitivity_weighting = directives.UpdateSensitivityWeights()

# Defining a starting value for the trade-off parameter (beta) between the data misfit and the regularization.
starting_beta = directives.BetaEstimate_ByEig(beta0_ratio=beta0_ratio)

# Set the rate of reduction in trade-off parameter (beta) each time the inverse problem is solved. And set the number of Gauss-Newton iterations for each trade-off paramter value.
beta_schedule = directives.BetaSchedule(
    coolingFactor=coolingFactor, coolingRate=coolingRate
)

# Options for outputting recovered models and predicted data for each beta.
save_iteration = directives.SaveOutputEveryIteration(save_txt=True)

# Options for outputting recovered models and predicted data for each beta.
save_model = directives.SaveModelEveryIteration()

# Setting a stopping criteria for the inversion.
target_misfit = directives.TargetMisfit(chifact=chifact)

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


###################################
# Plotting

# look at the iterations (load from within directory)
models = load_all_models(".", "*InversionModel-2025-04-22*.npy")

models = load_all_models(".", "*InversionModel-2025-04-22*.npy")
for i, model in enumerate(models):
    if i == len(models) - 1:
        title = f"Final inversion (n = {i+1})"
    else:
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

# replot synthetic model so you can compare them next to one another
plot_results(
    mesh,
    plotting_map,
    topo_2d,
    np.exp(log_resistivity_model),
    title="Synthetic model",
    cmap=cmap,
    vmin=vmin,
    vmax=vmax,
)

# plot tikhonov curves
beta, phid, phim = plot_tikhonov_curves_from_file(
    "InversionModel*.txt", target_misfit=data_dc.nD * chifact
)

# calculate data residuals for plot, set up colorbar
dpred_final = simulation.dpred(recovered_log_resistivity_model)
residual = 100 * abs((dpred_final - data_dc.dobs)) / data_dc.dobs
clim = [0, 20]
ticks = np.linspace(clim[0], clim[1], 5)  # for example: 0, 4, 8, 12, 16, 20

# Plot residuals pseudosection
fig = plt.figure(figsize=(8, 2.75))
ax1 = fig.add_axes([0.1, 0.15, 0.75, 0.78])
plot_pseudosection(
    survey,
    residual,
    plot_type="contourf",
    ax=ax1,
    clim=[0, 20],
    # scale = 'linear',
    # cbar_label="%",
    # mask_topography=True,
    create_colorbar=False,
    contourf_opts={"levels": 20, "cmap": mpl.cm.RdYlBu},
)

ax2 = fig.add_axes([0.9, 0.17, 0.03, 0.7])
norm = mpl.colors.Normalize(vmin=clim[0], vmax=clim[1])
cbar = mpl.colorbar.ColorbarBase(
    ax2, orientation="vertical", cmap=mpl.cm.RdYlBu, norm=norm, ticks=ticks
)
cbar.set_label("%", rotation=270, labelpad=15, size=12)

ax1.set_title("Percentage residual")

# show all the plots at once
plt.show()
