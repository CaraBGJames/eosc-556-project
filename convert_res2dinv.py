import numpy as np
from functions import extract_resistivity_from_res2dinv

# from simpeg.electromagnetics.static.utils import generate_dcip_sources_line

# extract data from file
file_path = "data/Wenner_1-2024-07-30-142945.dat"
resistivity_data = extract_resistivity_from_res2dinv(file_path)

# define survey
# data_array = np.loadtxt(data_filename, skiprows=1)

# dobs = data_array[:, -1]
# A = data_array[:, 0:2]
# B = data_array[:, 2:4]
# M = data_array[:, 4:6]
# N = data_array[:, 6:8]

# survey = generate_survey_from_abmn_locations(
#     locations_a=A, locations_b=B, locations_m=M, locations_n=M, data_type="volt"
# )

# dc_data = data.Data(survey, dobs=dobs)
