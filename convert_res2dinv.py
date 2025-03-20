import numpy as np
from simpeg.electromagnetics.static.utils import generate_dcip_sources_line


def read_res2dinv(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()

    # Extract header information
    project_info = lines[0].strip().split()
    project_name = project_info[0]
    array_type = project_info[1]

    electrode_spacing = float(lines[1].strip())  # Electrode spacing
    num_data_points = int(lines[2].strip())  # Number of data points

    # Skip header text and get the actual measurement type
    measurement_type = int(lines[5].strip())  # Type of measurement (0 or 1)

    # Read the data section
    electrode_positions = []
    data = []
    data_start_line = 12
    for line in lines[data_start_line : data_start_line + num_data_points]:
        values = line.strip().split()
        if len(values) >= 11:
            x1, z1 = float(values[1]), float(values[2])
            x2, z2 = float(values[3]), float(values[4])
            x3, z3 = float(values[5]), float(values[6])
            x4, z4 = float(values[7]), float(values[8])
            resistivity = float(values[9])
            error = float(values[10])
            chargeability = [float(v) for v in values[11:]]

            electrode_positions.extend([(x1, z1), (x2, z2), (x3, z3), (x4, z4)])

            data.append(
                {
                    "coordinates": [(x1, z1), (x2, z2), (x3, z3), (x4, z4)],
                    "resistivity": resistivity,
                    "error": error,
                    "chargeability": chargeability,
                }
            )

    # Remove duplicates and sort by x position
    electrode_positions = sorted(list(set(electrode_positions)))

    # Get end points from electrode positions
    x_positions = [pos[0] for pos in electrode_positions]
    end_points = [min(x_positions), max(x_positions)]

    # Create topography array (X, Z coordinates)
    topo = np.array(electrode_positions)

    # Define SimPEG parameters
    survey_type = "dipole-dipole"  # Likely for Wenner array
    data_type = (
        "apparent_resistivity" if measurement_type == 0 else "apparent_chargeability"
    )
    dimension_type = "2D"
    num_rx_per_src = len(x_positions) - 1  # Or define manually
    station_spacing = electrode_spacing

    # Generate sources for SimPEG survey
    sources = generate_dcip_sources_line(
        survey_type=survey_type,
        data_type=data_type,
        dimension_type=dimension_type,
        end_points=np.array(end_points),
        topo=topo,
        num_rx_per_src=num_rx_per_src,
        station_spacing=station_spacing,
    )

    return sources


# Example usage
file_path = "data/Wenner_1-2024-07-30-142945.dat"
sources = read_res2dinv(file_path)

print(f"Generated {len(sources)} sources for SimPEG survey.")
