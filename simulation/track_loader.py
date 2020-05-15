from pathlib import Path
import csv
import numpy as np


def load_track(file_name):
    """
    Get the position of left and right cones of a track, given the file name in the "tracks" directory

    Args:
        file_name (str): File name in the "tracks" directory. Please only provide the file name.
            For instance, if the name of the track is "fsg_alex_cones.txt", do not provide
            "../tracks/fsg_alex_cones.txt". Instead, just use "fsg_alex_cones.txt"

    Returns:
        np.ndarray: [left_cone_positions, right_cone_positions]
    """
    left_cone_positions = []
    right_cone_positions = []

    try:
        file_path = Path("../tracks").joinpath(file_name)

        with open(file_path, 'r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            # Discard first line as it is the description
            next(csv_reader)
            # Append subsequent lines to data
            for row in csv_reader:
                left_cone_positions.append(np.array([float(row[0]), float(row[1])]))
                right_cone_positions.append(np.array([float(row[2]), float(row[3])]))

        return left_cone_positions, right_cone_positions

    except FileNotFoundError:
        print(f"File {file_name} not found")


if __name__ == "__main__":
    left_cones, right_cones = load_track("fsg_alex_cones.txt")
    print(f"Left cones count: {len(left_cones)}")
    print(f"Right cones count: {len(right_cones)}")
