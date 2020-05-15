import math
import numpy as np


def find_nearest_cone(car_position, cone_positions, current_cone_index, searchable_size):
    """
    Calculates the index of the nearest next cone from the car (next_cone_index)

    Args:
        car_position (array_like): x and y position of the car
        cone_positions (array_like): position of known cones in the format [[x0, y0], [xN, yN]]
        current_cone_index (int): the starting index of search
        searchable_size (int): the maximum number of cones after current_cone_index which are included in search

    Returns:
        tuple: next_cone_index, next_cone_distance
            - next_cone_index (int): index of the nearest next cone from the car
            - next_cone_distance (float): distance from the current car position to the next nearest cone
    """
    # Only search within the position_index up to n_index_search
    possible_cones = cone_positions[current_cone_index:(current_cone_index + searchable_size)]
    dx = np.zeros(len(possible_cones))
    dy = np.zeros(len(possible_cones))

    # Calculate difference in x and y to compute distance from one cone to next
    for i in range(len(possible_cones)):
        dx[i] = car_position[0] - possible_cones[i][0]
        dy[i] = car_position[1] - possible_cones[i][1]

    # Build a list of distances and find the minimum squared distance
    squared_distances = [idx ** 2 + idy ** 2 for (idx, idy) in zip(dx, dy)]
    min_squared_dist = min(squared_distances)

    # Since we have built a list of distances, we can just get the next cone index by searching for
    # the index of the minimum squared distance value.
    next_cone_index = squared_distances.index(min_squared_dist) + current_cone_index
    next_cone_distance = math.sqrt(min_squared_dist)
    return next_cone_index, next_cone_distance
