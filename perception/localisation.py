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
    # Only search within the position_index up to searchable_size
    searchable_cones = cyclic_fetch_elements_in_array(cone_positions, current_cone_index, searchable_size)

    # Calculate difference in x and y to compute distance from one cone to next
    dx = [car_position[0] - cone[0] for cone in searchable_cones]
    dy = [car_position[1] - cone[1] for cone in searchable_cones]

    # Build a list of distances and find the minimum squared distance
    squared_distances = [idx ** 2 + idy ** 2 for (idx, idy) in zip(dx, dy)]
    min_squared_dist = min(squared_distances)

    # Since we have built a list of distances, we can just get the next cone index by searching for
    # the index of the minimum squared distance value.
    unadjusted_index = squared_distances.index(min_squared_dist) + current_cone_index
    # To account for overflown elements in our searchable_cones, we adjust the index by using
    # the modulo operator "%" over the total number of cones
    next_cone_index = unadjusted_index % len(cone_positions)
    next_cone_distance = math.sqrt(min_squared_dist)
    return next_cone_index, next_cone_distance


def cyclic_fetch_elements_in_array(array, start_index, searchable_size):
    """
    Fetches elements without worrying about reaching the end of the array

    Args:
        array (array_like): anything in the form of array, can be an array of ADT (Abstract Data Type)
        start_index (int): the starting index to slice from
        searchable_size (int): the number of elements included from start_index

    Returns:
        list of object
    """
    array_length = len(array)
    # Determine if we start_index + searchable_size will exceed the array length (overflow),
    # and if yes, we want to calculate how many elements will exceed, which we call overflow_n.
    overflow_n = start_index + searchable_size - array_length
    # If it is larger than 0, that means we have an overflow
    if overflow_n > 0:
        # We need to return 2 concatenated arrays:
        # 1. Elements from the current index to the maximum length of the array
        # 2. Elements from the start to the overflow_n
        return array[start_index:array_length] + array[0:overflow_n]
    else:
        # Otherwise, if we do not have overflow, return elements from
        # the starting index up to the searchable size as usual
        return array[start_index:(start_index + searchable_size)]
