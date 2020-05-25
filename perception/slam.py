import math
from random import randint
from perception.localisation import find_nearest_cone


class SLAM:
    """
    SLAM extracts the raw environment state and gives state which conforms to the Perception team structure
    """
    def __init__(self, noise=False, min_searchable_size=1, max_searchable_size=10):
        """
        Args:
            noise (bool, optional): If True, the returned state in step() will be augmented with noise.
        """
        self.noise = noise

        # Left and right indexes refer to the current cone index. That is, the current position of the car
        # in terms of the index within the cones list.
        self.left_index = 0
        self.right_index = 0

        # Searchable size refers to how many cones SLAM can see at a time. We can vary this to emulate real-life noise.
        self.min_searchable_size = min_searchable_size
        self.max_searchable_size = max_searchable_size

    def update(self, x, y, linear_velocity, yaw, angular_velocity, left_cone_positions, right_cone_positions):
        """
        Outputs observed state conforming to the structure outlined by the Perception team

        Args:
            x (float): horizontal position of the car in stationary frame
            y (float): vertical position of the car in stationary frame
            linear_velocity (tuple of float): a 1D vector of length 2, denoting the linear velocity in x and y direction
            yaw (float): angular deviation of the car from the horizontal axis in stationary frame (I think ?)
            angular_velocity (float): rate of change of yaw
            left_cone_positions (np.ndarray of [x, y]): in sim, refers to the hardcoded left cone positions in course
            right_cone_positions (np.ndarray of [x, y]): in sim, refers to the hardcoded right cone positions in course

        Returns:
            tuple: (x, y, v, yaw, w, [left_cone_positions], [right_cone_positions])
                - x, y, yaw: same as argument
                - v (float): absolute linear velocity
                - w (float): angular velocity, omega, same as argument
                - left & right cone positions -> [[x0, y0], [xN, yN]], where N is the searchable size
        """
        # # Initialise variables
        # car = self.env.car
        # car_position = car.hull.position
        # left_cone_positions = self.env.left_cone_positions
        # right_cone_positions = self.env.right_cone_positions

        # Noise augmentation
        if self.noise is True:
            # If we want some noise, set the number of searchable cones within [min_search, max_search], inclusive
            # at end points (randint already includes both end points)
            searchable_size = randint(self.min_searchable_size, self.max_searchable_size)
        else:
            # Else, if we do not want noise, set number of searchable cones to be the maximum of each side
            searchable_size = self.max_searchable_size

        # Get x, y, v, yaw
        v = math.sqrt(linear_velocity[0] ** 2 + linear_velocity[1] ** 2)
        w = angular_velocity

        # Get next cones
        # Find next cone indexes and self-update
        self.left_index, _ = find_nearest_cone(x, y, left_cone_positions, self.left_index, searchable_size)
        self.right_index, _ = find_nearest_cone(x, y, right_cone_positions, self.right_index, searchable_size)
        # Once we obtain the next cone index, we can obtain the next cone positions from
        # the next cone index up to the number of searchable cones.
        observed_left_cones = left_cone_positions[self.left_index:(self.left_index + searchable_size + 1)]
        observed_right_cones = right_cone_positions[self.right_index:(self.right_index + searchable_size + 1)]

        return x, y, v, yaw, w, observed_left_cones, observed_right_cones


# Test correct output
if __name__ == "__main__":
    slam = SLAM(noise=False, min_searchable_size=1, max_searchable_size=10)

    # Exactly 10 cones
    left_cones = [[-1.7667, 1.4703], [2.7609, 1.7154], [7.2786, 1.727], [12.035, 1.777], [16.539, 1.8575],
                  [21.034, 1.9058], [25.825, 1.8237], [29.008, 1.6882], [32.854, 1.5324], [36.054, 1.406]]
    right_cones = [[-1.2054, -2.4342], [3.1781, -1.9323], [7.4856, -1.8135], [12.132, -1.684], [16.791, -1.634],
                   [21.016, -1.5777], [25.949, -1.6844], [29.048, -1.7289], [32.612, -1.837], [35.959, -2.0252]]
    position = ((left_cones[0][0] + right_cones[0][0]) / 2.0, (left_cones[0][1] + right_cones[0][1]) / 2.0)
    linear_vel = (3.0, 0.2)
    angle = 0
    angular_vel = 0.05

    output = slam.update(position[0], position[1], linear_vel, angle, angular_vel, left_cones, right_cones)
    assert output[0] == position[0]
    assert output[1] == position[1]
    assert math.fabs(output[2] - math.hypot(linear_vel[0], linear_vel[1])) <= 0.0005  # Tolerable error for linear vel
    assert output[3] == angle
    assert output[4] == angular_vel

    # Check that all the cones are identical, given exactly 10 cones to observe and 10 for max_searchable_size
    cone_mismatch = False
    for i in range(len(left_cones)):
        obs_left_cones = output[5]
        obs_right_cones = output[6]
        # Check left cone x
        if obs_left_cones[i][0] != left_cones[i][0]:
            cone_mismatch = True
        # Check left cone y
        if obs_left_cones[i][1] != left_cones[i][1]:
            cone_mismatch = True
        # Check right cone x
        if obs_right_cones[i][0] != right_cones[i][0]:
            cone_mismatch = True
        # Check right cone y
        if obs_right_cones[i][1] != right_cones[i][1]:
            cone_mismatch = True
    assert cone_mismatch is False

