from random import randint
from simulation.environment import Environment
from perception.localisation import find_nearest_cone


class SLAM:
    def __init__(self, env, noise=False):
        """
        SLAM extracts the raw environment state and gives state which conforms to the Perception team structure

        Args:
            env (Environment): Store reference to environment for easy access.
            noise (bool, optional): If True, the returned state in step() will be augmented with noise.
        """
        self.env = env
        self.noise = noise

        # Left and right indexes refer to the current cone index. That is, the current position of the car
        # in terms of the index within the cones list.
        self.left_index = 0
        self.right_index = 0

        # Searchable size refers to how many cones SLAM can see at a time. We can vary this to emulate real-life noise.
        self.min_searchable_size = 1
        self.max_searchable_size = 10

    def step(self):
        """
        Outputs observed state conforming to the structure outlined by the Perception team

        Returns:
            tuple: (x, y, v, yaw, [list_of_obs_left_cones], [list_of_obs_right_cones])
                - obs = observed
                - list_of_obs_cones -> [[x0, y0], [xN, yN]], where N is the searchable size
        """
        # Initialise variables
        car = self.env.car
        car_position = car.hull.position
        left_cone_positions = self.env.left_cones_positions
        right_cone_positions = self.env.right_cones_positions

        # Noise augmentation
        if self.noise is True:
            # If we want some noise, set the number of searchable cones within [min_search, max_search], inclusive
            # at end points (randint already includes both end points)
            searchable_size = randint(self.min_searchable_size, self.max_searchable_size)
        else:
            # Else, if we do not want noise, set number of searchable cones to be the maximum of each side
            searchable_size = self.max_searchable_size

        # Get x, y, v, yaw
        x = car_position[0]
        y = car_position[1]
        v = car.hull.linearVelocity
        yaw = car.hull.angle

        # Get next cones
        # Find next cone indexes and self-update
        self.left_index, _ = find_nearest_cone(car_position, left_cone_positions, self.left_index, searchable_size)
        self.right_index, _ = find_nearest_cone(car_position, right_cone_positions, self.right_index, searchable_size)
        # Once we obtain the next cone index, we can obtain the next cone positions from
        # the next cone index up to the number of searchable cones.
        obs_left_cones = left_cone_positions[self.left_index:(self.left_index + searchable_size)]
        obs_right_cones = right_cone_positions[self.right_index:(self.right_index + searchable_size)]

        return x, y, v, yaw, obs_left_cones, obs_right_cones
