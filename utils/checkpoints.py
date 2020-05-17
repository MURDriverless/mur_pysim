import numpy as np

"""
    To add:
        - orientation (angle) check towards next track point
"""


class Checkpoint:
    current_index = 0

    def __init__(self, track_coordinates: dict):
        self.track_coordinates = track_coordinates

    def check(self, car_pos: (int, int), threshold=5) -> int:
        """
            Calculates the Euclidean distance from the car's current position to the next checkpoint
            If this distance is less than the threshold, the new checkpoint will is designated.
        """
        obj_pos = self.track_coordinates.get(self.current_index)

        x = np.array([car_pos[0], obj_pos[0]])
        y = np.array([car_pos[1], obj_pos[1]])

        x_dist = np.sqrt((x[0] - x[1]) ** 2)
        y_dist = np.sqrt((y[0] - y[1]) ** 2)

        dist = x_dist + y_dist
        print(dist)

        if dist < threshold:
            self.current_index += 1

        return dist

    @classmethod
    def reset(cls):
        cls.current_index = 0
