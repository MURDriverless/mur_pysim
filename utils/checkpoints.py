import numpy as np

"""
    To add:
        - orientation (angle) check towards next track point
"""


class Checkpoint:
    index = 0

    def __init__(self, track_coordinates: dict):
        self.track_coordinates = track_coordinates

    def check_dist(self, car_pos: (int, int), threshold=5) -> int:
        """
            Calculates the Euclidean distance from the car's current position to the next checkpoint
            If this distance is less than the threshold, the new checkpoint will is designated.
        """
        obj_pos = self.track_coordinates.get(self.index)

        x = np.array([car_pos[0], obj_pos[0]])
        y = np.array([car_pos[1], obj_pos[1]])

        x_dist = np.sqrt((x[0] - x[1]) ** 2)
        y_dist = np.sqrt((y[0] - y[1]) ** 2)

        dist = x_dist + y_dist

        if dist < threshold:
            self.index += 1

        return dist

    def check_xytheta_dist(self, x_pos, y_pos):
        """
            Calculates the difference between the x_pos and y_pos of the car and the x_pos and
            y_pos of the objective. Additionally, calculates the angle between obj and car.

        :return: x_diff (np.float32), y_diff (np.float32), theta (np.float32)
        """
        obj_pos = self.track_coordinates.get(self.index)

        x_delta = np.float32(x_pos - obj_pos[0])
        y_delta = np.float32(y_pos - obj_pos[1])
        theta = np.arctan(x_diff / y_diff)

        return x_delta, y_delta, theta


    @classmethod
    def reset(cls):
        cls.index = 0

