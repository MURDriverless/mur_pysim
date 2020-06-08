from abc import ABCMeta, abstractmethod


class PathPlannerInterface(metaclass=ABCMeta):
    @abstractmethod
    def plan(self, states):
        """
        Generates reference positions along the prediction horizon

        Args:
            states (numpy.ndarray): Current states observed by Perception.

        Returns:
            numpy.ndarray: 2D-matrix of size (2, N), where
                1. The rows correspond to x and y coordinates in inertial frame
                2. N = prediction horizon length

        Notes:
            1. For justification on using numpy.ndarray, see PathFollowerInterface
        """
        pass
