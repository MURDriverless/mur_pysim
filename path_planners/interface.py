from abc import ABCMeta, abstractmethod


class PathPlannerInterface(metaclass=ABCMeta):
    @abstractmethod
    def plan(self, states):
        """
        Generates reference states along the prediction horizon

        Args:
            states (numpy.ndarray): Current states observed by Perception.

        Returns:
            numpy.ndarray: 2D-matrix of size (NX, N), where
                NX = number of states used in model and
                N = prediction horizon length.
                If some of the states do not have reference, just set it to 0 and in PathFollower,
                we will set it manually to ignore that reference.

        Notes:
            1. For justification on using numpy.ndarray, see PathFollowerInterface
        """
        pass
