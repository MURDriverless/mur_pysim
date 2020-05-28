from abc import ABCMeta, abstractmethod


class PathFollowerInterface(metaclass=ABCMeta):
    @abstractmethod
    def move(self, states, reference=None):
        """
        Issues actuation commands to motor control to bring the current state closer to the reference states

        Args:
            states (numpy.ndarray): Current states observed by Perception.
            reference (numpy.ndarray, optional): 2D matrix of size (NX, N), where
                NX = number of states used in the model, and
                N = prediction horizon length.
                Remember to check the PathPlanner used to see if any of reference
                is intentionally set to 0 to avoid tracking that reference. If that is the
                case, remember to ignore the state reference in PathFollower.

        Returns:
            numpy.ndarray: Actuation commands to be sent to EMC (input to the motor plant)
                Has the format:(throttle, steering angle rate)
                - throttle: constrained between -1 and 1. 1 signifies maximum acceleration,
                  while -1 indicates maximum deceleration.
                - steering angle rate: unsure of the constraints for now, will update later.

        Notes:
            1. Check perception/slam.py to see the structure of `states`.
            2. We are using numpy.ndarray as the standard format for array instead of Python's
            in-built list. This is because the size of ndarray is static after creation, so it
            is supposedly faster at read and write processes. Python's list on the other hand,
            allows for dynamic growing, so it is less favoured
        """
        pass
