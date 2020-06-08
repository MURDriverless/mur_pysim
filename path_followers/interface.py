from abc import ABCMeta, abstractmethod


class PathFollowerInterface(metaclass=ABCMeta):
    @abstractmethod
    def move(self, states, reference):
        """
        Issues actuation commands to motor control to bring the current state closer to the reference states

        Args:
            states (numpy.ndarray): Current states observed by Perception.
            reference (numpy.ndarray): 2D matrix of size (2, N), where
                2 = number of position coordinates (x & y), and
                N = prediction horizon length.
                We only expect positional reference from Path Planner to make the API cleaner,
                and if needed, we can use that reference to generate our own reference
                (such as yaw).

        Returns:
            numpy.ndarray: Actuation commands to be sent to EMC (input to the motor plant)
                Has the format:(throttle, steering angle rate)
                - throttle: constrained between -1 and 1. 1 signifies maximum acceleration,
                  while -1 indicates maximum deceleration.
                - steering angle rate: unsure of the constraints for now, will update later.

        Notes:
            1. We are using numpy.ndarray as the standard format for array instead of Python's
            in-built list. This is because the size of ndarray is static after creation, so it
            is supposedly faster at read and write processes. Python's list on the other hand,
            allows for dynamic growing, so it is less favoured
        """
        pass
