from abc import ABCMeta, abstractmethod


class PathFollowerContract(metaclass=ABCMeta):
    @abstractmethod
    def move(self, states, reference=None):
        """
        Issues actuation commands to motor control.

        Args:
            states (list of float): Current states observed by Perception.
            reference (list of float, optional): An array of of the same size as states.
                Its purpose is to correct the current state using the reference states,
                so `reference` needs to share the same structure as `states`. This is
                typically passed on from Path Planner. If there is a None entry, the
                reference for that state will be the same as the state, such that there
                is supposedly 0 error.

        Returns:
            list of float: Actuation commands to be sent to EMC. Has the following format:
                (throttle, steering angle rate)
                - throttle: constrained between -1 and 1. 1 signifies maximum acceleration,
                  while -1 indicates maximum deceleration.
                - steering angle rate: unsure of the constraints for now, will update later.

        Notes:
            `states` will contain an array of cone positions, and in this particular case,
            the `reference` should be set to None, and developers are encouraged to explain
            why they are only accessing the first n elements of the reference, because
            the last one is related to the cones.
        """
        pass
