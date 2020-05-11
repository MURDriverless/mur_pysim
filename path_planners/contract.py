from abc import ABCMeta, abstractmethod


class PathPlannerContract(metaclass=ABCMeta):
    @abstractmethod
    def plan(self, states):
        """
        Generates reference signals (not limited to positions)

        Args:
            states (numpy.ndarray): Current states observed by Perception.

        Returns:
            numpy.ndarray: An array of the same size as states. Path Planners
                are intended to generate reference values for the Path Followers.
                Note that a Path Planner is not required to provide reference
                states for all of the predictive model states. For instance,
                an RRT Planner may only provide reference for position, but not
                velocity. In that case, set the velocity reference entry "None".

        Notes:
            1. For justification on using numpy.ndarray, see PathFollowerContract
        """
        pass
