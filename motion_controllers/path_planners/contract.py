from abc import ABCMeta, abstractmethod


class PathPlannerContract(metaclass=ABCMeta):
    @abstractmethod
    def plan_path(self, current_state):
        pass
