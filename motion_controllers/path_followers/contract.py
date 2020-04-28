from abc import ABCMeta, abstractmethod


class PathFollowerContract(metaclass=ABCMeta):
    @abstractmethod
    def follow_path(self, optimal_state, current_state):
        pass
