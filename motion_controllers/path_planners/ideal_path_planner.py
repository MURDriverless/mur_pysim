from motion_controllers.path_planners.contract import PathPlannerContract


class IdealPathPlanner(PathPlannerContract):
    def __init__(self, env):
        self.env = env

    def plan_path(self, current_state):
        # Assuming that the centre of the path is the optimal path,
        # we return just the centre coordinates of all the track tiles
        return self.env.track_tiles_coordinates
