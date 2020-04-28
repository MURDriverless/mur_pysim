from motion_controllers.path_followers.contract import PathFollowerContract


class BasicPathFollower(PathFollowerContract):
    def follow_path(self, optimal_state, current_state):
        # For now, just accelerate
        return 0, 0.1, 0    # (steer, gas, brake)
