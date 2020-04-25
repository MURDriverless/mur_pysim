import math
from world.parameters import TRACK_WIDTH, ROAD_SENSOR_COLOUR
from world.models.road_sensor import RoadSensor


class RoadSensorFactory:
    @staticmethod
    def create(env, index, current_node, previous_node):
        # Unpack values
        alpha_current, beta_current, x_current, y_current = current_node
        alpha_previous, beta_previous, x_previous, y_previous = previous_node

        # Compute variables
        node = (x_current, y_current)
        top_left = (x_current - TRACK_WIDTH * math.cos(beta_current),
                    y_current - TRACK_WIDTH * math.sin(beta_current))
        top_right = (x_current + TRACK_WIDTH * math.cos(beta_current),
                     y_current + TRACK_WIDTH * math.sin(beta_current))
        bottom_left = (x_previous - TRACK_WIDTH * math.cos(beta_previous),
                       y_previous - TRACK_WIDTH * math.sin(beta_previous))
        bottom_right = (x_previous + TRACK_WIDTH * math.cos(beta_previous),
                        y_previous + TRACK_WIDTH * math.sin(beta_previous))

        vertices = (top_left, top_right, bottom_right, bottom_left)

        # A bit unclear about this. Swig and Box2D operation
        fixtures = env.fd_tile
        fixtures.shape.vertices = vertices

        # Data
        data = env.world.CreateStaticBody(fixtures=fixtures)
        data.userData = data
        data.color = ROAD_SENSOR_COLOUR
        data.road_visited = False
        data.road_friction = 1.0
        data.fixtures[0].sensor = True

        return RoadSensor(node=node, vertices=vertices, data=data, colour=ROAD_SENSOR_COLOUR)