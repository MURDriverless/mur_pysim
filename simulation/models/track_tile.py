import math
from Box2D import *
from pyglet import gl
from simulation.parameters import TRACK_WIDTH, TRACK_TILE_COLOUR
from simulation.constants import TRACK_TILE_TYPE


class TrackTile:
    def __init__(self, world, current_node, previous_node):
        # # Obtain required data
        position = (current_node[2], current_node[3])   # Don't pass into bodyDef(), otherwise contact won't trigger
        vertices = self._get_vertices(current_node, previous_node)

        # 1. Define body
        bodyDef = b2BodyDef()

        # 2. Instantiate body in world
        instance = world.CreateStaticBody(defn=bodyDef)

        # 3. Create shape
        shape = b2PolygonShape(vertices=vertices)

        # 4. Define fixture
        fixtureDef = b2FixtureDef(shape=shape, isSensor=True)

        # 5. Instantiate fixture in body
        instance.CreateFixture(fixtureDef)

        # 6. Finally, point userData of "instance" to the current instance of TrackTile
        instance.userData = self

        self.type = TRACK_TILE_TYPE
        self.position = position
        self.b2Data = instance
        self.colour = TRACK_TILE_COLOUR
        self.tile_visited = False
        self.tile_friction = 1.0

    def render(self):
        vertices = self.b2Data.fixtures[0].shape.vertices

        # Colour the area
        gl.glColor4f(self.colour[0], self.colour[1], self.colour[2], 1)

        # Mark boundary of the tile using the vertices in self.poly
        for vertex in vertices:
            gl.glVertex3f(vertex[0], vertex[1], 0)

    @staticmethod
    def _get_vertices(current_node, previous_node):
        # Unpack values
        alpha_current, beta_current, x_current, y_current = current_node
        alpha_previous, beta_previous, x_previous, y_previous = previous_node

        # Compute vertices
        top_left = (x_current - TRACK_WIDTH * math.cos(beta_current),
                    y_current - TRACK_WIDTH * math.sin(beta_current))
        top_right = (x_current + TRACK_WIDTH * math.cos(beta_current),
                     y_current + TRACK_WIDTH * math.sin(beta_current))
        bottom_right = (x_previous + TRACK_WIDTH * math.cos(beta_previous),
                        y_previous + TRACK_WIDTH * math.sin(beta_previous))
        bottom_left = (x_previous - TRACK_WIDTH * math.cos(beta_previous),
                       y_previous - TRACK_WIDTH * math.sin(beta_previous))

        return top_left, top_right, bottom_right, bottom_left

