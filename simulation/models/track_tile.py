from Box2D import *
from pyglet import gl
from simulation.parameters import TRACK_TILE_COLOUR
from simulation.type_names import TRACK_TILE_TYPE


class TrackTile:
    def __init__(self, world, current_left, current_right, next_left, next_right):
        vertices = (current_left, next_left, next_right, current_right)

        # Don't pass position into bodyDef(), otherwise contact won't trigger
        position = self.get_position(current_left, current_right, next_left, next_right)

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
    def get_position(current_left, current_right, next_left, next_right):
        left_middle_x = (current_left[0] + next_left[0])
        right_middle_x = (current_right[0] + next_right[0])
        left_middle_y = (current_left[1] + next_left[1])
        right_middle_y = (current_right[1] + next_right[1])
        tile_x = (left_middle_x + right_middle_x) / 2.0
        tile_y = (left_middle_y + right_middle_y) / 2.0
        return tile_x, tile_y

