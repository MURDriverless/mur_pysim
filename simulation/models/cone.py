from Box2D import *
from pyglet import gl
from simulation.parameters import CONE_RADIUS
from simulation.type_names import CONE_TYPE


class Cone:
    def __init__(self, world, position, colour):
        # # Obtain required data
        vertices = self._get_vertices(position, CONE_RADIUS)

        # 1. Define body
        bodyDef = b2BodyDef()   # Don't pass position into bodyDef(), otherwise contact won't trigger

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

        self.type = CONE_TYPE
        self.position = position
        self.b2Data = instance
        self.colour = colour

    def render(self):
        vertices = self.b2Data.fixtures[0].shape.vertices

        # Colour the area
        gl.glColor4f(self.colour[0], self.colour[1], self.colour[2], 1)

        # Mark boundary of the tile using the vertices in self.poly
        for vertex in vertices:
            gl.glVertex3f(vertex[0], vertex[1], 0)

    @staticmethod
    def _get_vertices(position, radius):
        x_pos, y_pos = position

        top_left = (x_pos - radius / 2.0, y_pos + radius / 2.0)
        top_right = (x_pos + radius / 2.0, y_pos + radius / 2.0)
        bottom_right = (x_pos + radius / 2.0, y_pos - radius / 2.0)
        bottom_left = (x_pos - radius / 2.0, y_pos - radius / 2.0)

        return top_left, top_right, bottom_right, bottom_left
