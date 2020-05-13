from Box2D import *
from pyglet import gl
from simulation.parameters import GROUND_COLOUR
from simulation.constants import GROUND_TYPE


class Ground:
    def __init__(self, world, length, width):
        # Obtain required data
        position = (0, 0)

        # 1. Define body
        bodyDef = b2BodyDef(position=position, active=False)

        # 2. Create instance of body in world
        instance = world.CreateStaticBody(defn=bodyDef)

        # 3. Create shape
        shape = b2PolygonShape(box=(length, width))

        # 4. Define fixture
        fixtureDef = b2FixtureDef(shape=shape)

        # 5. Instantiate fixture in body
        instance.CreateFixture(fixtureDef)

        # 6. Finally, point userData of "instance" to the current instance of TrackTile
        instance.userData = self

        self.type = GROUND_TYPE
        self.b2Data = instance
        self.colour = GROUND_COLOUR

    def render(self):
        vertices = self.b2Data.fixtures[0].shape.vertices

        # Colour the area
        gl.glColor4f(self.colour[0], self.colour[1], self.colour[2], 1)

        for vertex in vertices:
            gl.glVertex3f(vertex[0], vertex[1], 0)
