from pyglet import gl


class RoadSensor:
    def __init__(self, node, vertices, data, colour):
        self.node = node
        self.vertices = vertices
        self.data = data
        self.colour = colour

    def render(self):
        top_left, top_right, bottom_left, bottom_right = self.vertices

        # Colour the area
        gl.glColor4f(self.colour[0], self.colour[1], self.colour[2], 1)

        # Mark boundary of the tile using the vertices in self.poly
        gl.glVertex3f(top_left[0], top_left[1], 0)
        gl.glVertex3f(top_right[0], top_right[1], 0)
        gl.glVertex3f(bottom_left[0], bottom_left[1], 0)
        gl.glVertex3f(bottom_right[0], bottom_right[1], 0)
