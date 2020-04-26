import math
import numpy as np
from gym.envs.classic_control import rendering
from pyglet import gl
from simulation.parameters import SCALE, STATE_W, STATE_H, VIDEO_W, VIDEO_H, WINDOW_W, WINDOW_H, ZOOM


def follower_view_transform(car, time):
    # rendering.Transform() is connected to the global instance of Pyglet.gl,
    # so "transform" will update whatever Pyglet instance currently displayed
    transform = rendering.Transform()

    # Calculate transform-update values
    zoom = 0.1 * SCALE * max(1 - time, 0) + ZOOM * SCALE * min(time, 1)  # Animate zoom first second
    # zoom_state = ZOOM * SCALE * STATE_W / WINDOW_W
    # zoom_video = ZOOM * SCALE * VIDEO_W / WINDOW_W
    scroll_x = car.hull.position[0]
    scroll_y = car.hull.position[1]
    angle = -car.hull.angle
    velocity = car.hull.linearVelocity
    if np.linalg.norm(velocity) > 0.5:
        angle = math.atan2(velocity[0], velocity[1])

    # Update transform
    transform.set_scale(zoom, zoom)
    transform.set_translation(
        WINDOW_W / 2 - (scroll_x * zoom * math.cos(angle) - scroll_y * zoom * math.sin(angle)),
        WINDOW_H / 4 - (scroll_x * zoom * math.sin(angle) + scroll_y * zoom * math.cos(angle)))
    transform.set_rotation(angle)

    return transform


def get_viewport_size(mode, window):
    if mode == 'rgb_array':
        VP_W = VIDEO_W
        VP_H = VIDEO_H
    elif mode == 'state_pixels':
        VP_W = STATE_W
        VP_H = STATE_H
    else:
        pixel_scale = 1
        if hasattr(window.context, '_nscontext'):
            pixel_scale = window.context._nscontext.view().backingScaleFactor()  # pylint: disable=protected-access
        VP_W = int(pixel_scale * WINDOW_W)
        VP_H = int(pixel_scale * WINDOW_H)

    return VP_W, VP_H


def render_indicators(W, H, car, reward, score_label):
    gl.glBegin(gl.GL_QUADS)
    s = W / 40.0
    h = H / 40.0
    gl.glColor4f(0, 0, 0, 1)
    gl.glVertex3f(W, 0, 0)
    gl.glVertex3f(W, 5 * h, 0)
    gl.glVertex3f(0, 5 * h, 0)
    gl.glVertex3f(0, 0, 0)

    def vertical_ind(place, val, color):
        gl.glColor4f(color[0], color[1], color[2], 1)
        gl.glVertex3f((place + 0) * s, h + h * val, 0)
        gl.glVertex3f((place + 1) * s, h + h * val, 0)
        gl.glVertex3f((place + 1) * s, h, 0)
        gl.glVertex3f((place + 0) * s, h, 0)

    def horiz_ind(place, val, color):
        gl.glColor4f(color[0], color[1], color[2], 1)
        gl.glVertex3f((place + 0) * s, 4 * h, 0)
        gl.glVertex3f((place + val) * s, 4 * h, 0)
        gl.glVertex3f((place + val) * s, 2 * h, 0)
        gl.glVertex3f((place + 0) * s, 2 * h, 0)

    true_speed = np.sqrt(np.square(car.hull.linearVelocity[0]) + np.square(car.hull.linearVelocity[1]))
    vertical_ind(5, 0.02 * true_speed, (1, 1, 1))
    vertical_ind(7, 0.01 * car.wheels[0].omega, (0.0, 0, 1))  # ABS sensors
    vertical_ind(8, 0.01 * car.wheels[1].omega, (0.0, 0, 1))
    vertical_ind(9, 0.01 * car.wheels[2].omega, (0.2, 0, 1))
    vertical_ind(10, 0.01 * car.wheels[3].omega, (0.2, 0, 1))
    horiz_ind(20, -10.0 * car.wheels[0].joint.angle, (0, 1, 0))
    horiz_ind(30, -0.8 * car.hull.angularVelocity, (1, 0, 0))
    gl.glEnd()
    score_label.text = "%04i" % reward
    score_label.draw()
