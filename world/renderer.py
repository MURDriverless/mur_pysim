import math

import numpy as np
import pyglet
from pyglet import gl

from world.parameters import STATE_W, STATE_H, VIDEO_W, VIDEO_H, WINDOW_W, WINDOW_H, SCALE, PLAYFIELD, ZOOM


class Renderer:
    def __init__(self, env):
        self.env = env

    def render(self, mode='human'):
        assert mode in ['human', 'state_pixels', 'rgb_array']

        env = self.env
        car = env.car

        if env.viewer is None:
            from gym.envs.classic_control import rendering
            env.viewer = rendering.Viewer(WINDOW_W, WINDOW_H)
            env.score_label = pyglet.text.Label('0000', font_size=36,
                                                x=20,
                                                y=WINDOW_H * 2.5 / 40.00,
                                                anchor_x='left',
                                                anchor_y='center',
                                                color=(255, 255, 255, 255))
            env.transform = rendering.Transform()

        # reset() not called yet
        if env.time == -1.0:
            return

        zoom = 0.1 * SCALE * max(1 - env.time, 0) + ZOOM * SCALE * min(env.time, 1)  # Animate zoom first second
        zoom_state = ZOOM * SCALE * STATE_W / WINDOW_W
        zoom_video = ZOOM * SCALE * VIDEO_W / WINDOW_W
        scroll_x = car.hull.position[0]
        scroll_y = car.hull.position[1]
        angle = -car.hull.angle
        vel = car.hull.linearVelocity
        if np.linalg.norm(vel) > 0.5:
            angle = math.atan2(vel[0], vel[1])
        env.transform.set_scale(zoom, zoom)
        env.transform.set_translation(
            WINDOW_W / 2 - (scroll_x * zoom * math.cos(angle) - scroll_y * zoom * math.sin(angle)),
            WINDOW_H / 4 - (scroll_x * zoom * math.sin(angle) + scroll_y * zoom * math.cos(angle)))
        env.transform.set_rotation(angle)

        car.draw(env.viewer, mode != "state_pixels")

        arr = None
        win = env.viewer.window
        win.switch_to()
        win.dispatch_events()

        win.clear()
        transform = env.transform
        if mode == 'rgb_array':
            VP_W = VIDEO_W
            VP_H = VIDEO_H
        elif mode == 'state_pixels':
            VP_W = STATE_W
            VP_H = STATE_H
        else:
            pixel_scale = 1
            if hasattr(win.context, '_nscontext'):
                pixel_scale = win.context._nscontext.view().backingScaleFactor()  # pylint: disable=protected-access
            VP_W = int(pixel_scale * WINDOW_W)
            VP_H = int(pixel_scale * WINDOW_H)

        gl.glViewport(0, 0, VP_W, VP_H)
        transform.enable()
        self.render_world()
        for geom in env.viewer.onetime_geoms:
            geom.render()
        env.viewer.onetime_geoms = []
        transform.disable()
        self.render_indicators(WINDOW_W, WINDOW_H)

        if mode == 'human':
            win.flip()
            return env.viewer.isopen

        image_data = pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()
        arr = np.fromstring(image_data.get_data(), dtype=np.uint8, sep='')
        arr = arr.reshape(VP_H, VP_W, 4)
        arr = arr[::-1, :, 0:3]

        return arr

    def render_world(self):
        gl.glBegin(gl.GL_QUADS)
        gl.glColor4f(0.4, 0.8, 0.4, 1.0)
        gl.glVertex3f(-PLAYFIELD, +PLAYFIELD, 0)
        gl.glVertex3f(+PLAYFIELD, +PLAYFIELD, 0)
        gl.glVertex3f(+PLAYFIELD, -PLAYFIELD, 0)
        gl.glVertex3f(-PLAYFIELD, -PLAYFIELD, 0)
        gl.glColor4f(0.4, 0.9, 0.4, 1.0)
        k = PLAYFIELD / 20.0
        for x in range(-20, 20, 2):
            for y in range(-20, 20, 2):
                gl.glVertex3f(k * x + k, k * y + 0, 0)
                gl.glVertex3f(k * x + 0, k * y + 0, 0)
                gl.glVertex3f(k * x + 0, k * y + k, 0)
                gl.glVertex3f(k * x + k, k * y + k, 0)

        road_sensors = self.env.road_sensors
        # cones = self.env.cones

        for i in range(0, len(road_sensors)):
            road_sensors[i].render()
        # for j in range(0, len(cones)):
        #     cones[j].render()

        gl.glEnd()

    def render_indicators(self, W, H):
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

        car = self.env.car
        reward = self.env.reward
        score_label = self.env.score_label

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
