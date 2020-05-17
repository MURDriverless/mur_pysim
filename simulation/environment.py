import numpy as np

import gym
from gym import spaces
from gym.utils import seeding, EzPickle
from gym.envs.classic_control import rendering

from Box2D import b2World

import pyglet
from pyglet import gl

from simulation.parameters import FPS, PLAYFIELD, STATE_W, STATE_H, WINDOW_W, WINDOW_H
from simulation.contact_listener import ContactListener
from simulation.track_coordinates_builder import TrackCoordinatesBuilder
from simulation.models.ground import Ground
from simulation.models.track_tile import TrackTile
from simulation.models.cone import Cone
from simulation.utils.rendering import follower_view_transform, get_viewport_size, render_indicators
from simulation.utils.state import State
from simulation.dynamics.car_dynamics import Car


class Environment(gym.Env, EzPickle):
    def __init__(self, verbose=False):
        EzPickle.__init__(self)

        # General and utils variables
        self.verbose = verbose
        self.np_random = None
        self.seed()

        # Box2D variables
        self.time = -1.0  # Set time to -1.0 to indicate that models is not ready yet
        self.car = None
        self.contact_listener = ContactListener(self)
        self.world = b2World((0, 0), contactListener=self.contact_listener)
        self.ground = None
        self.track_tiles_coordinates = None    # For easy access in StateTransformer
        self.track_tiles = []
        self.cones = []
        self.tile_visited_count = 0

        # PyGLet variables
        self.viewer = None
        self.invisible_state_window = None
        self.invisible_video_window = None
        self.score_label = None
        self.transform = None

        # RL-related variables
        # action_space has the following structure (steer, gas, brake). -1, +1 is for left and right steering
        self.state = None
        self.done = False
        self.action_space = spaces.Box(np.array([-1, 0, 0]), np.array([+1, +1, +1]), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=255, shape=(STATE_H, STATE_W, 3), dtype=np.uint8)
        self.reward = 0.0
        self.prev_reward = 0.0

    def step(self, action):
        car = self.car
        world = self.world

        # Apply action
        if action is not None:
            car.steer(-action[0])
            car.gas(action[1])
            car.brake(action[2])

        car.step(1.0 / FPS)
        world.Step(1.0 / FPS, 6 * 30, 2 * 30)
        # Update elapsed time
        self.time += 1.0 / FPS
        # Since we are assuming car to have infinite fuel, always set fuel_spent to 0
        # self.reward -=  10 * self.car.fuel_spent / ENGINE_POWER
        car.fuel_spent = 0.0

        # Calculate step reward
        step_reward = 0
        # Penalty for stopping and wasting time
        self.reward -= 0.1
        self.prev_reward = self.reward
        # Compute step reward and update previous reward
        step_reward += self.reward - self.prev_reward  # Current recorded reward minus previous reward

        # Check if done
        if self.tile_visited_count == len(self.track_tiles):
            self.done = True

        # Penalise further and terminate if car is out of bounds
        x, y = car.hull.position

        if abs(x) > PLAYFIELD or abs(y) > PLAYFIELD:
            step_reward -= 100

        state = self.state.update(self)

        return state

    def reset(self):
        self._destroy()
        self.time = -1.0
        self.tile_visited_count = 0
        self.state = None
        self.done = False
        self.reward = 0.0
        self.prev_reward = 0.0
        
        # Build ground
        self.ground = Ground(self.world, PLAYFIELD, PLAYFIELD)

        # Build track tiles
        self.track_tiles_coordinates = TrackCoordinatesBuilder.load_track(self)
        self.track_tiles = [TrackTile(self.world, self.track_tiles_coordinates[i], self.track_tiles_coordinates[i - 1])
                            for i, element in enumerate(self.track_tiles_coordinates)]
        # Build cones
        cones_coordinates = []
        for i in range(0, len(self.track_tiles)):
            sensor_vertices = self.track_tiles[i].b2Data.fixtures[0].shape.vertices
            for j in range(0, len(sensor_vertices)):
                cones_coordinates.append(sensor_vertices[j])
        self.cones = [Cone(world=self.world, position=(cone_coordinate[0], cone_coordinate[1]))
                      for cone_coordinate in cones_coordinates]

        init_angle = 0
        init_x, init_y = self.track_tiles[0].position

        self.car = Car(self.world, init_angle=init_angle, init_x=init_x, init_y=init_y)
        self.state = State(self)

        return np.array(tuple(self.state.info.values()))

    def render(self, mode='human'):
        assert mode in ['human', 'state_pixels', 'rgb_array']

        # Instantiate viewer
        if self.viewer is None:
            self.viewer = rendering.Viewer(WINDOW_W, WINDOW_H)
            self.score_label = pyglet.text.Label('0000', font_size=36,
                                                 x=20,
                                                 y=WINDOW_H * 2.5 / 40.00,
                                                 anchor_x='left',
                                                 anchor_y='center',
                                                 color=(255, 255, 255, 255))
            self.transform = rendering.Transform()

        # reset() not called yet, so no need to render
        if self.time == -1.0:
            return

        self.car.draw(self.viewer, mode != "state_pixels")
        self.transform = follower_view_transform(self.car, self.time)

        # Setup window
        window = self.viewer.window
        window.switch_to()
        window.dispatch_events()
        window.clear()
        VP_W, VP_H = get_viewport_size(mode, window)

        # Start drawing
        gl.glViewport(0, 0, VP_W, VP_H)
        # Transform view to follow the car and render the contents of the world
        self.transform.enable()
        self.render_world()
        # Render onetime geometries
        for geom in self.viewer.onetime_geoms:
            geom.render()
        # And empty the geometries afterwards
        self.viewer.onetime_geoms = []
        # Since the world has been rendered, and indicators below are not part of the world, disable transform
        self.transform.disable()
        render_indicators(WINDOW_W, WINDOW_H, car=self.car, reward=self.reward, score_label=self.score_label)

        if mode == 'human':
            window.flip()
            return self.viewer.isopen
        else:
            image_data = pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()
            arr = np.fromstring(image_data.get_data(), dtype=np.uint8, sep='')
            arr = arr.reshape(VP_H, VP_W, 4)
            arr = arr[::-1, :, 0:3]
            return arr

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _destroy(self):
        if not self.track_tiles:
            return
        self.world.DestroyBody(self.ground.b2Data)
        for track_tile in self.track_tiles:
            self.world.DestroyBody(track_tile.b2Data)
        self.track_tiles = []
        self.car.destroy()

    def render_world(self):
        gl.glBegin(gl.GL_QUADS)

        self.ground.render()

        for tile in self.track_tiles:
            tile.render()

        for cone in self.cones:
            cone.render()

        gl.glEnd()
