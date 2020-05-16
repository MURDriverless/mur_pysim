import math
import numpy as np

import gym
from gym import spaces
from gym.envs.box2d.car_dynamics import Car
from gym.utils import seeding, EzPickle
from gym.envs.classic_control import rendering

from Box2D import b2World

import pyglet
from pyglet import gl

from perception.slam import SLAM
from simulation.track_loader import load_track
from simulation.parameters import FPS, PLAYFIELD, WINDOW_W, WINDOW_H
from simulation.contact_listener import ContactListener
from simulation.models.ground import Ground
from simulation.models.track_tile import TrackTile
from simulation.models.cone import Cone
from simulation.utils.rendering import follower_view_transform, get_viewport_size, render_indicators


class Environment(gym.Env, EzPickle):
    def __init__(self, verbose=False):
        EzPickle.__init__(self)

        # General and utils variables
        self.verbose = verbose
        self.np_random = None
        self.seed()

        # Simulation entities
        self.contact_listener = ContactListener(env=self)
        self.slam = SLAM(env=self, noise=False)

        # Track data
        # Load left and right cone positions
        self.left_cone_positions, self.right_cone_positions = load_track("fsg_alex_cones.txt")

        # Track objects
        self.world = b2World((0, 0), contactListener=self.contact_listener)
        self.ground = None
        self.left_cones = []
        self.right_cones = []
        self.track_tiles = []
        self.car = None

        # Simulation data
        # Note that we do not have self.state and self.done .
        # The rationale is that `state` and `done` do not actually need to be stored in memory. Hence,
        # we can just return these data from function calls without committing to memory.
        self.time = -1.0  # Set time to -1.0 to indicate that the models are not ready yet
        self.reward = 0.0
        self.previous_reward = 0.0
        self.tile_visited_count = 0

        # PyGLet variables
        self.viewer = None
        self.invisible_state_window = None
        self.invisible_video_window = None
        self.score_label = None
        self.transform = None

        # observation_space is left out as we are not using pixels for our state
        self.action_space = spaces.Box(np.array([-1, -1]), np.array([+1, +1]), dtype=np.float32)
        self.observation_space = None

    def step(self, action):
        """
        Performs one simulation step using action and returns the state of simulation

        Args:
            action (list): [throttle, steering]
                - throttle (float): [-1, +1] -> -1 for max. braking, and +1 for max. acceleration
                - steering (float): [-1, +1] -> -1 for left steering, and +1 for right steering

        Returns:
            tuple: (state, step_reward, done)
                - state (tuple): state observed by SLAM in the following format
                    (x, y, v, yaw, [list_of_left_cones_ahead], [list_of_right_cones_ahead]), where the number
                    of cones listed ahead varies. The list of cones contains positions in the format [x_cone, y_cone]
                - step_reward (float): reward for the action taken only during this step (can be negative for penalty)
                - done (bool): whether the simulation has completed
        """
        # Track previous reward before it gets updated
        self.previous_reward = self.reward

        car = self.car
        world = self.world

        # Apply action if it is not empty
        if action is not None and len(action) > 0:
            throttle = action[0]
            steering = action[1]
            # Constrain inputs
            if throttle >= 1.0:
                throttle = 1.0
            elif throttle <= -1.0:
                throttle = -1.0
            if steering >= 1.0:
                steering = 1.0
            elif steering <= -1.0:
                steering = -1.0
            car.gas(throttle if throttle > 0 else 0)
            car.brake(throttle if throttle < 0 else 0)
            car.steer(-steering)

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
        # Compute step reward and update previous reward
        step_reward += self.reward - self.previous_reward  # Current recorded reward minus previous reward

        # Determine if done
        done = False

        # Check if done
        if self.tile_visited_count == len(self.track_tiles):
            done = True

        # Penalise further and terminate if car is out of bounds
        x, y = car.hull.position
        if abs(x) > PLAYFIELD or abs(y) > PLAYFIELD:
            done = True
            step_reward -= 100

        state = self.slam.step()

        return state, step_reward, done, {}

    def reset(self):
        self._destroy_objects()
        self.time = -1.0  # Set time to -1.0 to indicate that the models are not ready yet
        self.reward = 0.0
        self.previous_reward = 0.0
        self.tile_visited_count = 0
        
        # Build ground
        self.ground = Ground(self.world, PLAYFIELD, PLAYFIELD)

        # Build left and right cones
        self.left_cones = [Cone(world=self.world, position=(cone[0], cone[1])) for cone in self.left_cone_positions]
        self.right_cones = [Cone(world=self.world, position=(cone[0], cone[1])) for cone in self.right_cone_positions]

        # Build track tiles
        # Assume that the number of left and right cones are equal
        # Build track tiles from 0 to N-1, where N is the total number of cones.
        self.track_tiles = [TrackTile(self.world, self.left_cone_positions[i], self.right_cone_positions[i],
                                      self.left_cone_positions[i+1], self.right_cone_positions[i+1])
                            for i in range(len(self.left_cone_positions)-1)]

        # For the track fsg_alex_cones.txt, we need to turn the car by -90 degrees to align it in the correct direction
        init_angle = math.radians(-90)
        init_x, init_y = self.track_tiles[-1].position
        self.car = Car(self.world, init_angle, init_x, init_y)
        return self.step([])

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

    def _destroy_objects(self):
        # Destroy track objects
        if self.ground:
            self.world.DestroyBody(self.ground.b2Data)
        if self.track_tiles:
            [self.world.DestroyBody(tile.b2Data) for tile in self.track_tiles]
        if self.left_cones:
            [self.world.DestroyBody(cone.b2Data) for cone in self.left_cones]
        if self.right_cones:
            [self.world.DestroyBody(cone.b2Data) for cone in self.right_cones]
        if self.car:
            self.car.destroy()
        # Re-initialise the track objects
        self.ground = None
        self.track_tiles = []
        self.left_cones = []
        self.right_cones = []
        self.car = None

    def render_world(self):
        gl.glBegin(gl.GL_QUADS)
        self.ground.render()
        [tile.render() for tile in self.track_tiles]
        [cone.render() for cone in self.left_cones]
        [cone.render() for cone in self.right_cones]
        gl.glEnd()
