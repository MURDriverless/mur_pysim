import numpy as np

import gym
from gym import spaces
from gym.envs.box2d.car_dynamics import Car
from gym.utils import seeding, EzPickle

import Box2D
from Box2D.b2 import (fixtureDef, polygonShape)

from world.parameters import STATE_W, STATE_H, FPS
from world.physics_engine import PhysicsEngine
from world.renderer import Renderer
from world.rewarder import Rewarder
from world.track_builder import TrackBuilder
from world.perception import Perception
from world.track_position_observer import TrackPositionObserver
from world.factories.road_sensor_factory import RoadSensorFactory


class Environment(gym.Env, EzPickle):
    perception = None
    physics_engine = None
    renderer = None
    rewarder = None

    def __init__(self, verbose=False):
        EzPickle.__init__(self)

        # General and utils variables
        self.verbose = verbose
        self.np_random = None
        self.seed()

        # Helper entities
        self.perception = Perception(self)
        self.physics_engine = PhysicsEngine(self)
        self.renderer = Renderer(self)

        # Simulation world variables
        self.time = -1.0  # Set time to -1.0 to indicate that world is not ready yet
        self.fd_tile = fixtureDef(shape=polygonShape(vertices=[(0, 0), (1, 0), (1, -1), (0, -1)]))
        self.contact_listener = TrackPositionObserver(self)
        self.car = None
        self.world = Box2D.b2World((0, 0), contactListener=self.contact_listener)
        self.road_sensors = []
        # self.cones = []
        self.tile_visited_count = 0

        # RL-related variables
        # action_space has the following structure (steer, gas, brake). -1, +1 is for left and right steering
        self.state = None
        self.action_space = spaces.Box(np.array([-1, 0, 0]), np.array([+1, +1, +1]), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=255, shape=(STATE_H, STATE_W, 3), dtype=np.uint8)
        self.reward = 0.0
        self.prev_reward = 0.0

        # Rendering
        self.viewer = None
        self.invisible_state_window = None
        self.invisible_video_window = None
        self.score_label = None
        self.transform = None

    def step(self, action):
        self.time += 1.0/FPS
        if action is not None:
            self.physics_engine.step(action)

        self.state = self.render("state_pixels")

        step_reward = Rewarder.reward(self)

        # Check if done
        done = False
        if self.tile_visited_count == len(self.road_sensors):
            done = True

        return self.state, step_reward, done, {}

    def reset(self):
        self._destroy()
        self.reward = 0.0
        self.prev_reward = 0.0
        # self.tile_visited_count = 0
        self.time = -1.0

        # Load road sensor coordinates
        road_sensor_coordinates = TrackBuilder.load_track(self)
        self.road_sensors = [RoadSensorFactory.create(self, i,
                                                      road_sensor_coordinates[i],
                                                      road_sensor_coordinates[i - 1])
                             for i, element in enumerate(road_sensor_coordinates)]
        # cones_coordinates = []
        # for i in range(0, len(road_sensor_coordinates)):
        #     sensor_vertices = road_sensor_coordinates[i].vertices
        #     for j in range(0, len(sensor_vertices)):
        #         cones_coordinates.append(sensor_vertices[j])
        # self.cones = []

        init_node = self.road_sensors[0].node
        init_angle = 0
        init_x = init_node[0]
        init_y = init_node[1]

        self.car = Car(self.world, init_angle=init_angle, init_x=init_x, init_y=init_y)

        return self.step(None)[0]

    def render(self, mode='human'):
        return self.renderer.render(mode)

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _destroy(self):
        if not self.road_sensors:
            return
        for sensor in self.road_sensors:
            self.world.DestroyBody(sensor)
        self.road_sensors = []
        self.car.destroy()
