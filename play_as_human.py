import numpy as np
from pyglet.window import key
from simulation.environment import Environment

restart = None


def play_as_human():
    action = np.array([0.0, 0.0])

    def key_press(k, mod):
        global restart
        if k == 0xff0d: restart = True
        if k == key.LEFT:  action[1] = -1.0
        if k == key.RIGHT: action[1] = +1.0
        if k == key.UP:    action[0] = +1.0
        if k == key.DOWN:  action[0] = -0.8  # set 1.0 for wheels to block to zero rotation

    def key_release(k, mod):
        if k == key.LEFT and action[1] == -1.0: action[1] = 0
        if k == key.RIGHT and action[1] == +1.0: action[1] = 0
        if k == key.UP:    action[0] = 0
        if k == key.DOWN:  action[0] = 0

    env = Environment()
    env.reset()
    env.render()
    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release

    isopen = True

    while isopen:
        env.reset()
        total_reward = 0.0
        steps = 0
        restart = False
        while True:
            s, r, done, info = env.step(action)
            total_reward += r
            if steps % 200 == 0 or done:
                print("\naction " + str(["{:+0.2f}".format(x) for x in action]))
                print("step {} total_reward {:+0.2f}".format(steps, total_reward))
            steps += 1
            isopen = env.render()
            if done or restart or isopen is False:
                break

    env.close()


if __name__ == "__main__":
    play_as_human()
