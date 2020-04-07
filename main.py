import car_racing
from pynput.mouse import Button, Controller

SCREEN_HEIGHT = 2160
SCREEN_HEIGHT_HALF = SCREEN_HEIGHT / 2
SCREEN_WIDTH = 3840
SCREEN_WIDTH_HALF = SCREEN_WIDTH / 2


def mouse_controller(controller):
    """ Function for converting the mouse position into continuous variable for acceleration and steering """
    x, y = controller.position

    if SCREEN_HEIGHT_HALF - y <= 0:
        z_ret = (y - SCREEN_HEIGHT_HALF) / 1000
        y_ret = 0
    else:
        z_ret = 0
        y_ret = (SCREEN_HEIGHT_HALF - y) / 1000

    return (x - SCREEN_WIDTH_HALF) / 1000, y_ret, z_ret


if __name__ == "__main__":
    env = car_racing.CarRacing(load_track=True)
    env.reset()
    mouse = Controller()

    total_reward = 0
    steps = 0

    while True:
        env.render()
        pos_x, pos_y, pos_z = mouse_controller(mouse)
        action = (pos_x, pos_y, pos_z)

        observation, reward, done, info = env.step(action)

        total_reward += reward

        if steps % 200 == 0 or done:
            print(f"Step: {steps} Total Reward: {total_reward}")

        steps += 1
