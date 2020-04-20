import car_racing
from utils import track, checkpoints, debugging

if __name__ == "__main__":
    env = car_racing.CarRacing(load_track=True)
    env.reset()
    track_xy = track.Coordinates.load()
    cp = checkpoints.Checkpoint(track_xy)

    steps, total_reward = 0, 0

    while True:
        env.render()
        mouse_pos = debugging.MouseController.position()
        observation, reward, done, info = env.step(mouse_pos)
        car_pos = (env.car.hull.position[0], env.car.hull.position[1])
        cp.check(car_pos)

        if steps % 200 == 0 or done:
            print(f"Step: {steps} Total Reward: {total_reward}")

        steps += 1
