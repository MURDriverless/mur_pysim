import numpy as np
from pathlib import Path


class Coordinates:
    path = Path(__file__).parents[1].joinpath('paths')

    @classmethod
    def save(cls, env) -> bool:
        """  Saves track coordinates as an enumerated dict """
        success = False

        track = np.array(env.track)
        x = list(track[:, 2])
        y = list(track[:, 3])
        track_xy = {}

        for i, track_x in enumerate(x):
            track_xy.update({i: (track_x, y[i])})

        with open(Path(cls.path).joinpath('track_coordinates.txt'), 'w') as file:
            file.write(str(track_xy))

        success = True

        return success

    @classmethod
    def load(cls, env=None, file_dir=None) -> dict:
        """ Loads track coordinates """
        if env is None:
            try:
                if file_dir is not None:
                    file_dir = '../paths/track_coordinates.txt'
                else:
                    file_dir = Path(cls.path).joinpath('track_coordinates.txt')

                with open(file_dir, 'r') as file:
                    track_xy = eval(file.read())
                print(f"File {file_dir} loaded!")
                return track_xy

            except FileNotFoundError:
                print(f"Error loading track coordinates. Use TrackEdit.save_coordinates() prior to loading")

        else:
            track = np.array(env.track)
            x = list(track[:, -2])
            y = list(track[:, -3])
            track_xy = {i: (track_x, y[i]) for i, track_x in enumerate(x)}

            return track_xy
