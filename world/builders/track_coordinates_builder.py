import math
import os
import pickle

SCALE = 6.0
TOTAL_CHECKPOINTS = 12
TRACK_RADIUS = 900 / SCALE
# Describes the number of track tiles between each checkpoints
TRACK_DETAIL_STEP = 21 / SCALE
TRACK_TURN_RATE = 0.31

# Get absolute path of the tracks directory
TRACKS_DIR = os.path.abspath(
    # os.getcwd() will always resolve to the mur_pysim root directory
    os.path.join(os.getcwd(), 'tracks')
)


def build_track_coordinates(env, load_track=True):
    """ Build track coordinates from either loading previous track or generating a new one

    Parameters
    ----------
    env: Environment
    load_track: bool

    Returns
    -------
    list of (float, float, float, float)
    Format follows the result of calling _create_track_edges()
    """
    if load_track is True:
        track = _load_track_coordinates_from_file(env)
    else:
        track = _generate_track_coordinates_randomly(env, save_track=True)

    return track


def _load_track_coordinates_from_file(env):
    """ Create track from an existing file

    Parameters
    ----------
    env: Environment

    Returns
    -------
    list of (float, float, float, float)
    Format follows the result of calling _create_track_edges()
    """
    all_tracks = []

    # Open the "tracks" directory and get the track files ending in .txt
    for file in os.listdir(TRACKS_DIR):
        if file.endswith('.txt'):
            all_tracks.append(os.path.join(TRACKS_DIR, file))

    all_tracks.sort()

    # Load the last track in all_tracks
    with open(os.path.join(all_tracks[-1]), 'rb') as f:
        track = pickle.load(f)

    # Not sure what start_alpha does
    env.start_alpha = 2 * math.pi * (-0.5) / TOTAL_CHECKPOINTS

    # Debug message to tell that the file has been loaded
    print(f'Track: {all_tracks[-1]} loaded!')

    # Return the list of track coordinates
    return track


def _generate_track_coordinates_randomly(env, save_track=True):
    """ Programmatically create track and optionally save it

    Parameters
    ----------
    env: Environment
    save_track: bool

    Returns
    -------
    list of (float, float, float, float)
    Format follows the result of calling _create_track_edges()
    """
    # Create checkpoint nodes
    checkpoint_nodes = [
        _create_checkpoint_node(env, i, TOTAL_CHECKPOINTS, TRACK_RADIUS) for i in range(TOTAL_CHECKPOINTS)]

    # Generate track coordinates which connect the checkpoint nodes
    track_edges = _create_track_edges(checkpoint_nodes, TRACK_RADIUS, TRACK_TURN_RATE, TRACK_DETAIL_STEP)

    # Optionally save track
    if save_track is True:
        _save_track_coordinates(track_edges)

    return track_edges


def _save_track_coordinates(track_edges):
    """ Save track edges into a file, where the file name is generated automatically

    Parameters
    ----------
    track_edges: list of (float, float, float, float)
        Format is (alpha, beta, x, y) after calling _create_track_edges()

    Returns
    -------
    None
    """
    # If no existing tracks, save as track-0.txt
    if len(os.listdir(TRACKS_DIR)) == 0:
        with open(os.path.join(TRACKS_DIR, 'track-0.txt'), 'wb') as file:
            pickle.dump(track_edges, file)
    else:
        dir_items = sorted(os.listdir(TRACKS_DIR))
        # Get the number of existing files in TRACKS_DIR, and add it by 1
        # which will be the index of the new track coordinates
        index = max([int(num) for num in dir_items[-1] if num.isdigit()]) + 1
        # Save the new track coordinates with the new index
        with open(os.path.join(TRACKS_DIR, f'track-{index}.txt'), 'wb') as file:
            pickle.dump(track_edges, file)
    print("Track Saved.")


def _create_checkpoint_node(env, checkpoint_index, total_checkpoints, track_radius):
    """ Generate the coordinates of a checkpoint node in cartesian form

    Parameters
    ----------
    env: Environment
        Used to write start_alpha attribute and access np_random utils
    checkpoint_index: int
        The index of current checkpoint. Used to calculate multiples of 'elementary' angle
    total_checkpoints: int
        The total number of checkpoints. Used to calculate the elementary angle 'alpha'
    track_radius: float
        Used to convert from polar coordinates to cartesian coordinates, along with 'alpha'

    Returns
    -------
    (float, float, float)
    alpha: float
        Elementary angle in radians, required to provide information about the polar form
    x: float
    y: float
    """
    if checkpoint_index == 0:
        alpha = 0
        radius = 1.5 * track_radius
    elif checkpoint_index == total_checkpoints - 1:
        # Set alpha as the total angle in a circle (2*pi) divided by the number of checkpoints
        # and multiplied with the index of the current checkpoint, which is the last index
        alpha = 2 * math.pi * checkpoint_index / total_checkpoints
        radius = 1.5 * track_radius
        env.start_alpha = 2 * math.pi * (-0.5) / total_checkpoints
    else:
        # Similar to alpha above, but this time we add it by a random angle
        # between 0 and the elementary angle of each checkpoint (1/total_checkpoints)
        alpha = 2 * math.pi * checkpoint_index / total_checkpoints + \
                env.np_random.uniform(0, 2 * math.pi * 1 / total_checkpoints)
        # Randomly pick the radius between a third of the track radius and full radius
        radius = env.np_random.uniform(track_radius / 3, track_radius)

    # Return result in cartesian coordinates
    return alpha, radius * math.cos(alpha), radius * math.sin(alpha)


def _create_track_edges(checkpoint_nodes, track_radius, track_turn_rate, track_detail_step):
    """ Return a list of track edges connecting checkpoint nodes in cartesian coordinates

    Parameters
    ----------
    checkpoint_nodes: list of (float, float, float)
        The list of checkpoint nodes in the form of (alpha, x, y), generated by calling
        _create_checkpoint_node over the total number of checkpoints.
    track_radius: float
    track_turn_rate: float
    track_detail_step: float

    Returns
    -------
    list of (float, float, float, float)
    alpha: float
    beta: float
    x: float
    y: float
    """
    x, y, beta = 1.5 * track_radius, 0, 0
    dest_i = 0
    laps = 0
    track_edges = []
    no_freeze = 2500
    # Whether we are journeying from
    visited_other_side = False

    # No idea
    while True:
        alpha = math.atan2(y, x)
        if visited_other_side and alpha > 0:
            laps += 1
            visited_other_side = False

        if alpha < 0:
            visited_other_side = True
            alpha += 2 * math.pi

        # Find destination from checkpoints (no idea)
        while True:
            failed = True

            while True:
                dest_alpha, dest_x, dest_y = checkpoint_nodes[dest_i % len(checkpoint_nodes)]
                if alpha <= dest_alpha:
                    failed = False
                    break

                dest_i += 1
                if dest_i % len(checkpoint_nodes) == 0:
                    break
            if not failed:
                break

            alpha -= 2 * math.pi
            continue

        # Declare variables which I don't understand
        r1x = math.cos(beta)
        r1y = math.sin(beta)
        p1x = -r1y
        p1y = r1x
        dest_dx = dest_x - x
        dest_dy = dest_y - y

        # Destination vector projected on radius
        projection = r1x * dest_dx + r1y * dest_dy

        # Normalise beta-alpha: if they are larger than 1.5pi, reduce beta by
        # a full revolution of 2pi until they are smaller than 1.5pi.
        # Similarly, if they are smaller than -1.5pi, add beta by a full revolution
        # until it is larger than 1.5pi
        while beta - alpha > 1.5 * math.pi:
            beta -= 2 * math.pi
        while beta - alpha < -1.5 * math.pi:
            beta += 2 * math.pi

        # No idea
        prev_beta = beta
        projection *= SCALE

        # Normalise beta again by checking the projection value. If it is larger
        # than 0.3, reduce it once by min() function below. Conversely if projection
        # is smaller than -0.3, add it once by min() function below
        if projection > 0.3:
            beta -= min(track_turn_rate, abs(0.001 * projection))
        if projection < -0.3:
            beta += min(track_turn_rate, abs(0.001 * projection))

        # x and y go further into "one detail-step" (similar to one time-step)
        x += p1x * track_detail_step
        y += p1y * track_detail_step

        # Append edge to list of tracks
        track_edges.append((alpha, prev_beta * 0.5 + beta * 0.5, x, y))

        # No idea
        if laps > 4:
            break
        no_freeze -= 1
        if no_freeze == 0:
            break

    return track_edges
