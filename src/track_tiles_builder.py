import math
import numpy as np
from .track_coordinates_builder import SCALE, TRACK_DETAIL_STEP, TRACK_TURN_RATE

TRACK_WIDTH = 40 / SCALE
BORDER = 8 / SCALE
BORDER_MIN_COUNT = 4
ROAD_COLOR = [0.4, 0.4, 0.4]


def create_track_tiles(env, track_coordinates, load_track=True, verbose=False):
    """ Create track tiles using the track coordinates previously built

    Parameters
    ----------
    env: Environment
    track_coordinates: list of (float, float, float, float)
    load_track: bool
        Required to enforce i1 and i2 variable below
    verbose: bool

    Returns
    -------
    (tiles, tiles_poly)
    tiles: list of Object
        Check _create_tile() function below to see the return type
    tiles_poly: list of (vertices, colour)
        vertices: Check _get_tile_vertices() function to see the return type
        colour: [float, float, float]; see _get_tile_colour() function below to see return type

    Side Effects
    ------------
    1. env.fd_tile.shape.vertices = tile_vertices
    """
    if not load_track:
        i1, i2 = -1, -1
        if verbose:
            print("Track generation: %i..%i -> %i-tiles track" % (i1, i2, i2 - i1))

    tiles = []
    tiles_poly = []

    first_beta = track_coordinates[0][1]
    first_perp_x = math.cos(first_beta)
    first_perp_y = math.sin(first_beta)

    # Length of perpendicular jump to put together head and tail
    well_glued_together = np.sqrt(
        np.square(first_perp_x * (track_coordinates[0][2] - track_coordinates[-1][2])) +
        np.square(first_perp_y * (track_coordinates[0][3] - track_coordinates[-1][3])))
    if well_glued_together > TRACK_DETAIL_STEP:
        return False

    # Red-white border on hard turns
    border = [False] * len(track_coordinates)
    for i in range(len(track_coordinates)):
        good = True
        oneside = 0
        for neg in range(BORDER_MIN_COUNT):
            beta1 = track_coordinates[i - neg - 0][1]
            beta2 = track_coordinates[i - neg - 1][1]
            good &= abs(beta1 - beta2) > TRACK_TURN_RATE * 0.2
            oneside += np.sign(beta1 - beta2)
        good &= abs(oneside) == BORDER_MIN_COUNT
        border[i] = good

    for i in range(len(track_coordinates)):
        for neg in range(BORDER_MIN_COUNT):
            border[i - neg] |= border[i]

    for i in range(len(track_coordinates)):
        previous_track_coordinate = track_coordinates[i - 1]
        current_track_coordinate = track_coordinates[i]

        # Get the tile properties
        tile_vertices = _get_tile_vertices(previous_track_coordinate, current_track_coordinate)
        tile_colour = _get_tile_colour(i, ROAD_COLOR)
        tile_poly = _get_poly_data(tile_vertices, tile_colour)

        # Create tile
        env.fd_tile.shape.vertices = tile_vertices
        tile = _create_tile(env, env.fd_tile, tile_colour)

        tiles.append(tile)
        tiles_poly.append(tile_poly)

        # Get the border properties
        if border[i]:
            border_vertices = _get_border_vertices(previous_track_coordinate, current_track_coordinate)
            border_colour = _get_border_colour(i)
            border_poly = _get_poly_data(border_vertices, border_colour)
            tiles_poly.append(border_poly)

    return tiles, tiles_poly


def _get_tile_vertices(previous_track_coordinate, current_track_coordinate):
    # Unpack values
    alpha1, beta1, x1, y1 = current_track_coordinate
    alpha2, beta2, x2, y2 = previous_track_coordinate
    # Set the vertices value
    road1_l = (x1 - TRACK_WIDTH * math.cos(beta1), y1 - TRACK_WIDTH * math.sin(beta1))
    road1_r = (x1 + TRACK_WIDTH * math.cos(beta1), y1 + TRACK_WIDTH * math.sin(beta1))
    road2_l = (x2 - TRACK_WIDTH * math.cos(beta2), y2 - TRACK_WIDTH * math.sin(beta2))
    road2_r = (x2 + TRACK_WIDTH * math.cos(beta2), y2 + TRACK_WIDTH * math.sin(beta2))
    return [road1_l, road1_r, road2_r, road2_l]


def _get_border_vertices(previous_track_coordinate, current_track_coordinate):
    # Unpack values
    alpha1, beta1, x1, y1 = current_track_coordinate
    alpha2, beta2, x2, y2 = previous_track_coordinate
    side = np.sign(beta2 - beta1)
    # Set the vertices value
    b1_l = (x1 + side * TRACK_WIDTH * math.cos(beta1), y1 + side * TRACK_WIDTH * math.sin(beta1))
    b1_r = (x1 + side * (TRACK_WIDTH + BORDER) * math.cos(beta1),
            y1 + side * (TRACK_WIDTH + BORDER) * math.sin(beta1))
    b2_l = (x2 + side * TRACK_WIDTH * math.cos(beta2), y2 + side * TRACK_WIDTH * math.sin(beta2))
    b2_r = (x2 + side * (TRACK_WIDTH + BORDER) * math.cos(beta2),
            y2 + side * (TRACK_WIDTH + BORDER) * math.sin(beta2))
    return [b1_l, b1_r, b2_r, b2_l]


def _get_tile_colour(tile_index, road_colours):
    # To determine different shade colour
    shader_coefficient = 0.01 * (tile_index % 3)
    return [road_colour + shader_coefficient for road_colour in road_colours]


def _get_border_colour(tile_index):
    return (1, 1, 1) if tile_index % 2 == 0 else (1, 0, 0)


def _get_poly_data(vertices, colour):
    return vertices, colour


def _create_tile(env, fd_tile, tile_colour):
    tile = env.world.CreateStaticBody(fixtures=fd_tile)
    tile.userData = tile
    tile.color = tile_colour
    tile.road_visited = False
    tile.road_friction = 1.0
    tile.fixtures[0].sensor = True
    return tile
