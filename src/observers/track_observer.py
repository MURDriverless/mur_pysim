from Box2D.b2 import contactListener
from ..world import ROAD_COLOR


"""
Purpose:
TrackObserver knows the tile the car is in contact with at any given time,
assuming that the car is the ONLY dynamic object in the static world. 

If the current tile is a track tile (meaning the car is along the track):
1. Append the tile to the list of visited tiles in the car
2. Reward the car for staying on track by notifying the Environment

Else, if the car is off-track, TrackObserver does nothing.
"""


class TrackObserver(contactListener):

    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        self._contact(contact, True)

    def EndContact(self, contact):
        self._contact(contact, False)

    def _contact(self, contact, begin):
        # The original code lists the variables as "tile" and "obj".
        # "tile" refers to the road tile in the track.
        # "obj" has been replaced with "car" as in our assumption, the
        # car is the only moving object.
        tile = None
        car = None
        # When contact happens, it is assumed that only two bodies are involved.
        # By convention, these are called A and B (as per Java's Box2D docs).
        # Here, we are accessing the properties of two bodies which are in contact.
        bodyA = contact.fixtureA.body.userData
        bodyB = contact.fixtureB.body.userData

        # Check if one of the bodies is a "road_friction" type.
        # If bodyA is one, set tile as bodyA and car as the other body.
        if bodyA and "road_friction" in bodyA.__dict__:
            tile = bodyA
            car = bodyB
        if bodyB and "road_friction" in bodyB.__dict__:
            tile = bodyB
            car = bodyA

        # If none of the bodies is a "road_friction" type, then the car
        # is off-track, so we do not need perform further operations, return.
        if not tile:
            return

        # Update the colour properties of the road
        tile.color[0] = ROAD_COLOR[0]
        tile.color[1] = ROAD_COLOR[1]
        tile.color[2] = ROAD_COLOR[2]

        # Assertion check: if car is None (null) or it does not have the "tiles"
        # property (a car is programmed to remember the tiles visited), return.
        if not car or "tiles" not in car.__dict__:
            return

        # If "begin" is True, happens when the contact starts, we perform the
        # aforementioned operations
        if begin:
            # Add the current tile to visited tiles in the car
            # (even if it has been visited before)
            car.tiles.add(tile)
            # If the tile has not been visited previously,
            # mark it as visited and notify Environment
            if not tile.road_visited:
                tile.road_visited = True
                # Since the tile represents the road in world.py, we want to reward
                # the car for staying on the track road by notifying self.env:
                # 1. Add static reward by amount of 1000 / total length of simulation track.
                #    Note that 1000 is the total frames allowed to finish the game.
                # 2. Increase the number of visited tiles by 1
                self.env.reward += 1000.0 / len(self.env.track)
                self.env.tile_visited_count += 1
        # If the car has ended contact (begin = False), meaning it has arrived
        # at the new tile, remove the current tile the car is on, as we will
        # perform the update operation again at the next iteration
        else:
            car.tiles.remove(tile)
            # print tile.road_friction, "DEL", len(obj.tiles) -- should delete to zero when on grass (this works)
