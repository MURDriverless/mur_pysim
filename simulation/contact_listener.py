from Box2D import b2ContactListener


class ContactListener(b2ContactListener):
    def __init__(self, env):
        b2ContactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        self._contact(contact, True)

    def EndContact(self, contact):
        self._contact(contact, False)

    def _contact(self, contact, begin):
        # Declare variables we want to define
        track_tile = None
        car = None

        # When contact happens, it is assumed that only two bodies are involved.
        bodyA = contact.fixtureA.body
        bodyB = contact.fixtureB.body

        # Check if one of the bodies is a "road_friction" type.
        # If bodyA is one, set track_tile as bodyA and car as the other body.
        if bodyA.userData and "road_friction" in bodyA.userData.__dict__:
            track_tile = bodyA
            car = bodyB
        if bodyB.userData and "road_friction" in bodyB.userData.__dict__:
            track_tile = bodyB
            car = bodyA

        # If none of the bodies is a "road_friction" type, then the car
        # is off-track, so we do not need perform further operations, return.
        if car is not None and not track_tile:
            return

        # Assertion check: if car is None (null), return.
        if not car:
            return

        # If "begin" is True, happens when the contact starts, we perform the
        # aforementioned operations
        if begin:
            # If the tile has not been visited previously,
            # mark it as visited and notify Environment
            if not track_tile.userData.road_visited:
                track_tile.userData.road_visited = True
                # Since the tile represents the road, we want to reward
                # the car for staying on the track road by notifying self.env:
                # 1. Add static reward by amount of 1000 / total length of simulation track.
                #    Note that 1000 is the total frames allowed to finish the game.
                # 2. Increase the number of visited tiles by 1
                self.env.reward += 1000.0 / len(self.env.track_tiles)
                self.env.tile_visited_count += 1
        # # If the car has ended contact (begin = False), meaning it has arrived
        # # at the new tile, remove the current tile the car is on, as we will
        # # perform the update operation again at the next iteration
        # else:
        #     pass
        #     # car.tiles.remove(track_tile)
        #     # print tile.road_friction, "DEL", len(obj.tiles) -- should delete to zero when on grass (this works)
