from Box2D import b2ContactListener
from simulation.type_names import CONE_TYPE, TRACK_TILE_TYPE


class ContactListener(b2ContactListener):
    def __init__(self, env):
        b2ContactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        self._contact(contact, True)

    def EndContact(self, contact):
        self._contact(contact, False)

    def _contact(self, contact, begin):
        car = None
        cone = None
        track_tile = None

        bodyA = contact.fixtureA.body
        bodyB = contact.fixtureB.body

        # Since track_tile and cone are static bodies, they will never collide in Box2D's rules.
        if bodyA.userData and "type" in bodyA.userData.__dict__:
            if bodyA.userData.type is TRACK_TILE_TYPE:
                track_tile = bodyA
                car = bodyB
                cone = None
            elif bodyA.userData.type is CONE_TYPE:
                cone = bodyA
                car = bodyB
                track_tile = None
        if bodyB.userData and "type" in bodyB.userData.__dict__:
            if bodyB.userData.type is TRACK_TILE_TYPE:
                track_tile = bodyB
                car = bodyA
                cone = None
            elif bodyB.userData.type is CONE_TYPE:
                cone = bodyB
                car = bodyA
                track_tile = None

        # Assertion check: if car is None (null), return.
        if not car:
            return

        if begin:
            # If car has contacted a track tile, which means it's on-track, give reward.
            # The caveat is that the reward is only given if the car has not visited the tile
            # Otherwise, the agent may maximise reward by staying on the tile to avoid penalties
            if track_tile is not None and track_tile.userData.tile_visited is False:
                # 1. Mark tile as visited
                track_tile.userData.tile_visited = True
                # 2. Add static reward by amount of 1000 / total length of simulation track.
                #    Note that 1000 is the total frames allowed to finish the game.
                self.env.reward += 1000.0 / len(self.env.track_tiles)
                # 3. Increase the number of visited tiles by 1
                self.env.tile_visited_count += 1

            # If car has contacted a cone, we need to penalise it and terminate the simulation
            # early as we have failed
            # TODO: Activate penalty for hitting cones
            # if cone is not None:
            #     # Provide penalty relative to the total distance travelled:
            #     # The more tiles the car has visited, the lower the penalty will be
            #     penalty = 1000 * (1 - float(self.env.tile_visited_count / len(self.env.track_tiles)))
            #     # 1. Penalise agent
            #     self.env.reward -= penalty
