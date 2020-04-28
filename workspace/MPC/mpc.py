"""
Need 3 main components:
- Model State
- Actuators
- Update Equations

------------------------------------------
MODEL STATE
- x-pos of vehicle (x)
- y-pos of vehicle (y)
- orientation of vehicle (theta)
- velocity of vehicle (v)

ACTUATORS
- delta steering angle (delta-sa)
- acceleration (a)

UPDATE EQUATIONS
[
    x(t+1) = x(t) + v(t) * cos(sa) * dt
    y(t+1) = y(t) + v(t) * sin(sa) * dt
    sa(t+1) = sa(t) + (v(t) / Lf) * delta-sa * dt
    v(t+1) = v(t) + a * dt
]
------------------------------------------

"""