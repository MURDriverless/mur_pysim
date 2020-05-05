# LTV MPC with Virtual State
1. Set the initial input guess to zero
2. Set n = 1

while n <= N do:

    3. Obtain guesses for inputs over horizon n
    4. Calculate guesses for states over horizon n
    5. Linearize system dynamics over guesses
    6. Linearize track boundary optimization function over guesses
    7. Linearize track error over guessees
    8. Solve LTV MPC with Virtual State over horizon n
    9. Calculate inputs over horizon n handling constraints
    10. Set n = n + 1

## 3. Obtain Guesses for Inputs over horizon
This involves 


## 4. Calculate guesses for states over horizon

## 5. Linearize system dynamics over guesses

## 6. Linearize track boundary optimization 

## 7. Linearize track error over guesses

## 8. Solve LTV MPC with Virtual State over horizon

## 9. Calculate inputs over horizon n w/ constraints 

## Tracking Error
Tracking error is the difference between the position of the car
and the virtual reference. The virtual reference is the position
along the path to the current virtual state.

# Virtual State


# Increasing Horizon
THe increasing horizon initializer guesses for inputs over the
horizon. This is necessary because the LTV approach requires
guesses of the inputs and states to solve for inputs.