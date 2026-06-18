from enum import IntEnum
import numpy as np

class TargetState(IntEnum):
    X = 0
    Y = 1
    Z = 2
    VX = 3
    VY = 4
    VZ = 5
    WX = 6
    WY = 7
    WZ = 8

class Target:
    """Represents a hostile target with its state and simple 3D evasive maneuvering."""

    NUM_STATES = len(TargetState)

    def __init__(self, initial_state: np.ndarray, length: float = 12.0, diameter: float = 1.0):
        self.state = initial_state
        self.L = length
        self.D_ref = diameter

    def position(self):
        return self.state[TargetState.X:TargetState.Z+1]

    def velocity(self):
        return self.state[TargetState.VX:TargetState.VZ+1]

    def acceleration(self):
        v = self.velocity()
        w = self.state[TargetState.WX:TargetState.WZ+1]
        return np.cross(w, v)

    def dynamics(self, target_state):
        v = target_state[TargetState.VX:TargetState.VZ+1]
        w = target_state[TargetState.WX:TargetState.WZ+1]

        dpdt = v
        dvdt = np.cross(w, v)
        dwdt = np.zeros(3, dtype=float) # No change in angular velocity for simplicity
        return np.hstack((dpdt, dvdt, dwdt))