import numpy as np

class MissileGuidance:
    def __init__(self, max_lat_accel, N=4.0):
        self.max_lat_accel = max_lat_accel
        self.N = N

    def compute_guidance(self, missile_pos, missile_vel, target_pos, target_vel):
            """Calculates the desired lateral acceleration guidance command, perpendicular to the missile's velocity vector, using the pure proportional navigation (PN) guidance law.

            missile_pos: Position vector of the missile in the inertial frame
            missile_vel: Velocity vector of the missile in the inertial frame
            target_pos: Position vector of the target in the inertial frame
            target_vel: Velocity vector of the target in the inertial frame
            N: Proportional navigation constant
            """

            rel_pos = target_pos - missile_pos
            rel_vel = target_vel - missile_vel
            rel_range = np.linalg.norm(rel_pos)

            if rel_range < 1e-3:
                return np.zeros(3)  # Avoid division by zero if target is extremely close

            los_dir = rel_pos / rel_range
            los_rate = np.cross(los_dir, rel_vel) / rel_range

            # Pure PN commands acceleration perpendicular to the missile's velocity vector
            a_lat = self.N * np.cross(los_rate, missile_vel)

            # TODO: Add a feedforward term to counter-act gravity with additional lateral acceleration?
            # TODO: Test difference with/without against taget flying level and with no acceleration, and with missile launched level at the same altitude as the target

            return self.limit_lateral_accel(a_lat)

    def limit_lateral_accel(self, a_lat):
        """Limits the lateral acceleration to the missile's structural G-limits."""

        lat_accel_mag = np.linalg.norm(a_lat)
        if lat_accel_mag > self.max_lat_accel:
            a_lat = a_lat * (self.max_lat_accel / lat_accel_mag)
        return a_lat