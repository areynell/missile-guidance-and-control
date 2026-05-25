from enum import IntEnum

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# from scipy.integrate import solve_ivp

from controller import MissileController
import utils

class MissileState(IntEnum):
    X = 0
    Y = 1
    Z = 2
    QW = 3
    QX = 4
    QY = 5
    QZ = 6
    VX = 7
    VY = 8
    VZ = 9
    WX = 10
    WY = 11
    WZ = 12
    M = 13

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

class Missile:
    """Represents a missile with physical parameters and state, as well as associated dynamics and guidance law."""

    NUM_STATES = len(MissileState)

    def __init__(self, initial_state, missile_params, atmospheric_params):
        self.state = np.array([
            initial_state['x'],
            initial_state['y'],
            initial_state['z'],
            initial_state['qw'],
            initial_state['qx'],
            initial_state['qy'],
            initial_state['qz'],
            initial_state['vx'],
            initial_state['vy'],
            initial_state['vz'],
            initial_state['wx'],
            initial_state['wy'],
            initial_state['wz'],
            initial_state['m'],], dtype=float)

        self.g0 = 9.81
        self.Isp = missile_params['Isp']
        self.T = missile_params['T']

        # Drag coefficients
        self.CD_0 = missile_params['CD_0'] # Base drag coefficient at zero angle of attack, based on parasitic/skin friction drag of the missile's body and fins
        self.CD_alpha = missile_params['CD_alpha'] # Drag static stability derivative (dCD/dAlpha)
        self.CD_delta = missile_params['CD_delta'] # Drag control derivative (dCD/dDelta)

        # Lift coefficients
        self.CL_0 = missile_params['CL_0'] # Base lift coefficient at zero angle of attack, which is typically 0 for a symmetric missile body
        self.CL_alpha = missile_params['CL_alpha'] # Lift static stability derivative (dCL/dAlpha)
        self.CL_delta = missile_params['CL_delta'] # Lift control derivative (dCL/dDelta)

        # Side-force coefficients
        self.CY_0 = missile_params['CY_0'] # Base side-force coefficient at zero sideslip, which is typically 0 for a symmetric missile body
        self.CY_beta = missile_params['CY_beta'] # Side-force static stability derivative (dCY/dBeta)
        self.CY_delta = missile_params['CY_delta'] # Side-force control derivative (dCY/dDelta)

        # Rolling moment coefficients
        self.Cl_0 = missile_params['Cl_0'] # Base rolling moment coefficient at zero aileron deflection, which is typically 0 for a symmetric missile body
        self.Cl_p = missile_params['Cl_p'] # Roll damping dynamic stability derivative (dCl/dp)
        self.Cl_delta = missile_params['Cl_delta'] # Roll control derivative (dCl/dDelta)

        # Pitching moment coefficients
        self.Cm_0 = missile_params['Cm_0'] # Base pitching moment coefficient at zero angle of attack, which is typically 0 for a symmetric missile body
        self.Cm_alpha = missile_params['Cm_alpha'] # Pitch static stability derivative (dCm/dAlpha)
        self.Cm_q = missile_params['Cm_q'] # Pitch damping dynamic stability derivative (dCm/dq)
        self.Cm_delta = missile_params['Cm_delta'] # Pitch control derivative (dCm/dDelta)

        # Yawing moment coefficients
        self.Cn_0 = missile_params['Cn_0'] # Base yawing moment coefficient at zero sideslip, which is typically 0 for a symmetric missile body
        self.Cn_beta = missile_params['Cn_beta'] # Yaw static stability derivative (dCn/dBeta)
        self.Cn_r = missile_params['Cn_r'] # Yaw damping dynamic stability derivative (dCn/dr)
        self.Cn_delta = missile_params['Cn_delta'] # Yaw control derivative (dCn/dDelta)

        print("Missile Aerodynamic Force Coefficients: CD_0 = {:.3f}, CD_alpha = {:.3f}, CL_0 = {:.3f}, CL_alpha = {:.3f}, CL_delta = {:.3f}, CY_0 = {:.3f}, CY_beta = {:.3f}, CY_delta = {:.3f}".format(
            self.CD_0, self.CD_alpha, self.CL_0, self.CL_alpha, self.CL_delta, self.CY_0, self.CY_beta, self.CY_delta
        ))

        print("Missile Aerodynamic Moment Coefficients: Cl_0 = {:.3f}, Cl_p = {:.3f}, Cl_delta = {:.3f}, Cm_0 = {:.3f}, Cm_alpha = {:.3f}, Cm_q = {:.3f}, Cm_delta = {:.3f}, Cn_0 = {:.3f}, Cn_beta = {:.3f}, Cn_r = {:.3f}, Cn_delta = {:.3f}".format(
            self.Cl_0, self.Cl_p, self.Cl_delta, self.Cm_0, self.Cm_alpha, self.Cm_q, self.Cm_delta, self.Cn_0, self.Cn_beta, self.Cn_r, self.Cn_delta
        ))

        self.D_ref = missile_params['D_ref']
        self.A_ref = np.pi * (self.D_ref / 2.0)**2 # Reference area for aerodynamic force calculations, typically the maximum cross-sectional area of the missile
        self.L = missile_params['L']
        self.max_lat_accel = missile_params['max_lat_accel']
        self.m_dry = missile_params['m_dry']
        self.kill_radius = missile_params['kill_radius']

        self.rho0 = atmospheric_params['rho0']
        self.H_scale = atmospheric_params['H_scale']
        self.vi_wind = atmospheric_params['vi_wind']

        self.flight_phase = "BOOST"
        self.a_lat_desired = np.zeros(3, dtype=float)

        # Safe minimum velocity threshold for aerodynamic effectiveness
        self.v_min_aero = 1.0

        # Missile flight controller with appropriate gains for roll, pitch, and yaw control
        roll_gains = {
            'Kp_roll': 0.5,
            'Kp_roll_rate': 0.02,
            'Ki_roll_rate': 0.01
        }
        pitch_gains = {
            'Kdc_pitch': 1.0,
            'Ka_pitch_rate': 0.1,
            'Ki_pitch_rate': 0.25,
            'Kr_pitch_rate': 0.25
        }
        yaw_gains = {
            'Kdc_yaw': 1.0,
            'Ka_yaw_rate': 0.1,
            'Ki_yaw_rate': 0.25,
            'Kr_yaw_rate': 0.25
        }
        v_ref = 500.00 # Reference speed for gain scheduling
        P_dyn_ref = 0.5 * self.rho0 * v_ref**2 # Reference dynamic pressure for gain scheduling
        P_dyn_min = 100.0 # Minimum dynamic pressure for gain scheduling to avoid excessive control deflections at very low dynamic pressures
        integral_limit = 2.0 # Anti-windup limit for integral terms in the controller (rad)
        delta_limit = np.deg2rad(45.0) # Maximum control surface deflection (rad)
        self.controller = MissileController(roll_gains, pitch_gains, yaw_gains, P_dyn_min, P_dyn_ref, integral_limit, delta_limit)
        self.control_deltas = np.zeros(3, dtype=float) # [delta_a, delta_e, delta_r]

    def position(self):
        """Returns the missile's position vector in the inertial frame."""
        return self.state[MissileState.X:MissileState.Z+1]

    def orientation(self):
        """Returns the missile's orientation as a quaternion (qw, qx, qy, qz) representing the rotation from the body frame to the inertial frame."""
        return self.state[MissileState.QW:MissileState.QZ+1]

    def velocity(self):
        """Returns the missile's velocity vector in the body frame."""
        return self.state[MissileState.VX:MissileState.VZ+1]

    def angular_velocity(self):
        """Returns the missile's angular velocity vector in the body frame."""
        return self.state[MissileState.WX:MissileState.WZ+1]

    def speed(self):
        """Returns the missile's speed (velocity magnitude)."""
        return np.linalg.norm(self.velocity())

    def mass(self):
        """Returns the missile's mass."""
        return self.state[MissileState.M]

    def alpha(self, vb_missile, vi_wind, R_bi):
        """Calculates the missile's angle of attack based on the its velocity vector in the body frame and the wind velocity in the inertial frame."""
        vb_rel = vb_missile - R_bi @ vi_wind
        return np.arctan2(vb_rel[2], vb_rel[0])

    def beta(self, vb_missile, vi_wind, R_bi):
        """Calculates the missile's sideslip angle based on the its velocity vector in the body frame and the wind velocity in the inertial frame."""
        vb_rel = vb_missile - R_bi @ vi_wind
        vb_rel_mag = np.linalg.norm(vb_rel)
        if vb_rel_mag < self.v_min_aero:
            return 0.0 # Prevent divide-by-zero at launch
        return np.arcsin(np.clip(vb_rel[1] / vb_rel_mag, -1.0, 1.0))

    def update_flight_phase(self):
        """Updates the missile's current flight phase (BOOST or COAST) based on its mass relative to the dry mass after burnout."""
        if self.mass() > self.m_dry:
            self.flight_phase = "BOOST"
        else:
            self.flight_phase = "COAST"

    def current_flight_phase(self):
        """Returns the current flight phase of the missile (BOOST or COAST)."""
        return self.flight_phase

    def thrust(self, mass):
        """Returns the thrust vector along the missile's longitudinal axis."""
        if mass > self.m_dry:
            return np.array([self.T, 0.0, 0.0], dtype=float)
        else:
            return np.array([0.0, 0.0, 0.0], dtype=float)

    def compute_aerodynamic_forces(self, altitude, vb_missile, vi_wind, R_bi, control_deltas):
        """Calculates the aerodynamic forces acting on the missile in the wind frame, and then transforms them back to the body frame."""

        # All aerodynamic forces are computed w.r.t. the wind (relative velocity) frame, since this is what the missile "feels" aerodynamically
        vb_rel = vb_missile - R_bi @ vi_wind
        vb_rel_mag = np.linalg.norm(vb_rel)

        # Aerodynamic angles
        if vb_rel_mag > self.v_min_aero:
            alpha = np.arctan2(vb_rel[2], vb_rel[0]) # Angle of attack
            beta = np.arcsin(np.clip(vb_rel[1] / vb_rel_mag, -1.0, 1.0)) # Sideslip angle
        else:
            return np.zeros(3, dtype=float), np.zeros(3, dtype=float) # No aerodynamic forces if relative velocity is near zero to avoid numerical issues

        # Dynamic pressure
        altitude = max(altitude, 0.0) # Altitude clamped to zero
        rho = self.rho0 * np.exp(-altitude / self.H_scale)
        P_dyn = 0.5 * rho * vb_rel_mag**2

        # Roll (aileron), pitch (elevator) and yaw (rudder) control surface deflections
        delta_a, delta_e, delta_r = control_deltas

        # Calculate aerodynamic force coefficients in the wind (relative velocity) frame
        CD = self.CD_0 + (self.CD_alpha * alpha**2) + self.CD_delta * (delta_e**2 + delta_r**2)
        CY = self.CY_0 + (self.CY_beta * beta) + (self.CY_delta * delta_r)
        CL = self.CL_0 + (self.CL_alpha * alpha) + (self.CL_delta * delta_e)

        # Calculate aerodynamic forces in the wind (relative velocity) frame
        # NOTE: It's standard for A_ref (reference area) to be a constant (i.e., it does NOT change with alpha or beta). For missiles, this is typically the missile's maximum cross-sectional area
        Fw_drag = P_dyn * self.A_ref * CD
        Fw_sideslip = P_dyn * self.A_ref * CY
        Fw_lift = P_dyn * self.A_ref * CL

        # Aerodynamic force vector in the wind (relative velocity) frame
        Fw_aero = np.array([-Fw_drag, Fw_sideslip, -Fw_lift], dtype=float)

        # Transformation of aerodynamic forces from the wind (relative velocity) frame to body frame
        # Angle of attack rotation matrix (about the y-axis)
        Ry_alpha = np.array([[np.cos(alpha), 0.0, -np.sin(alpha)],
                             [0.0,           1.0,            0.0],
                             [np.sin(alpha), 0.0,  np.cos(alpha)]], dtype=float)
        # Sideslip rotation matrix (about the z-axis)
        Rz_beta = np.array([[np.cos(beta), -np.sin(beta), 0.0],
                            [np.sin(beta),  np.cos(beta), 0.0],
                            [0.0,           0.0,          1.0]], dtype=float)
        # TODO: Check that the rotation order is correct
        # R_bw = Ry_alpha @ Rz_beta
        R_bw = Rz_beta @ Ry_alpha
        Fb_aero = R_bw @ Fw_aero

        return Fb_aero, Fw_aero

    def compute_aerodynamic_moments(self, altitude, vb_missile, vi_wind, R_bi, wb, control_deltas):

        # Relative velocity in missile's body frame
        vb_rel = vb_missile - R_bi @ vi_wind
        vb_rel_mag = np.linalg.norm(vb_rel)

        # Aerodynamic angles
        if vb_rel_mag > self.v_min_aero:
            alpha = np.arctan2(vb_rel[2], vb_rel[0]) # Angle of attack
            beta = np.arcsin(np.clip(vb_rel[1] / vb_rel_mag, -1.0, 1.0)) # Sideslip angle
        else:
            return np.zeros(3, dtype=float) # No aerodynamic moments if relative velocity is near zero to avoid numerical issues

        # Dynamic pressure
        altitude = max(altitude, 0.0) # Altitude clamped to zero
        rho = self.rho0 * np.exp(-altitude / self.H_scale)
        P_dyn = 0.5 * rho * vb_rel_mag**2

        # Angular velocity components in body frame
        wx, wy, wz = wb

        # Roll (aileron), pitch (elevator) and yaw (rudder) control surface deflections
        delta_a, delta_e, delta_r = control_deltas

        # Calculate aerodynamic moment coefficients
        CMx = self.Cl_0 + (self.Cl_p * wx * self.D_ref / (2.0 * vb_rel_mag)) + (self.Cl_delta * delta_a)
        CMy = self.Cm_0 + (self.Cm_alpha * alpha) + (self.Cm_q * wy * self.D_ref / (2.0 * vb_rel_mag)) + (self.Cm_delta * delta_e)
        CMz = self.Cn_0 + (self.Cn_beta * beta) + (self.Cn_r * wz * self.D_ref / (2.0 * vb_rel_mag)) + (self.Cn_delta * delta_r)

        # Calculate aerodynamic moments in body frame
        # NOTE: It's standard for A_ref (reference area) and D_ref (reference length) to be constants.
        # For missiles, A_ref is typically the maximum cross-sectional area and D_ref is typically the missile diameter.
        Mx = P_dyn*self.A_ref*self.D_ref*CMx
        My = P_dyn*self.A_ref*self.D_ref*CMy
        Mz = P_dyn*self.A_ref*self.D_ref*CMz

        return np.array([Mx, My, Mz], dtype=float)

    def desired_lateral_accel(self):
        """Returns the guidance law's desired lateral acceleration in the inertial frame."""
        return self.a_lat_desired

    def achieved_lateral_accel(self):
        """Computes the achieved lateral acceleration executed by the flight controller in the inertial frame, excluding thrust (axial) contribution."""

        # TODO: Check that these calculations are correct. Pure PN gives desired lateral acceleration relative to the missile's velocity vector, whereas
        # the achieved acceleration calculation here is based on the missile's body frame which is not necessarily aligned with the velocity vector

        p = self.position()
        vb = self.velocity()
        q = self.orientation()
        m = self.mass()

        R_ib = utils.quaternion_to_rotation_matrix(q)
        R_bi = R_ib.T

        Fb_aero, _ = self.compute_aerodynamic_forces(p[2], vb, self.vi_wind, R_bi, self.control_deltas)

        # Total specific force in body frame
        a_body = Fb_aero / m

        # Zero out the axial (thrust) direction — body X is the missile's longitudinal axis
        a_lat_body = np.array([0.0, a_body[1], a_body[2]], dtype=float)

        # Rotate lateral acceleration to inertial frame for comparison with desired lateral acceleration from guidance law
        return R_ib @ a_lat_body

    def pure_pn_guidance(self, missile_pos, missile_vel, target_pos, target_vel, N=4.0):
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
        a_lat = N * np.cross(los_rate, missile_vel)

        # TODO: Add a feedforward term to counter-act gravity with additional lateral acceleration?
        # TODO: Test difference with/without against taget flying level and with no acceleration, and with missile launched level at the same altitude as the target

        return self.limit_lateral_accel(a_lat)

    def limit_lateral_accel(self, a_lat):
        """Limits the lateral acceleration to the missile's structural G-limits."""

        lat_accel_mag = np.linalg.norm(a_lat)
        if lat_accel_mag > self.max_lat_accel:
            a_lat = a_lat * (self.max_lat_accel / lat_accel_mag)
        return a_lat

    def update_guidance(self, target):
        vb = self.velocity()
        q = self.orientation()
        R_ib = utils.quaternion_to_rotation_matrix(q)  # Body to inertial frame rotation matrix
        vi = R_ib @ vb # Velocity in inertial frame

        # TODO: Does PN guidance need to take inertial wind velocity into account?

        self.a_lat_desired = self.pure_pn_guidance(self.position(), vi, target.position(), target.velocity())
        self.update_flight_phase()

    def update_control(self, dt):
        p = self.position()
        vb = self.velocity()
        q = self.orientation()
        wb = self.angular_velocity()
        m = self.mass()

        R_ib = utils.quaternion_to_rotation_matrix(q) # Body to inertial frame rotation matrix
        R_bi = R_ib.T

        roll_cmd_rad = 0.0 # For skid-to-turn (STT) missiles, we typically command zero roll angle to keep the lift vector aligned with the desired lateral acceleration direction
        a_cmd_body = R_bi @ self.a_lat_desired

        # Compute missile's current lateral acceleration
        Fb_aero, _ = self.compute_aerodynamic_forces(p[2], vb, self.vi_wind, R_bi, self.control_deltas)
        a_body = Fb_aero / m # Lateral acceleration experienced by the missile

        # Dynamic pressure
        vb_rel = vb - R_bi @ self.vi_wind
        vb_rel_mag = np.linalg.norm(vb_rel)
        altitude = max(p[2], 0.0) # Altitude clamped to zero
        rho = self.rho0 * np.exp(-altitude / self.H_scale)
        P_dyn = 0.5 * rho * vb_rel_mag**2

        self.control_deltas = self.controller.update(roll_cmd_rad, a_cmd_body, a_body, wb, q, P_dyn, dt)

    def dynamics(self, missile_state):
        """6-DOF missile dynamics based on Newton-Euler equations for a rigid body, with forces and moments from gravity, thrust, and aerodynamics."""

        p = missile_state[MissileState.X:MissileState.Z+1]
        q = missile_state[MissileState.QW:MissileState.QZ+1]
        vb = missile_state[MissileState.VX:MissileState.VZ+1]
        wb = missile_state[MissileState.WX:MissileState.WZ+1]
        m = missile_state[MissileState.M]

        # Inertia matrix for a solid cylinder (missile body) about its center of mass, aligned with the body axes
        I = np.diag(np.array([
            0.5 * m * (self.D_ref / 2.0)**2,
            1.0/12.0 * m * (3.0 * (self.D_ref / 2.0)**2 + self.L**2),
            1.0/12.0 * m * (3.0 * (self.D_ref / 2.0)**2 + self.L**2)
        ], dtype=float))
        I_inv = np.linalg.inv(I)

        q = utils.quaternion_normalization(q)
        R_ib = utils.quaternion_to_rotation_matrix(q)  # Body to inertial
        R_bi = R_ib.T # Inertial to body

        # Constant gravity in missile's body frame
        Fi_grav = np.array([0.0, 0.0, -m*self.g0], dtype=float)
        Fb_grav = R_bi @ Fi_grav

        # Thrust force along missile's longitudinal body axis
        Fb_thrust = self.thrust(m)

        # Aerodynamic forces (drag, sideslip, lift) transformed from wind frame to body frame
        Fb_aero, _ = self.compute_aerodynamic_forces(p[2], vb, self.vi_wind, R_bi, self.control_deltas)

        # Total force in missile's body frame
        Fb_total = Fb_grav + Fb_thrust + Fb_aero

        # Aerodynamic moments (roll, pitch, yaw) in body frame
        Mb_aero = self.compute_aerodynamic_moments(p[2], vb, self.vi_wind, R_bi, wb, self.control_deltas)

        # Total moment/torque in missile's body frame
        Mb_total = Mb_aero

        # State derivatives based on Newton-Euler equations for a rigid body
        dpdt = R_ib @ vb
        dqdt = 0.5 * utils.quaternion_multiply(q, np.hstack((0.0, wb)))
        dvbdt = Fb_total / m - np.cross(wb, vb)
        dwbdt = I_inv @ (Mb_total - np.cross(wb, I @ wb))
        dmdt = -Fb_thrust[0] / (self.Isp * self.g0)

        return np.hstack((dpdt, dqdt, dvbdt, dwbdt, dmdt))

class Target:
    """Represents a hostile target with its state and simple 3D evasive maneuvering."""

    NUM_STATES = len(TargetState)

    def __init__(self, initial_state):
        self.state = np.array([
            initial_state['x'],
            initial_state['y'],
            initial_state['z'],
            initial_state['vx'],
            initial_state['vy'],
            initial_state['vz'],
            initial_state['wx'],
            initial_state['wy'],
            initial_state['wz']
        ], dtype=float)

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

def update_sim_states(missile: Missile, target: Target, dt: float):
    """Update missile and target jointly using scipy's solve_ivp."""

    # Combine missile and target states into a single vector for joint integration
    joint_state = np.hstack((missile.state, target.state))

    # Propagate the combined state forward by dt using adaptive integration (RK45 default)
    updated_state = rk4_update(combined_dynamics, joint_state, dt, missile, target)
    # sol = solve_ivp(combined_dynamics, (0.0, dt), joint_state, args=(missile, target))
    # updated_state = sol.y[:, -1]

    # Update missile and target states from the updated combined state vector
    missile.state = updated_state[:missile.NUM_STATES]
    target.state = updated_state[missile.NUM_STATES:missile.NUM_STATES + target.NUM_STATES]

# def combined_dynamics(t: float, state: np.ndarray, missile: Missile, target: Target) -> np.ndarray:
def combined_dynamics(state: np.ndarray, missile: Missile, target: Target) -> np.ndarray:
    """Combines the missile and target dynamics into a single state derivative vector."""

    # Split the combined state vector back into missile and target states
    missile_state = state[:missile.NUM_STATES]
    target_state = state[missile.NUM_STATES:missile.NUM_STATES + target.NUM_STATES]

    # Compute the derivatives for both missile and target
    missile_dynamics = missile.dynamics(missile_state)
    target_dynamics = target.dynamics(target_state)

    return np.hstack((missile_dynamics, target_dynamics))

def rk4_update(f: callable, state: np.ndarray, dt: float, *args) -> np.ndarray:
    """Propagate the simulation state forward by one time step using the 4th-order Runge-Kutta method."""

    k1 = f(state, *args)
    k2 = f(state + 0.5 * dt * k1, *args)
    k3 = f(state + 0.5 * dt * k2, *args)
    k4 = f(state + dt * k3, *args)

    next_state = state + (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)
    return next_state

def run_simulation():
    """Runs a missile interception simulation and collects data for analysis and visualization."""

    t = 0.0
    dt = 0.01
    dt_far = dt
    dt_close = 0.1*dt
    range_close = 1000.0
    t_max = 50.0

    # Parameters that ROUGHLY approximate MIM-104 Patriot PAC-2 variant
    missile_params = {
        'T': 150.0e3, # Thrust (N)
        # 'T': 70.0e3, # Thrust (N)
        'Isp': 260.0, # Specific impulse (s)
        'CD_0': 0.1, # Base drag coefficient at zero angle of attack, based on parasitic/skin friction drag of the missile's body and fins
        'CD_alpha': 5.0, # Drag static stability derivative (dCD/dAlpha), which captures how the drag coefficient increases with angle of attack
        'CD_delta': 2.0, # Drag control derivative (dCD/dDelta), which captures how the drag coefficient changes with control surface deflection
        'CY_0': 0.0, # Base side-force coefficient at zero sideslip, which is typically 0 for a symmetric missile body
        'CY_beta': -20.0, # Side-force static stability derivative (dCY/dBeta), which captures how the missile generates side force to counteract sideslip and maintain directional stability.
        'CY_delta': 2.0, # Side-force control derivative (dCY/dDelta), which captures how effective the rudder is at generating side force for yaw control
        'CL_0': 0.0, # Base lift coefficient at zero angle of attack, which is typically 0 for a symmetric missile body
        'CL_alpha': 20.0, # Lift static stability derivative (dCL/dAlpha), which captures how effective the missile's body and fins are at generating lift to achieve maneuvering
        'CL_delta': 2.0, # Lift control derivative (dCL/dDelta), which captures how effective the elevator is at generating lift for pitch control
        'Cl_0': 0.0, # Base rolling moment coefficient at zero aileron deflection, which is typically 0 for a symmetric missile body
        'Cl_p': -2.0, # Base roll damping dynamic stability derivative (dCl/dp), which captures how the missile's roll rate generates a restoring rolling moment to stabilize roll oscillations
        'Cl_delta': 0.5, # Roll control derivative (dCl/dDelta), which captures how effective the aileron is at generating rolling moment for roll control
        'Cm_0': 0.0, # Base pitching moment coefficient at zero angle of attack, which is typically 0 for a symmetric missile body
        'Cm_alpha': -2.0, # Pitch static stability derivative (dCm/dAlpha), which captures how the missile's angle of attack generates a restoring pitching moment to stabilize pitch oscillations
        'Cm_q': -2.0, # Pitch damping dynamic stability derivative (dCm/dq), which captures how the missile's pitch rate generates a restoring pitching moment to stabilize pitch oscillations
        'Cm_delta': 2.0, # Pitch control derivative (dCm/dDelta), which captures how effective the elevator is at generating pitching moment for pitch control
        'Cn_0': 0.0, # Base yawing moment coefficient at zero sideslip, which is typically 0 for a symmetric missile body
        'Cn_beta': 2.0, # Yaw static stability derivative (dCn/dBeta), which captures how the missile's sideslip angle generates a restoring yawing moment to stabilize directional oscillations
        'Cn_r': -2.0, # Yaw damping dynamic stability derivative (dCn/dr), which captures how the missile's yaw rate generates a restoring yawing moment to stabilize directional oscillations
        'Cn_delta': 2.0, # Yaw control derivative (dCn/dDelta), which captures how effective the rudder is at generating yawing moment for yaw control
        'D_ref': 0.41, # Missile diameter (m)
        'L': 5.0, # Missile length (m)
        'max_lat_accel': 35.0 * 9.81, # Maximum structural lateral acceleration (m/s^2) limit
        'm_total': 900.0, # Initial total mass (kg)
        'm_dry': 550.0, # Dry mass after burnout (kg)
        'kill_radius': 30.0, # Kill radius for successful interception (m)
    }

    # Initial pitch of 45 degrees
    initial_pitch = -np.deg2rad(45.0)
    qw_init = np.cos(initial_pitch / 2.0)
    qy_init = np.sin(initial_pitch / 2.0)

    missile_initial_state = {
        'x': 0.0,
        'y': 0.0,
        'z': 0.0,
        'qw': qw_init,
        'qx': 0.0,
        'qy': qy_init,
        'qz': 0.0,
        'vx': 0.0,
        'vy': 0.0,
        'vz': 0.0,
        'wx': 0.0,
        'wy': 0.0,
        'wz': 0.0,
        'm': missile_params['m_total']
    }

    atmospheric_params = {
        'rho0': 1.225, # kg/m^3 at sea level
        'H_scale': 8500.0, # Scale height for exponential atmosphere (meters)
        'vi_wind': np.array([0.0, 0.0, 0.0], dtype=float) # Wind velocity in inertial frame
    }

    missile = Missile(missile_initial_state, missile_params, atmospheric_params)

    np.random.seed(7)
    target_position_xy = np.array([
        np.random.uniform(10000.0, 20000.0),
        np.random.uniform(10000.0, 20000.0),
    ], dtype=float)

    # Making target's initial velocity vector point towards the xy origin
    target_speed = np.random.uniform(300.0, 800.0)
    direction_to_origin_xy = -target_position_xy / np.linalg.norm(target_position_xy)
    target_velocity_xy = target_speed * direction_to_origin_xy

    target_initial_state = {
        'x': target_position_xy[0],
        'y': target_position_xy[1],
        'z': np.random.uniform(10000.0, 20000.0),
        'vx': target_velocity_xy[0] + np.random.uniform(-100.0, 100.0),
        'vy': target_velocity_xy[1] + np.random.uniform(-100.0, 100.0),
        'vz':  np.random.uniform(-50.0, 50.0),
        'wx': np.random.uniform(-0.1, 0.1),
        'wy': np.random.uniform(-0.1, 0.1),
        'wz': np.random.uniform(-0.1, 0.1),
    }

    print("Initial Target State:", target_initial_state)

    target = Target(target_initial_state)

    # Data collection
    intercepted = False

    missile_history = {
        "time": [],
        "position": [],
        "orientation": [],
        "rpy_deg": [],
        "velocity": [],
        "angular_velocity": [],
        "mass": [],
        "flight_phase": [],
        "a_lat_desired": [],
        "a_lat_achieved": [],
        "thrust": [],
        "alpha": [],
        "beta": [],
        "control_deltas": [],
        "Fb_aero": [],
        "Fw_aero": [],
        "dynamic_pressure": []
    }

    target_history = {
        "time": [],
        "position": [],
        "velocity": []
    }

    print("Simulating missile interception...")

    # Guidance and simulation loop
    while t < t_max:
        rel_range = np.linalg.norm(target.position() - missile.position())

        # Use smaller time steps when missile is close to target for better interception accuracy
        if rel_range < range_close:
            dt = dt_close
        else:
            dt = dt_far

        # Computing lateral acceleration command at start of each guidance cycle based on current missile and target states
        missile.update_guidance(target)
        missile.update_control(dt)

        # assert abs(np.dot(missile.lateral_accel_cmd(), missile.velocity())) < 1.0e-6, "Lateral acceleration command is not perpendicular to velocity vector!"

        R_ib = utils.quaternion_to_rotation_matrix(missile.orientation())
        R_bi = R_ib.T
        alpha = missile.alpha(missile.velocity(), missile.vi_wind, R_bi)
        beta = missile.beta(missile.velocity(), missile.vi_wind, R_bi)

        print(f"Time: {t:.2f} s, "
              f"Missile Mass: {missile.mass():.1f} kg, "
              f"Speed: {missile.speed():.1f} m/s, "
              f"Distance to Target: {np.linalg.norm(target.position() - missile.position()):.1f} m, "
              f"Flight Phase: {missile.current_flight_phase()}, "
              f"Lateral G: {np.linalg.norm(missile.achieved_lateral_accel()) / missile.g0:.1f} G, "
              f"Alpha: {np.rad2deg(alpha):.1f} deg, "
              f"Beta: {np.rad2deg(beta):.1f} deg")

        missile_history["time"].append(t)
        missile_history["position"].append(missile.position().copy())
        missile_history["orientation"].append(missile.orientation().copy())
        missile_history["rpy_deg"].append(utils.quaternion_to_rpy_deg(missile.orientation()).copy())
        missile_history["velocity"].append(missile.velocity().copy())
        missile_history["angular_velocity"].append(missile.angular_velocity().copy())
        missile_history["mass"].append(missile.mass())
        missile_history["flight_phase"].append(missile.current_flight_phase())
        missile_history["a_lat_desired"].append(missile.desired_lateral_accel().copy())
        missile_history["a_lat_achieved"].append(missile.achieved_lateral_accel().copy())
        missile_history["thrust"].append(missile.thrust(missile.mass()))
        missile_history["alpha"].append(alpha)
        missile_history["beta"].append(beta)
        missile_history["control_deltas"].append(missile.control_deltas.copy())
        Fb_aero, Fw_aero = missile.compute_aerodynamic_forces(missile.position()[2], missile.velocity(), missile.vi_wind, R_bi, missile.control_deltas)
        missile_history["Fb_aero"].append(Fb_aero)
        missile_history["Fw_aero"].append(Fw_aero)
        missile_history["dynamic_pressure"].append(0.5 * missile.rho0 * np.exp(-missile.position()[2] / missile.H_scale) * np.linalg.norm(missile.velocity() - R_bi @ missile.vi_wind)**2)

        target_history["time"].append(t)
        target_history["position"].append(target.position().copy())
        target_history["velocity"].append(target.velocity().copy())

        if rel_range < missile.kill_radius:
            print(f"Proximity detonation. Target destroyed at {t:.2f} s. Distance: {rel_range:.1f} m")
            intercepted = True
            break
        if target.position()[2] < 0.0:
            print(f"Target impacted the ground at {t:.2f} s. Distance to missile: {rel_range:.1f} m")
            intercepted = False
            break
        if missile.position()[2] < 0.0:
            print(f"Missile impacted the ground at {t:.2f} s. Distance to target: {rel_range:.1f} m")
            intercepted = False
            break

        update_sim_states(missile, target, dt)
        t += dt

    if not intercepted:
        print("Target evaded interception.")

    # Convert dict of lists to dict of numpy arrays for easier plotting later
    missile_history["time"] = np.array(missile_history["time"])
    missile_history["position"] = np.array(missile_history["position"])
    missile_history["orientation"] = np.array(missile_history["orientation"])
    missile_history["rpy_deg"] = np.array(missile_history["rpy_deg"])
    missile_history["velocity"] = np.array(missile_history["velocity"])
    missile_history["angular_velocity"] = np.array(missile_history["angular_velocity"])
    missile_history["mass"] = np.array(missile_history["mass"])
    missile_history["a_lat_desired"] = np.array(missile_history["a_lat_desired"])
    missile_history["a_lat_achieved"] = np.array(missile_history["a_lat_achieved"])
    missile_history["thrust"] = np.array(missile_history["thrust"])
    missile_history["alpha"] = np.array(missile_history["alpha"])
    missile_history["beta"] = np.array(missile_history["beta"])
    missile_history["control_deltas"] = np.array(missile_history["control_deltas"])
    missile_history["Fb_aero"] = np.array(missile_history["Fb_aero"])
    missile_history["Fw_aero"] = np.array(missile_history["Fw_aero"])
    missile_history["dynamic_pressure"] = np.array(missile_history["dynamic_pressure"])
    target_history["time"] = np.array(target_history["time"])
    target_history["position"] = np.array(target_history["position"])
    target_history["velocity"] = np.array(target_history["velocity"])

    return missile_history, target_history, intercepted

def plot_metrics(missile_hist, target_hist):
    fig, axes = plt.subplots(4, 3, figsize=(16, 8), constrained_layout=True)
    fig.suptitle('Missile Interception Metrics', fontsize=14, weight='bold')

    # # Missile position vs. time
    # ax = axes[0, 0]
    # ax.plot(missile_hist["time"], missile_hist["position"][:, 0], label='x')
    # ax.plot(missile_hist["time"], missile_hist["position"][:, 1], label='y')
    # ax.plot(missile_hist["time"], missile_hist["position"][:, 2], label='z')
    # ax.set_xlabel('Time (s)')
    # ax.set_ylabel('Position (m)')
    # ax.grid()
    # ax.legend()

    # Missile dynamic pressure vs. time
    ax = axes[0, 0]
    ax.plot(missile_hist["time"], missile_hist["dynamic_pressure"])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Dynamic Pressure (Pa)')
    ax.grid()

    # Missile velocity vs. time
    ax = axes[0, 1]
    ax.plot(missile_hist["time"], missile_hist["velocity"][:, 0], label='vx')
    ax.plot(missile_hist["time"], missile_hist["velocity"][:, 1], label='vy')
    ax.plot(missile_hist["time"], missile_hist["velocity"][:, 2], label='vz')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Velocity (m/s)')
    ax.grid()
    ax.legend()

    # Relative range vs. time
    ax = axes[0, 2]
    range_to_target = np.linalg.norm(target_hist["position"] - missile_hist["position"], axis=1)
    ax.plot(missile_hist["time"], range_to_target)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Range-to-target (m)')
    ax.grid()

    # Missile speed vs. time
    ax = axes[1, 0]
    ax.plot(missile_hist["time"], np.linalg.norm(missile_hist["velocity"], axis=1))
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Speed (m/s)')
    ax.grid()

    # Missile lateral g-load vs. time
    structural_g_limit = 35.0
    g = 9.81
    ax = axes[1, 1]
    ax.plot(missile_hist["time"], np.linalg.norm(missile_hist["a_lat_desired"], axis=1) / g, label='Desired')
    ax.plot(missile_hist["time"], np.linalg.norm(missile_hist["a_lat_achieved"], axis=1) / g, label='Achieved')
    # ax.plot(missile_hist["time"], missile_hist["a_lat_desired"][:, 0] / g, label='Desired Lat Accel X')
    # ax.plot(missile_hist["time"], missile_hist["a_lat_desired"][:, 1] / g, label='Desired Lat Accel Y')
    # ax.plot(missile_hist["time"], missile_hist["a_lat_desired"][:, 2] / g, label='Desired Lat Accel Z')
    # ax.plot(missile_hist["time"], missile_hist["a_lat_achieved"][:, 0] / g, label='Achieved Lat Accel X')
    # ax.plot(missile_hist["time"], missile_hist["a_lat_achieved"][:, 1] / g, label='Achieved Lat Accel Y')
    # ax.plot(missile_hist["time"], missile_hist["a_lat_achieved"][:, 2] / g, label='Achieved Lat Accel Z')
    ax.axhline(y=structural_g_limit, color='red', linestyle='--', label=f'Structural limit ({structural_g_limit} G)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Lateral G-Load (G)')
    ax.grid()
    ax.legend()

    # Missile thrust vs. time
    ax = axes[1, 2]
    ax.plot(missile_hist["time"], np.linalg.norm(missile_hist["thrust"], axis=1))
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Thrust (N)')
    ax.grid()

    # Missile mass vs. time
    ax = axes[2, 0]
    ax.plot(missile_hist["time"], missile_hist["mass"])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Mass (kg)')
    ax.grid()

    # Missile orientation vs. time
    ax = axes[2, 1]
    ax.plot(missile_hist["time"], missile_hist["rpy_deg"][:, 0], label='Roll')
    ax.plot(missile_hist["time"], missile_hist["rpy_deg"][:, 1], label='Pitch')
    ax.plot(missile_hist["time"], missile_hist["rpy_deg"][:, 2], label='Yaw')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Orientation (deg)')
    ax.grid()
    ax.legend()

    # Missile angular velocity vs. time
    ax = axes[2, 2]
    ax.plot(missile_hist["time"], missile_hist["angular_velocity"][:, 0], label='wx')
    ax.plot(missile_hist["time"], missile_hist["angular_velocity"][:, 1], label='wy')
    ax.plot(missile_hist["time"], missile_hist["angular_velocity"][:, 2], label='wz')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Angular Velocity (rad/s)')
    ax.grid()
    ax.legend()

    # Missile alpha and beta vs. time
    ax = axes[3, 0]
    ax.plot(missile_hist["time"], np.degrees(missile_hist["alpha"]), label='$\\alpha$')
    ax.plot(missile_hist["time"], np.degrees(missile_hist["beta"]), label='$\\beta$')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Angle (deg)')
    ax.grid()
    ax.legend()

    # Missile control surface deflections vs. time
    ax = axes[3, 1]
    ax.plot(missile_hist["time"], np.degrees(missile_hist["control_deltas"][:, 0]), label='Aileron')
    ax.plot(missile_hist["time"], np.degrees(missile_hist["control_deltas"][:, 1]), label='Elevator')
    ax.plot(missile_hist["time"], np.degrees(missile_hist["control_deltas"][:, 2]), label='Rudder')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Control Deflection (deg)')
    ax.grid()
    ax.legend()

    # Missile aerodynamic drag, side and lift forces in wind frame vs. time
    ax = axes[3, 2]
    ax.plot(missile_hist["time"], missile_hist["Fw_aero"][:, 0], label='Drag (Wind Frame)')
    ax.plot(missile_hist["time"], missile_hist["Fw_aero"][:, 1], label='Side Force (Wind Frame)')
    ax.plot(missile_hist["time"], missile_hist["Fw_aero"][:, 2], label='Lift (Wind Frame)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Aerodynamic Force (N)')
    ax.grid()
    ax.legend()

    plt.show()
    # Save the figure as a PNG file
    fig.savefig('media/missile_interception_metrics.png', dpi=300)

def animate_trajectories(missile_hist, target_hist):
    """Animates the missile and target trajectories and overlays interception telemetry info."""

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Static axis scaling (creates a 1:1:1 cubic aspect ratio)
    all_data = np.concatenate((np.array(missile_hist["position"]), np.array(target_hist["position"])), axis=0)
    max_range = np.array([
        all_data[:,0].max() - all_data[:,0].min(),
        all_data[:,1].max() - all_data[:,1].min(),
        all_data[:,2].max() - all_data[:,2].min()
    ]).max() / 2.0

    mid_x = (all_data[:,0].max() + all_data[:,0].min()) * 0.5
    mid_y = (all_data[:,1].max() + all_data[:,1].min()) * 0.5
    mid_z = (all_data[:,2].max() + all_data[:,2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    # Floor the z-axis near ground level
    ax.set_zlim(max(0, mid_z - max_range), max(0, mid_z - max_range) + max_range * 2)

    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_zlabel('Altitude Z (m)')
    ax.set_title('Surface-To-Air Interception', weight='bold')

    # Find the index where the flight phase switches to COAST
    flight_phases = missile_hist["flight_phase"]
    transition_idx = len(flight_phases)
    for i, flight_phase in enumerate(flight_phases):
        if flight_phase == "COAST":
            transition_idx = i
            break

    # Initialize drawing objects
    target_line, = ax.plot([], [], [], color='red', label='Target')
    target_vel_line, = ax.plot([], [], [], color='black')
    target_pt = ax.plot([], [], [], marker='o', color='red')[0]

    # Split missile trajectory into two lines based on flight phase
    missile_line_boost, = ax.plot([], [], [], color='orange', linewidth=2, label='Missile (Boost)')
    missile_line_coast, = ax.plot([], [], [], color='blue', linewidth=2, label='Missile (Coast)')
    missile_vel_line, = ax.plot([], [], [], color='black')
    missile_pt = ax.plot([], [], [], marker='o', color='orange')[0]
    a_lat_desired_line = ax.plot([], [], [], color='magenta', label='Desired Lateral Accel (Guidance)')[0]
    a_lat_achieved_line = ax.plot([], [], [], color='cyan', label='Achieved Lateral Accel (Controller)')[0]

    los_line = ax.plot([], [], [], color='black', linestyle='--')[0]

    # Interception telemetry text box
    telemetry_text = fig.text(
        0.02, 0.88, "",
        fontsize=11,
        fontfamily='monospace',
        verticalalignment='top',
        bbox=dict(facecolor='white')
    )

    ax.legend(loc="upper right")

    # Frame downsampling (~250 frames rendered for smooth playback)
    total_steps = len(target_hist["position"])
    frame_skip = max(1, total_steps // 250)
    frames = list(range(0, total_steps, frame_skip))
    if frames[-1] != total_steps - 1:
        frames.append(total_steps - 1)

    def update(frame_idx):
        target_pos = target_hist["position"]
        target_vel = target_hist["velocity"]

        target_line.set_data(target_pos[:frame_idx, 0], target_pos[:frame_idx, 1])
        target_line.set_3d_properties(target_pos[:frame_idx, 2])
        target_pt.set_data([target_pos[frame_idx, 0]], [target_pos[frame_idx, 1]])
        target_pt.set_3d_properties([target_pos[frame_idx, 2]])

        missile_pos = missile_hist["position"]
        missile_vel_body = missile_hist["velocity"]
        missile_orientation = missile_hist["orientation"]

        if frame_idx <= transition_idx:
            # Draw only the boost line
            missile_line_boost.set_data(missile_pos[:frame_idx, 0], missile_pos[:frame_idx, 1])
            missile_line_boost.set_3d_properties(missile_pos[:frame_idx, 2])

            missile_line_coast.set_data([], [])
            missile_line_coast.set_3d_properties([])

            missile_pt.set_color('orange') # Dot is orange during boost phase
        else:
            # Lock the boost line up to the transition point
            missile_line_boost.set_data(missile_pos[:transition_idx+1, 0], missile_pos[:transition_idx+1, 1])
            missile_line_boost.set_3d_properties(missile_pos[:transition_idx+1, 2])

            # Draw the coast line from the transition point to current frame
            missile_line_coast.set_data(missile_pos[transition_idx:frame_idx, 0], missile_pos[transition_idx:frame_idx, 1])
            missile_line_coast.set_3d_properties(missile_pos[transition_idx:frame_idx, 2])

            missile_pt.set_color('blue') # Dot turns blue during coast phase

        missile_pt.set_data([missile_pos[frame_idx, 0]], [missile_pos[frame_idx, 1]])
        missile_pt.set_3d_properties([missile_pos[frame_idx, 2]])

        # Update line of sight (LOS)
        los_line.set_data([target_hist["position"][frame_idx, 0], missile_pos[frame_idx, 0]],
                          [target_hist["position"][frame_idx, 1], missile_pos[frame_idx, 1]])
        los_line.set_3d_properties([target_hist["position"][frame_idx, 2], missile_pos[frame_idx, 2]])

        # Update velocity vectors (scaled for visibility)
        v_scale = 3.0
        target_vel_line.set_data([target_pos[frame_idx, 0], target_pos[frame_idx, 0] + target_vel[frame_idx, 0]*v_scale],
                                 [target_pos[frame_idx, 1], target_pos[frame_idx, 1] + target_vel[frame_idx, 1]*v_scale])
        target_vel_line.set_3d_properties([target_pos[frame_idx, 2], target_pos[frame_idx, 2] + target_vel[frame_idx, 2]*v_scale])

        q = missile_orientation[frame_idx]
        R_ib = utils.quaternion_to_rotation_matrix(q)
        missile_vel = R_ib @ missile_vel_body[frame_idx]
        missile_vel_line.set_data([missile_pos[frame_idx, 0], missile_pos[frame_idx, 0] + missile_vel[0]*v_scale], [missile_pos[frame_idx, 1], missile_pos[frame_idx, 1] + missile_vel[1]*v_scale])
        missile_vel_line.set_3d_properties([missile_pos[frame_idx, 2], missile_pos[frame_idx, 2] + missile_vel[2]*v_scale])

        # Update lateral acceleration vectors (scaled for visibility)
        a_scale = 50.0
        a_lat_desired = missile_hist["a_lat_desired"][frame_idx]
        a_lat_desired_line.set_data([missile_pos[frame_idx, 0], missile_pos[frame_idx, 0] + a_lat_desired[0]*a_scale], [missile_pos[frame_idx, 1], missile_pos[frame_idx, 1] + a_lat_desired[1]*a_scale])
        a_lat_desired_line.set_3d_properties([missile_pos[frame_idx, 2], missile_pos[frame_idx, 2] + a_lat_desired[2]*a_scale])
        a_lat_achieved = missile_hist["a_lat_achieved"][frame_idx]
        a_lat_achieved_line.set_data([missile_pos[frame_idx, 0], missile_pos[frame_idx, 0] + a_lat_achieved[0]*a_scale], [missile_pos[frame_idx, 1], missile_pos[frame_idx, 1] + a_lat_achieved[1]*a_scale])
        a_lat_achieved_line.set_3d_properties([missile_pos[frame_idx, 2], missile_pos[frame_idx, 2] + a_lat_achieved[2]*a_scale])

        # Update interception telemetry
        time = missile_hist["time"][frame_idx]
        flight_phase = missile_hist["flight_phase"][frame_idx]
        speed = np.linalg.norm(missile_vel_body[frame_idx])
        mass = missile_hist["mass"][frame_idx]
        dist = np.linalg.norm(target_hist["position"][frame_idx] - missile_hist["position"][frame_idx])
        g = 9.81
        lateral_g = np.linalg.norm(missile_hist["a_lat_achieved"][frame_idx]) / g
        speed_of_sound = 343.0 # m/s at sea level
        speed_mach = speed / speed_of_sound

        interception_info = (r"$\bf{Interceptor\ Missile\ Telemetry}$" "\n"
                             f"Time:         {time:.1f} s\n"
                             f"Flight Phase: {flight_phase}\n"
                             f"Speed:        Mach {speed_mach:.1f} ({speed:.0f} m/s)\n"
                             f"Mass:         {mass:.1f} kg\n"
                             f"Dist:         {dist:.1f} m\n"
                             f"Lateral G:    {lateral_g:.1f} G\n"
                             f"Alpha:        {np.degrees(missile_hist['alpha'][frame_idx]):.1f} deg\n"
                             f"Beta:         {np.degrees(missile_hist['beta'][frame_idx]):.1f} deg\n")


        telemetry_text.set_text(interception_info)

        return target_line, missile_line_boost, missile_line_coast, target_pt, missile_pt, los_line, target_vel_line, missile_vel_line, a_lat_desired_line, a_lat_achieved_line, telemetry_text

    anim = animation.FuncAnimation(fig, update, frames=frames, interval=10, blit=False, repeat=False)
    # Save animation as a GIF file
    anim.save('media/missile_interception_animation.gif', writer='pillow', fps=15)

    plt.show()

if __name__ == "__main__":
    missile_hist, target_hist, intercepted = run_simulation()
    plot_metrics(missile_hist, target_hist)
    animate_trajectories(missile_hist, target_hist)