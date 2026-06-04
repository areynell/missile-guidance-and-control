from enum import IntEnum
import numpy as np

from guidance import Guidance
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

        # Missile guidance law with specified maximum lateral acceleration and navigation constant (N) for pure proportional navigation
        self.guidance = Guidance(self.max_lat_accel, N=4.0)

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
        R_bw = utils.wind_to_body_rotation_matrix(alpha, beta) # Wind -> body
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

    def update_guidance(self, target):
        vb = self.velocity()
        q = self.orientation()
        R_ib = utils.quaternion_to_rotation_matrix(q)  # Body to inertial frame rotation matrix
        vi = R_ib @ vb # Velocity in inertial frame

        # TODO: Does PN guidance need to take inertial wind velocity into account?

        self.a_lat_desired = self.guidance.pure_pn_guidance(self.position(), vi, target.position(), target.velocity())
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