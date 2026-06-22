from enum import IntEnum
import numpy as np

from parameters import AtmosphericParams, MissileParams
from guidance import MissileGuidance
from controller import MissileController
from target import Target
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

    def __init__(self, initial_state: np.ndarray, missile_params: MissileParams, atmospheric_params: AtmosphericParams,
                 missile_guidance: MissileGuidance, missile_controller: MissileController):
        self.state = initial_state

        self.g0 = 9.81
        self.Isp = missile_params.propulsion.Isp
        self.T = missile_params.propulsion.thrust

        # Drag coefficients
        self.CD_0 = missile_params.aero.CD_0 # Base drag coefficient at zero angle of attack, based on parasitic/skin friction drag of the missile's body and fins
        self.CD_alpha = missile_params.aero.CD_alpha # Drag static stability derivative (dCD/dAlpha)
        self.CD_delta = missile_params.aero.CD_delta # Drag control derivative (dCD/dDelta)

        # Lift coefficients
        self.CL_0 = missile_params.aero.CL_0 # Base lift coefficient at zero angle of attack, which is typically 0 for a symmetric missile body
        self.CL_alpha = missile_params.aero.CL_alpha # Lift static stability derivative (dCL/dAlpha)
        self.CL_delta = missile_params.aero.CL_delta # Lift control derivative (dCL/dDelta)

        # Side-force coefficients
        self.CY_0 = missile_params.aero.CY_0 # Base side-force coefficient at zero sideslip, which is typically 0 for a symmetric missile body
        self.CY_beta = missile_params.aero.CY_beta # Side-force static stability derivative (dCY/dBeta)
        self.CY_delta = missile_params.aero.CY_delta # Side-force control derivative (dCY/dDelta)

        # Rolling moment coefficients
        self.Cl_0 = missile_params.aero.Cl_0 # Base rolling moment coefficient at zero aileron deflection, which is typically 0 for a symmetric missile body
        self.Cl_p = missile_params.aero.Cl_p # Roll damping dynamic stability derivative (dCl/dp)
        self.Cl_delta = missile_params.aero.Cl_delta # Roll control derivative (dCl/dDelta)

        # Pitching moment coefficients
        self.Cm_0 = missile_params.aero.Cm_0 # Base pitching moment coefficient at zero angle of attack, which is typically 0 for a symmetric missile body
        self.Cm_alpha = missile_params.aero.Cm_alpha # Pitch static stability derivative (dCm/dAlpha)
        self.Cm_q = missile_params.aero.Cm_q # Pitch damping dynamic stability derivative (dCm/dq)
        self.Cm_delta = missile_params.aero.Cm_delta # Pitch control derivative (dCm/dDelta)

        # Yawing moment coefficients
        self.Cn_0 = missile_params.aero.Cn_0 # Base yawing moment coefficient at zero sideslip, which is typically 0 for a symmetric missile body
        self.Cn_beta = missile_params.aero.Cn_beta # Yaw static stability derivative (dCn/dBeta)
        self.Cn_r = missile_params.aero.Cn_r # Yaw damping dynamic stability derivative (dCn/dr)
        self.Cn_delta = missile_params.aero.Cn_delta # Yaw control derivative (dCn/dDelta)

        print("Missile Aerodynamic Force Coefficients: CD_0 = {:.3f}, CD_alpha = {:.3f}, CL_0 = {:.3f}, CL_alpha = {:.3f}, CL_delta = {:.3f}, CY_0 = {:.3f}, CY_beta = {:.3f}, CY_delta = {:.3f}".format(
            self.CD_0, self.CD_alpha, self.CL_0, self.CL_alpha, self.CL_delta, self.CY_0, self.CY_beta, self.CY_delta
        ))

        print("Missile Aerodynamic Moment Coefficients: Cl_0 = {:.3f}, Cl_p = {:.3f}, Cl_delta = {:.3f}, Cm_0 = {:.3f}, Cm_alpha = {:.3f}, Cm_q = {:.3f}, Cm_delta = {:.3f}, Cn_0 = {:.3f}, Cn_beta = {:.3f}, Cn_r = {:.3f}, Cn_delta = {:.3f}".format(
            self.Cl_0, self.Cl_p, self.Cl_delta, self.Cm_0, self.Cm_alpha, self.Cm_q, self.Cm_delta, self.Cn_0, self.Cn_beta, self.Cn_r, self.Cn_delta
        ))

        self.D_ref = missile_params.structural.diameter
        self.A_ref = np.pi * (self.D_ref / 2.0)**2 # Reference area for aerodynamic force calculations, typically the maximum cross-sectional area of the missile
        self.L = missile_params.structural.length
        self.max_lat_accel = missile_params.structural.max_lateral_accel
        self.m_dry = missile_params.structural.dry_mass
        self.kill_radius = missile_params.warhead.kill_radius

        self.rho0 = atmospheric_params.sea_level_density
        self.H_scale = atmospheric_params.scale_height
        self.vi_wind = atmospheric_params.wind_vector

        self.flight_phase = "BOOST"
        self.a_lat_desired = np.zeros(3, dtype=float)

        # Safe minimum velocity threshold for aerodynamic effectiveness
        self.v_min_aero = 1.0

        self.guidance = missile_guidance
        self.controller = missile_controller
        self.virtual_control_deltas = np.zeros(3, dtype=float) # [delta_a, delta_e, delta_r]
        self.fin_deflections = np.zeros(4, dtype=float) # [delta_fin_1, delta_fin_2, delta_fin_3, delta_fin_4]
        self.delta_limit = self.controller.delta_limit # rad, maximum allowable control surface deflection

        # Mixing (M) and unmixing (M_pinv) matrices for X-configured fins
        # Each fin is oriented at 45 degrees to the missile axes
        sin45 = 1.0 / np.sqrt(2.0)
        self.M = np.array([
            [-1.0,  sin45, -sin45],
            [-1.0,  sin45,  sin45],
            [-1.0, -sin45,  sin45],
            [-1.0, -sin45, -sin45]
        ], dtype=float)
        self.M_pinv = np.linalg.pinv(self.M)

        self.prev_rel_range = np.inf

    def position(self) -> np.ndarray:
        """Returns the missile's position vector in the inertial frame."""
        return self.state[MissileState.X:MissileState.Z+1]

    def orientation(self) -> np.ndarray:
        """Returns the missile's orientation as a quaternion (qw, qx, qy, qz) representing the rotation from the body frame to the inertial frame."""
        return self.state[MissileState.QW:MissileState.QZ+1]

    def velocity(self) -> np.ndarray:
        """Returns the missile's velocity vector in the body frame."""
        return self.state[MissileState.VX:MissileState.VZ+1]

    def angular_velocity(self) -> np.ndarray:
        """Returns the missile's angular velocity vector in the body frame."""
        return self.state[MissileState.WX:MissileState.WZ+1]

    def speed(self) -> float:
        """Returns the missile's speed (velocity magnitude)."""
        return np.linalg.norm(self.velocity())

    def mass(self) -> float:
        """Returns the missile's mass."""
        return self.state[MissileState.M]

    def alpha(self, vb_missile: np.ndarray, vi_wind: np.ndarray, R_bi: np.ndarray) -> float:
        """Calculates the missile's angle of attack based on the its velocity vector in the body frame and the wind velocity in the inertial frame."""
        vb_rel = vb_missile - R_bi @ vi_wind
        return np.arctan2(vb_rel[2], vb_rel[0])

    def beta(self, vb_missile: np.ndarray, vi_wind: np.ndarray, R_bi: np.ndarray) -> float:
        """Calculates the missile's sideslip angle based on the its velocity vector in the body frame and the wind velocity in the inertial frame."""
        vb_rel = vb_missile - R_bi @ vi_wind
        vb_rel_mag = np.linalg.norm(vb_rel)
        if vb_rel_mag < self.v_min_aero:
            return 0.0 # Prevent divide-by-zero at launch
        return np.arcsin(np.clip(vb_rel[1] / vb_rel_mag, -1.0, 1.0))

    def update_flight_phase(self):
        """Updates the missile's current flight phase (boost or coast) based on its mass relative to the dry mass after burnout."""
        if self.mass() > self.m_dry:
            self.flight_phase = "Boost"
        else:
            self.flight_phase = "Coast"

    def current_flight_phase(self) -> str:
        """Returns the current flight phase of the missile (boost or coast)."""
        return self.flight_phase

    def thrust(self, mass: float) -> np.ndarray:
        """Returns the thrust vector along the missile's longitudinal axis."""
        if mass > self.m_dry:
            return np.array([self.T, 0.0, 0.0], dtype=float)
        else:
            return np.array([0.0, 0.0, 0.0], dtype=float)

    def compute_aerodynamic_forces(self, altitude: float, vb_missile: np.ndarray, vi_wind: np.ndarray, R_bi: np.ndarray, virtual_control_deltas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
        rho = utils.compute_air_density(altitude, self.rho0, self.H_scale)
        P_dyn = 0.5 * rho * vb_rel_mag**2

        # Roll (aileron), pitch (elevator) and yaw (rudder) control surface deflections
        delta_a, delta_e, delta_r = virtual_control_deltas

        # Calculate aerodynamic force coefficients in the wind (relative velocity) frame
        alpha_total = np.arccos(np.cos(alpha) * np.cos(beta))
        CD = self.CD_0 + (self.CD_alpha * alpha_total**2) + (self.CD_delta * (delta_e**2 + delta_r**2))
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

    def compute_aerodynamic_moments(self, altitude: float, vb_missile: np.ndarray, vi_wind: np.ndarray, R_bi: np.ndarray, wb: np.ndarray, virtual_control_deltas: np.ndarray) -> np.ndarray:
        """Calculates the aerodynamic moments acting on the missile in the body frame."""

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
        rho = utils.compute_air_density(altitude, self.rho0, self.H_scale)
        P_dyn = 0.5 * rho * vb_rel_mag**2

        # Angular velocity components in body frame
        wx, wy, wz = wb

        # Roll (aileron), pitch (elevator) and yaw (rudder) control surface deflections
        delta_a, delta_e, delta_r = virtual_control_deltas

        # Calculate aerodynamic moment coefficients
        Cl = self.Cl_0 + (self.Cl_p * wx * self.D_ref / (2.0 * vb_rel_mag)) + (self.Cl_delta * delta_a)
        Cm = self.Cm_0 + (self.Cm_alpha * alpha) + (self.Cm_q * wy * self.D_ref / (2.0 * vb_rel_mag)) + (self.Cm_delta * delta_e)
        Cn = self.Cn_0 + (self.Cn_beta * beta) + (self.Cn_r * wz * self.D_ref / (2.0 * vb_rel_mag)) + (self.Cn_delta * delta_r)

        # Calculate aerodynamic moments in body frame
        # NOTE: It's standard for A_ref (reference area) and D_ref (reference length) to be constants.
        # For missiles, A_ref is typically the maximum cross-sectional area and D_ref is typically the missile diameter.
        Mx = P_dyn*self.A_ref*self.D_ref*Cl
        My = P_dyn*self.A_ref*self.D_ref*Cm
        Mz = P_dyn*self.A_ref*self.D_ref*Cn

        return np.array([Mx, My, Mz], dtype=float)

    def detonate_warhead(self, target_pos: np.ndarray) -> bool:
        """
        Checks if the missile's warhead should detonate based on reaching the point of closest approach
        while being within the lethal kill radius.
        """

        rel_pos = target_pos - self.position()
        rel_range = np.linalg.norm(rel_pos)

        # Trigger if missile is within kill radius AND has passed the point of closest approach
        if self.prev_rel_range < self.kill_radius and rel_range > self.prev_rel_range:
            return True

        self.prev_rel_range = rel_range
        return False

    def desired_lateral_accel(self) -> np.ndarray:
        """Returns the guidance law's desired lateral acceleration in the inertial frame."""
        return self.a_lat_desired

    def achieved_lateral_accel(self) -> np.ndarray:
        """Computes the achieved lateral acceleration executed by the flight controller in the inertial frame, excluding thrust (axial) contribution."""

        # TODO: Check that these calculations are correct. Pure PN gives desired lateral acceleration relative to the missile's velocity vector, whereas
        # the achieved acceleration calculation here is based on the missile's body frame which is not necessarily aligned with the velocity vector

        p = self.position()
        vb = self.velocity()
        q = self.orientation()
        m = self.mass()

        R_ib = utils.quaternion_to_rotation_matrix(q)
        R_bi = R_ib.T

        Fb_aero, _ = self.compute_aerodynamic_forces(p[2], vb, self.vi_wind, R_bi, self.virtual_control_deltas)

        # Total specific force in body frame
        a_body = Fb_aero / m

        # Zero out the axial (thrust) direction — body X is the missile's longitudinal axis
        a_lat_body = np.array([0.0, a_body[1], a_body[2]], dtype=float)

        # Rotate lateral acceleration to inertial frame for comparison with desired lateral acceleration from guidance law
        return R_ib @ a_lat_body

    def update_guidance(self, target: Target) -> np.ndarray:
        """Updates the missile's desired lateral acceleration based on the guidance law."""

        vb = self.velocity()
        q = self.orientation()
        R_ib = utils.quaternion_to_rotation_matrix(q)  # Body to inertial frame rotation matrix
        vi = R_ib @ vb # Velocity in inertial frame

        # TODO: Does PN guidance need to take inertial wind velocity into account?

        self.a_lat_desired = self.guidance.compute_guidance(self.position(), vi, target.position(), target.velocity())
        self.update_flight_phase()

    def update_control(self, dt: float):
        """Updates the missile's control surface deflections based on the control law to achieve the desired lateral acceleration from the guidance law."""

        p = self.position()
        vb = self.velocity()
        q = self.orientation()
        wb = self.angular_velocity()
        m = self.mass()

        R_ib = utils.quaternion_to_rotation_matrix(q) # Body to inertial frame rotation matrix
        R_bi = R_ib.T

        a_cmd_body = R_bi @ self.a_lat_desired

        # Compute missile's current lateral acceleration
        Fb_aero, _ = self.compute_aerodynamic_forces(p[2], vb, self.vi_wind, R_bi, self.virtual_control_deltas)
        a_body = Fb_aero / m # Lateral acceleration experienced by the missile

        # Dynamic pressure
        vb_rel = vb - R_bi @ self.vi_wind
        vb_rel_mag = np.linalg.norm(vb_rel)
        rho = utils.compute_air_density(p[2], self.rho0, self.H_scale)
        P_dyn = 0.5 * rho * vb_rel_mag**2

        # Skid-to-turn (STT) missiles typically command zero roll angle
        roll_cmd_rad = 0.0
        # NOTE: For feedforward control, we're using CL and CY in the wind frame, but should actually be using CN and CY in the body frame. This appoximation only works for low alpha and beta angles
        # TODO: Think about using a struct/dataclass to pass arguments more cleanly and safely into update() function
        virtual_control_deltas = self.controller.update(roll_cmd_rad, a_cmd_body, a_body, wb, q, P_dyn, self.mass(), self.A_ref, self.CL_delta, self.CL_alpha, self.Cm_delta, self.Cm_alpha, self.CY_delta, self.CY_beta, self.Cn_delta, self.Cn_beta, dt)
        self.virtual_control_deltas, self.fin_deflections = self.apply_control_mixing_and_limits(virtual_control_deltas)

    def apply_control_mixing_and_limits(self, virtual_control_deltas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        1) Maps virtual controls to physical fin deflections
        2) Applies proportional position limiting to prevent cross-coupling
        3) Maps limited fin deflections back to saturated virtual controls for use in force and moment calculations.
        """

        # Map virtual control deltas (delta_a, delta_e, delta_r) to individual fin control deltas (delta_fin_1, delta_fin_2, delta_fin_3, delta_fin_4)
        fin_control_deltas = self.M @ virtual_control_deltas

        # Apply proportional scaling to enforce actuator limits without introducing cross-coupling.
        #
        # Naive clipping (np.clip) is avoided here because clipping the largest fin independently causes
        # the others to remain unchanged, which distorts the original mix and introduces spurious moments
        # in unintended axes after saturation.
        #
        # Proportional scaling preserves the relative ratios of all fin deflections by finding the
        # most-deflected fin and scaling the entire fin deflection vector down uniformly so that the
        # largest fin just reaches the limit. This maintains the correct direction of the commanded
        # moment vector even under saturation.
        max_deflection = np.max(np.abs(fin_control_deltas))
        if max_deflection > self.delta_limit:
            scaling_factor = self.delta_limit / max_deflection
            fin_control_deltas = scaling_factor * fin_control_deltas

        # Map saturated fin control deltas (delta_fin_1_sat, delta_fin_2_sat, delta_fin_3_sat, delta_fin_4_sat) back to saturated virtual control deltas
        virtual_control_deltas = self.M_pinv @ fin_control_deltas

        return virtual_control_deltas, fin_control_deltas

    def dynamics(self, missile_state: np.ndarray) -> np.ndarray:
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
        Fb_aero, _ = self.compute_aerodynamic_forces(p[2], vb, self.vi_wind, R_bi, self.virtual_control_deltas)

        # Total force in missile's body frame
        Fb_total = Fb_grav + Fb_thrust + Fb_aero

        # Aerodynamic moments (roll, pitch, yaw) in body frame
        Mb_aero = self.compute_aerodynamic_moments(p[2], vb, self.vi_wind, R_bi, wb, self.virtual_control_deltas)

        # Total moment/torque in missile's body frame
        Mb_total = Mb_aero

        # State derivatives based on Newton-Euler equations for a rigid body
        dpdt = R_ib @ vb
        dqdt = 0.5 * utils.quaternion_multiply(q, np.hstack((0.0, wb)))
        dvbdt = Fb_total / m - np.cross(wb, vb)
        dwbdt = I_inv @ (Mb_total - np.cross(wb, I @ wb))
        dmdt = -Fb_thrust[0] / (self.Isp * self.g0)

        return np.hstack((dpdt, dqdt, dvbdt, dwbdt, dmdt))