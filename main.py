import numpy as np

from parameters import AerodynamicParams, MissileParams, PropulsionParams, StructuralParams, AtmosphericParams, WarheadParams, ControllerParams, GuidanceParams
from missile import MissileState, Missile
from guidance import MissileGuidance
from controller import MissileController
from target import TargetState, Target
from simulation import run_simulation
from visualization import plot_metrics

def initialize_missile() -> Missile:
    """Initializes missile with parameters, initial state, guidance, and control."""

    missile_params = MissileParams(
        aero = AerodynamicParams(
            CD_0 = 0.1,
            CD_alpha = 5.0,
            CD_delta = 1.0,
            CY_0 = 0.0,
            CY_beta = -20.0,
            CY_delta = 2.0,
            CL_0 = 0.0,
            CL_alpha = 20.0,
            CL_delta = 2.0,
            Cl_0 = 0.0,
            Cl_p = -2.0,
            Cl_delta = 0.5,
            Cm_0 = 0.0,
            Cm_alpha = -2.0,
            Cm_q = -2.0,
            Cm_delta = 2.0,
            Cn_0 = 0.0,
            Cn_beta = 2.0,
            Cn_r = -2.0,
            Cn_delta = 2.0
        ),
        propulsion = PropulsionParams(
            thrust = 150.0e3,
            Isp = 260,
        ),
        structural = StructuralParams(
            diameter = 0.41,
            length = 5.0,
            dry_mass = 550.0,
            total_mass = 900.0,
            max_lateral_accel = 30 * 9.81
        ),
        warhead = WarheadParams(
            kill_radius = 30.0
        )
    )

    atmospheric_params = AtmosphericParams(
        sea_level_density = 1.225,
        scale_height = 8500.0,
        wind_vector = np.zeros(3)
    )

    guidance_params = GuidanceParams(
        N = 4.0,
        max_lateral_accel = missile_params.structural.max_lateral_accel
    )

    missile_guidance = MissileGuidance(guidance_params)

    # Missile flight controller with appropriate gains for roll, pitch, and yaw control
    v_ref = 500.0
    controller_params = ControllerParams(
        # Roll control gains
        Kp_roll = 0.5,
        Kp_roll_rate = 0.02,
        Ki_roll_rate = 0.01,

        # Pitch control gains
        Kdc_pitch = 1.0,
        Ka_pitch_rate = 0.1,
        Ki_pitch_rate = 0.15,
        Kr_pitch_rate = 0.25,

        # Yaw control gains
        Kdc_yaw = 1.0,
        Ka_yaw_rate = 0.1,
        Ki_yaw_rate = 0.15,
        Kr_yaw_rate = 0.25,

        # Additional parameters for gain scheduling and anti-windup
        v_ref = v_ref,
        P_dyn_ref = 0.5 * atmospheric_params.sea_level_density * v_ref**2,
        P_dyn_min = 100.0,
        integral_limit = 2.0,
        delta_limit = np.deg2rad(45.0)
    )

    missile_controller = MissileController(controller_params)

    # Initial pitch of 45 degrees
    initial_pitch = -np.deg2rad(45.0)
    qw_init = np.cos(initial_pitch / 2.0)
    qy_init = np.sin(initial_pitch / 2.0)

    initial_state = np.empty(len(MissileState), dtype=float)
    initial_state[MissileState.X] = 0.0
    initial_state[MissileState.Y] = 0.0
    initial_state[MissileState.Z] = 0.0
    initial_state[MissileState.QW] = qw_init
    initial_state[MissileState.QX] = 0.0
    initial_state[MissileState.QY] = qy_init
    initial_state[MissileState.QZ] = 0.0
    initial_state[MissileState.VX] = 0.0
    initial_state[MissileState.VY] = 0.0
    initial_state[MissileState.VZ] = 0.0
    initial_state[MissileState.WX] = 0.0
    initial_state[MissileState.WY] = 0.0
    initial_state[MissileState.WZ] = 0.0
    initial_state[MissileState.M] = missile_params.structural.total_mass

    return Missile(initial_state, missile_params, atmospheric_params, missile_guidance, missile_controller)

def initialize_target() -> Target:
    """Initializes target with initial state."""

    # NOTE: Using seed for reproducibility of random target initial states during development and testing. Remove or modify seed for more varied scenarios.
    np.random.seed(7)
    target_position_xy = np.array([
        np.random.uniform(10000.0, 20000.0),
        np.random.uniform(10000.0, 20000.0),
    ], dtype=float)

    # Making target's initial velocity vector point towards the xy origin
    target_speed = np.random.uniform(300.0, 800.0)
    direction_to_origin_xy = -target_position_xy / np.linalg.norm(target_position_xy)
    target_velocity_xy = target_speed * direction_to_origin_xy

    initial_state = np.empty(len(TargetState), dtype=float)
    initial_state[TargetState.X] = target_position_xy[0]
    initial_state[TargetState.Y] = target_position_xy[1]
    initial_state[TargetState.Z] = np.random.uniform(10000.0, 20000.0)
    initial_state[TargetState.VX] = target_velocity_xy[0] + np.random.uniform(-100.0, 100.0)
    initial_state[TargetState.VY] = target_velocity_xy[1] + np.random.uniform(-100.0, 100.0)
    initial_state[TargetState.VZ] =  np.random.uniform(-50.0, 50.0)
    initial_state[TargetState.WX] = np.random.uniform(-0.1, 0.1)
    initial_state[TargetState.WY] = np.random.uniform(-0.1, 0.1)
    initial_state[TargetState.WZ] = np.random.uniform(-0.1, 0.1)

    print("Target Initial State:")
    print(initial_state)

    return Target(initial_state)

def main():
    missile = initialize_missile()
    target = initialize_target()

    range_close = 1000.0
    dt_far = 0.01
    dt_close = 0.5*dt_far
    t_max = 100.0
    missile_log, target_log = run_simulation(missile, target, range_close, dt_far, dt_close, t_max)

    print("Generating post-flight interception metrics...")
    plot_metrics(missile_log, target_log)

if __name__ == "__main__":
    main()