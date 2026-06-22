import numpy as np

import utils
from missile import Missile
from target import Target
from visualization import SimulationVisualizer

def log_data(missile_log: dict, target_log: dict, missile: Missile, target: Target, t: float):
    """Logs current missile and target states into the provided log dictionaries."""

    R_ib = utils.quaternion_to_rotation_matrix(missile.orientation())
    R_bi = R_ib.T
    alpha = missile.alpha(missile.velocity(), missile.vi_wind, R_bi)
    beta = missile.beta(missile.velocity(), missile.vi_wind, R_bi)
    Fb_aero, Fw_aero = missile.compute_aerodynamic_forces(missile.position()[2], missile.velocity(), missile.vi_wind, R_bi, missile.virtual_control_deltas)
    rho = utils.compute_air_density(missile.position()[2], missile.rho0, missile.H_scale)
    vb_rel = missile.velocity() - R_bi @ missile.vi_wind
    vb_rel_mag = np.linalg.norm(vb_rel)
    P_dyn = 0.5 * rho * vb_rel_mag**2

    missile_log["time"].append(t)
    missile_log["position"].append(missile.position())
    missile_log["orientation"].append(missile.orientation())
    missile_log["rpy_deg"].append(utils.quaternion_to_rpy_deg(missile.orientation()))
    missile_log["velocity"].append(missile.velocity())
    missile_log["angular_velocity"].append(missile.angular_velocity())
    missile_log["mass"].append(missile.mass())
    missile_log["flight_phase"].append(missile.current_flight_phase())
    missile_log["a_lat_desired"].append(missile.desired_lateral_accel())
    missile_log["a_lat_achieved"].append(missile.achieved_lateral_accel())
    missile_log["thrust"].append(missile.thrust(missile.mass()))
    missile_log["alpha"].append(alpha)
    missile_log["beta"].append(beta)
    missile_log["virtual_control_deltas"].append(missile.virtual_control_deltas)
    missile_log["fin_deflections"].append(missile.fin_deflections.copy())
    missile_log["Fb_aero"].append(Fb_aero)
    missile_log["Fw_aero"].append(Fw_aero)
    missile_log["dynamic_pressure"].append(P_dyn)

    target_log["time"].append(t),
    target_log["position"].append(target.position()),
    target_log["velocity"].append(target.velocity())

def early_termination(missile: Missile, target: Target, visualizer: SimulationVisualizer, t: float) -> bool:
    """Checks various early termination conditions for simulation."""

    if missile.detonate_warhead(target.position()):
        visualizer.update(t, missile, target)
        print(f"Proximity detonation. Target destroyed at {t:.2f} s.")
        return True
    if target.position()[2] < 0.0:
        visualizer.update(t, missile, target)
        print(f"Target impacted the ground at {t:.2f} s.")
        return True
    if missile.position()[2] < 0.0:
        visualizer.update(t, missile, target)
        print(f"Missile impacted the ground at {t:.2f} s.")
        return True
    return False

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

def run_simulation(missile: Missile, target: Target, range_close:float, dt_far: float, dt_close: float, t_max: float, record: bool = False):
    """Runs a missile interception simulation and collects data for analysis and visualization."""

    missile_log = {
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
        "virtual_control_deltas": [],
        "fin_deflections": [],
        "Fb_aero": [],
        "Fw_aero": [],
        "dynamic_pressure": []
    }

    target_log = {
        "time": [],
        "position": [],
        "velocity": []
    }

    visualizer = SimulationVisualizer(missile, target, record)
    render_frame_skip = 5

    # Guidance and control simulation loop
    t = 0.0
    i = 0
    print("Simulating missile interception...")
    while t < t_max:
        rel_range = np.linalg.norm(target.position() - missile.position())

        # Use smaller time steps when missile is close to target for better interception accuracy
        if rel_range < range_close:
            dt = dt_close
        else:
            dt = dt_far

        missile.update_guidance(target)
        missile.update_control(dt)

        log_data(missile_log, target_log, missile, target, t)

        # Update live visualization data
        if i % render_frame_skip == 0:
            visualizer.update(t, missile, target)

        # Stop simulation early if termination conditions are met
        if early_termination(missile, target, visualizer, t):
            break

        update_sim_states(missile, target, dt)
        t += dt
        i += 1

    print(f"Time: {t:.2f} s, "
          f"Missile Mass: {missile.mass():.1f} kg, "
          f"Speed: {missile.speed():.1f} m/s, "
          f"Target Range: {rel_range:.1f} m, "
          f"Flight Phase: {missile.current_flight_phase()}, "
          f"Lateral G: {np.linalg.norm(missile_log['a_lat_achieved'][-1]) / missile.g0:.2f} G, "
          f"Alpha: {np.rad2deg(missile_log['alpha'][-1]):.1f} deg, "
          f"Beta: {np.rad2deg(missile_log['beta'][-1]):.1f} deg")

    visualizer.finalize()

    return missile_log, target_log
