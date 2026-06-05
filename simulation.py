import numpy as np

import utils
from missile import Missile
from target import Target

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

# def run_simulation(missile_params: MissileParams, atmospheric_params: AtmosphericParams, missile_initial_state: dict, target_initial_state: dict):
def run_simulation(missile: Missile, target: Target, dt: float, t_max: float):
    """Runs a missile interception simulation and collects data for analysis and visualization."""

    t = 0.0
    dt_far = dt
    dt_close = 0.1*dt
    range_close = 1000.0

    # Data collection
    intercepted = False

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
        "control_deltas": [],
        "Fb_aero": [],
        "Fw_aero": [],
        "dynamic_pressure": []
    }

    target_log = {
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
              f"Target Range: {np.linalg.norm(target.position() - missile.position()):.1f} m, "
              f"Flight Phase: {missile.current_flight_phase()}, "
              f"Lateral G: {np.linalg.norm(missile.achieved_lateral_accel()) / missile.g0:.1f} G, "
              f"Alpha: {np.rad2deg(alpha):.1f} deg, "
              f"Beta: {np.rad2deg(beta):.1f} deg")

        missile_log["time"].append(t)
        missile_log["position"].append(missile.position().copy())
        missile_log["orientation"].append(missile.orientation().copy())
        missile_log["rpy_deg"].append(utils.quaternion_to_rpy_deg(missile.orientation()).copy())
        missile_log["velocity"].append(missile.velocity().copy())
        missile_log["angular_velocity"].append(missile.angular_velocity().copy())
        missile_log["mass"].append(missile.mass())
        missile_log["flight_phase"].append(missile.current_flight_phase())
        missile_log["a_lat_desired"].append(missile.desired_lateral_accel().copy())
        missile_log["a_lat_achieved"].append(missile.achieved_lateral_accel().copy())
        missile_log["thrust"].append(missile.thrust(missile.mass()))
        missile_log["alpha"].append(alpha)
        missile_log["beta"].append(beta)
        missile_log["control_deltas"].append(missile.control_deltas.copy())
        Fb_aero, Fw_aero = missile.compute_aerodynamic_forces(missile.position()[2], missile.velocity(), missile.vi_wind, R_bi, missile.control_deltas)
        missile_log["Fb_aero"].append(Fb_aero)
        missile_log["Fw_aero"].append(Fw_aero)
        missile_log["dynamic_pressure"].append(0.5 * missile.rho0 * np.exp(-missile.position()[2] / missile.H_scale) * np.linalg.norm(missile.velocity() - R_bi @ missile.vi_wind)**2)

        target_log["time"].append(t)
        target_log["position"].append(target.position().copy())
        target_log["velocity"].append(target.velocity().copy())

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
    missile_log["time"] = np.array(missile_log["time"])
    missile_log["position"] = np.array(missile_log["position"])
    missile_log["orientation"] = np.array(missile_log["orientation"])
    missile_log["rpy_deg"] = np.array(missile_log["rpy_deg"])
    missile_log["velocity"] = np.array(missile_log["velocity"])
    missile_log["angular_velocity"] = np.array(missile_log["angular_velocity"])
    missile_log["mass"] = np.array(missile_log["mass"])
    missile_log["a_lat_desired"] = np.array(missile_log["a_lat_desired"])
    missile_log["a_lat_achieved"] = np.array(missile_log["a_lat_achieved"])
    missile_log["thrust"] = np.array(missile_log["thrust"])
    missile_log["alpha"] = np.array(missile_log["alpha"])
    missile_log["beta"] = np.array(missile_log["beta"])
    missile_log["control_deltas"] = np.array(missile_log["control_deltas"])
    missile_log["Fb_aero"] = np.array(missile_log["Fb_aero"])
    missile_log["Fw_aero"] = np.array(missile_log["Fw_aero"])
    missile_log["dynamic_pressure"] = np.array(missile_log["dynamic_pressure"])
    target_log["time"] = np.array(target_log["time"])
    target_log["position"] = np.array(target_log["position"])
    target_log["velocity"] = np.array(target_log["velocity"])

    return missile_log, target_log, intercepted
