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
              f"Target Range: {np.linalg.norm(target.position() - missile.position()):.1f} m, "
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
