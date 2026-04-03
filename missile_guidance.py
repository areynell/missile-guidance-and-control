from enum import IntEnum

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class MissileState(IntEnum):
    X = 0
    Y = 1
    Z = 2
    VX = 3
    VY = 4
    VZ = 5
    M = 6

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
            initial_state['vx'],
            initial_state['vy'],
            initial_state['vz'],
            initial_state['m']
        ], dtype=float)

        self.g0 = 9.81
        self.Isp = missile_params['Isp']
        self.T = missile_params['T']
        self.Cd = missile_params['Cd']
        self.Aref = missile_params['Aref']
        self.max_lat_accel = missile_params['max_lat_accel']
        self.m_dry = missile_params['m_dry']
        self.kill_radius = missile_params['kill_radius']

        self.rho0 = atmospheric_params['rho0']
        self.H_scale = atmospheric_params['H_scale']

        azimuth_init = np.deg2rad(45.0)
        elevation_init = np.deg2rad(65.0)
        self.v_dir_init = np.array([
            np.cos(elevation_init) * np.cos(azimuth_init),
            np.cos(elevation_init) * np.sin(azimuth_init),
            np.sin(elevation_init)
        ])

        self.phase = "BOOST"
        self.a_lat_cmd = np.zeros(3, dtype=float)

    def position(self):
        return self.state[MissileState.X:MissileState.Z+1]

    def velocity(self):
        return self.state[MissileState.VX:MissileState.VZ+1]

    def speed(self):
        return np.linalg.norm(self.velocity())

    def mass(self):
        return self.state[MissileState.M]

    def update_phase(self):
        if self.mass() > self.m_dry:
            self.phase = "BOOST"
        else:
            self.phase = "COAST"

    def current_phase(self):
        return self.phase

    def drag(self, v_mag, h):
        h = max(h, 0.0) # Altitude, clamped to zero
        rho = self.rho0 * np.exp(-h / self.H_scale)
        return 0.5 * rho * self.Cd * self.Aref * v_mag**2

    def lateral_accel_cmd(self):
        return self.a_lat_cmd

    def pure_pn_cmd(self, missile_pos, missile_vel, target_pos, target_vel, N=4.0):
        """ Calculates the lateral acceleration command using Pure Proportional Navigation (PPN)."""

        rel_pos = target_pos - missile_pos
        rel_vel = target_vel - missile_vel
        rel_range = np.linalg.norm(rel_pos)

        if rel_range < 1e-6:
            return np.zeros(3)  # Avoid division by zero if target is extremely close

        los_dir = rel_pos / rel_range
        los_rate = np.cross(los_dir, rel_vel) / rel_range

        # Pure PN commands acceleration perpendicular to the missile's velocity vector
        a_lat = N * np.cross(los_rate, missile_vel)

        # TODO: Add a feedforward term to counter-act gravity with additional lateral acceleration?

        return self.limit_lateral_accel(a_lat)

    def limit_lateral_accel(self, a_lat):
        """Limits the lateral acceleration to the missile's structural G-limits."""

        lat_accel_mag = np.linalg.norm(a_lat)
        if lat_accel_mag > self.max_lat_accel:
            a_lat = a_lat * (self.max_lat_accel / lat_accel_mag)
        return a_lat

    def update_guidance(self, target):
        """Updates the missile's lateral acceleration command based on the current target state."""
        a_lat_cmd = self.pure_pn_cmd(self.position(),
                                     self.velocity(),
                                     target.position(),
                                     target.velocity())
        self.a_lat_cmd = a_lat_cmd

        self.update_phase()

    def dynamics(self, missile_state):
        """Missile dynamics with aerodynamic drag, thrust, gravity, and pure PN guidance."""

        p = missile_state[MissileState.X:MissileState.Z+1]
        v = missile_state[MissileState.VX:MissileState.VZ+1]
        m = missile_state[MissileState.M]

        v_mag = np.linalg.norm(v)
        thrust = self.T if m > self.m_dry else 0.0

        # Thrust force along missile's velocity vector
        if v_mag < 1.0e-6:
            v_dir = self.v_dir_init
        else:
            v_dir = v / v_mag
        Ft = thrust * v_dir

         # Drag model
        h = max(p[2], 0.0) # Altitude, clamped to zero
        rho = self.rho0 * np.exp(-h / self.H_scale)
        Fd = 0.5 * rho * self.Cd * self.Aref * v_mag * v

        # Constant gravity in inertial frame
        a_grav = np.array([0.0, 0.0, -self.g0], dtype=float)

        # State derivatives
        dpdt = v
        dvdt = Ft / m - Fd / m + a_grav + self.a_lat_cmd
        dmdt = -thrust / (self.Isp * self.g0)

        return np.hstack((dpdt, dvdt, dmdt))

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

    def dynamics(self, target_state):
        v = target_state[TargetState.VX:TargetState.VZ+1]
        w = target_state[TargetState.WX:TargetState.WZ+1]

        dpdt = v
        dvdt = np.cross(w, v)
        dwdt = np.zeros(3, dtype=float) # No change in angular velocity for simplicity
        return np.hstack((dpdt, dvdt, dwdt))

def update_sim_states(missile: Missile, target: Target, dt: float):
    """Update missile and target jointly with one RK4 step."""

    # Combine missile and target states into a single vector for joint integration
    joint_state = np.hstack((missile.state, target.state))

    # Propagate the combined state forward by one time step using RK4 integration
    updated_state = rk4_update(combined_dynamics, joint_state, dt, missile, target)

    # Update missile and target states from the updated combined state vector
    missile.state = updated_state[:missile.NUM_STATES]
    target.state = updated_state[missile.NUM_STATES:missile.NUM_STATES + target.NUM_STATES]

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
    dt = 0.1
    dt_far = dt
    dt_close = 0.1*dt
    range_close = 1000.0
    t_max = 250.0

    # Parameters that ROUGHLY approximate MIM-104 Patriot PAC-2 variant
    missile_params = {
        'T': 150.0e3, # Newtons
        # 'T': 70.0e3, # Newtons
        'Isp': 260.0, # seconds
        'Cd': 0.3,
        'Aref': 0.132,
        'max_lat_accel': 35.0 * 9.81,
        'm_total': 900.0, # kg
        'm_dry': 550.0, # kg
        'kill_radius': 30.0 # meters
    }

    missile_initial_state = {
        'x': 0.0,
        'y': 0.0,
        'z': 0.0,
        'vx': 0.0,
        'vy': 0.0,
        'vz': 0.0,
        'm': missile_params['m_total']
    }

    atmospheric_params = {
        'rho0': 1.225, # kg/m^3 at sea level
        'H_scale': 8500.0 # Scale height for exponential atmosphere (meters)
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
        "velocity": [],
        "mass": [],
        "phase": [],
        "a_lat_cmd": [],
        "drag": []
    }

    target_history = {
        "time": [],
        "position": [],
        "velocity": []
    }

    print("Simulating missile interception...")

    # Guidance and simulation loop
    while t < t_max:
        # Computing lateral acceleration command at start of each guidance cycle based on current missile and target states
        missile.update_guidance(target)

        assert abs(np.dot(missile.lateral_accel_cmd(), missile.velocity())) < 1.0e-6, "Lateral acceleration command is not perpendicular to velocity vector!"

        print(f"Time: {t:.2f} s, "
              f"Missile Mass: {missile.mass():.1f} kg, "
              f"Speed: {missile.speed():.1f} m/s, "
              f"Distance to Target: {np.linalg.norm(target.position() - missile.position()):.1f} m, "
              f"Phase: {missile.current_phase()}, "
              f"Lateral G: {np.linalg.norm(missile.lateral_accel_cmd()) / missile.g0:.1f} G")

        missile_history["time"].append(t)
        missile_history["position"].append(missile.position().copy())
        missile_history["velocity"].append(missile.velocity().copy())
        missile_history["mass"].append(missile.mass())
        missile_history["phase"].append(missile.current_phase())
        missile_history["a_lat_cmd"].append(missile.lateral_accel_cmd().copy())
        missile_history["drag"].append(missile.drag(missile.speed(), missile.position()[2]))

        target_history["time"].append(t)
        target_history["position"].append(target.position().copy())
        target_history["velocity"].append(target.velocity().copy())

        rel_range = np.linalg.norm(target.position() - missile.position())
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
        # TODO: If missile almost hits target, but then misses, break out of simulation
        # TODO: Logic is if missile gets within 1.5x kill radius, but then goes back out to 2.0x kill radius, end simulation and count as miss

        # Use smaller time steps when missile is close to target for better interception accuracy
        if rel_range < range_close:
            dt = dt_close
        else:
            dt = dt_far

        update_sim_states(missile, target, dt)
        t += dt

    if not intercepted:
        print("Target evaded interception.")

    # Convert dict of lists to dict of numpy arrays for easier plotting later
    missile_history["time"] = np.array(missile_history["time"])
    missile_history["position"] = np.array(missile_history["position"])
    missile_history["velocity"] = np.array(missile_history["velocity"])
    missile_history["mass"] = np.array(missile_history["mass"])
    missile_history["a_lat_cmd"] = np.array(missile_history["a_lat_cmd"])
    missile_history["drag"] = np.array(missile_history["drag"])

    target_history["time"] = np.array(target_history["time"])
    target_history["position"] = np.array(target_history["position"])
    target_history["velocity"] = np.array(target_history["velocity"])

    return missile_history, target_history, intercepted

def plot_metrics(missile_hist, target_hist):
    fig = plt.figure(figsize=(15, 10), constrained_layout=True)
    fig.suptitle('Missile Interception Metrics', fontsize=14, weight='bold')

    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 1], hspace=0.08, wspace=0.08)

    # Relative range vs. time
    ax = fig.add_subplot(gs[0, 0:2])
    range_to_target = np.linalg.norm(target_hist["position"] - missile_hist["position"], axis=1)
    ax.plot(missile_hist["time"], range_to_target, label='Missile range to target (m)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Range (m)')
    ax.set_title('Missile Range to Target vs. Time')
    ax.grid()
    ax.legend()

    # Missile speed vs. time
    ax = fig.add_subplot(gs[0, 2:4])
    ax.plot(missile_hist["time"], np.linalg.norm(missile_hist["velocity"], axis=1), label='Missile speed (m/s)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Speed (m/s)')
    ax.set_title('Missile Speed vs. Time')
    ax.grid()
    ax.legend()

    # Missile lateral g-load vs. time
    structural_g_limit = 35.0
    g = 9.81
    ax = fig.add_subplot(gs[1, 0:2])
    ax.plot(missile_hist["time"], np.linalg.norm(missile_hist["a_lat_cmd"], axis=1) / g, label='Missile lateral G-load (G)')
    ax.axhline(y=structural_g_limit, color='red', linestyle='--', label=f'Missile structural limit ({structural_g_limit} G)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Lateral G-Load (G)')
    ax.set_title('Missile Lateral G-Load vs. Time')
    ax.grid()
    ax.legend()

    # Missile mass vs. time
    ax = fig.add_subplot(gs[1, 2:4])
    ax.plot(missile_hist["time"], missile_hist["mass"], label='Missile mass (kg)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Mass (kg)')
    ax.set_title('Missile Mass vs. Time')
    ax.grid()
    ax.legend()

    # Missile drag vs. time (centered using middle 2 of 4 columns)
    ax = fig.add_subplot(gs[2, 1:3])
    ax.plot(missile_hist["time"], missile_hist["drag"], label='Missile drag force (N)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Drag Force (N)')
    ax.set_title('Missile Drag Force vs. Time')
    ax.grid()
    ax.legend(loc='upper right')

    plt.show()
    # # Save the figure as a PNG file
    # fig.savefig('media/missile_interception_metrics.png', dpi=300)

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

    # Initialize drawing objects
    target_line, = ax.plot([], [], [], color='red', label='Target')
    target_vel_line, = ax.plot([], [], [], color='black')
    target_pt = ax.plot([], [], [], marker='o', color='red')[0]

    missile_line, = ax.plot([], [], [], color='blue', label='Interceptor Missile')
    missile_vel_line, = ax.plot([], [], [], color='black')
    missile_pt = ax.plot([], [], [], marker='o', color='blue')[0]
    a_lat_cmd_line = ax.plot([], [], [], color='magenta', label='PN Lateral Accel')[0]

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
        # Update target
        target_line.set_data(target_hist["position"][:frame_idx, 0], target_hist["position"][:frame_idx, 1])
        target_line.set_3d_properties(target_hist["position"][:frame_idx, 2])
        target_pt.set_data([target_hist["position"][frame_idx, 0]], [target_hist["position"][frame_idx, 1]])
        target_pt.set_3d_properties([target_hist["position"][frame_idx, 2]])

        # Update missile
        missile_line.set_data(missile_hist["position"][:frame_idx, 0], missile_hist["position"][:frame_idx, 1])
        missile_line.set_3d_properties(missile_hist["position"][:frame_idx, 2])
        missile_pt.set_data([missile_hist["position"][frame_idx, 0]], [missile_hist["position"][frame_idx, 1]])
        missile_pt.set_3d_properties([missile_hist["position"][frame_idx, 2]])

        # Update line of sight (LOS)
        los_line.set_data([target_hist["position"][frame_idx, 0], missile_hist["position"][frame_idx, 0]],
                          [target_hist["position"][frame_idx, 1], missile_hist["position"][frame_idx, 1]])
        los_line.set_3d_properties([target_hist["position"][frame_idx, 2], missile_hist["position"][frame_idx, 2]])

        # Update velocity vectors (scaled for visibility)
        v_scale = 3.0
        target_pos = target_hist["position"][frame_idx]
        target_vel = target_hist["velocity"][frame_idx]
        target_vel_line.set_data([target_pos[0], target_pos[0] + target_vel[0]*v_scale], [target_pos[1], target_pos[1] + target_vel[1]*v_scale])
        target_vel_line.set_3d_properties([target_pos[2], target_pos[2] + target_vel[2]*v_scale])

        missile_pos = missile_hist["position"][frame_idx]
        missile_vel = missile_hist["velocity"][frame_idx]
        missile_vel_line.set_data([missile_pos[0], missile_pos[0] + missile_vel[0]*v_scale], [missile_pos[1], missile_pos[1] + missile_vel[1]*v_scale])
        missile_vel_line.set_3d_properties([missile_pos[2], missile_pos[2] + missile_vel[2]*v_scale])

        # Update pure PN lateral acceleration vector (scaled for visibility)
        a_lat_cmd = missile_hist["a_lat_cmd"][frame_idx]
        a_scale = 50.0
        a_lat_cmd_line.set_data([missile_pos[0], missile_pos[0] + a_lat_cmd[0]*a_scale], [missile_pos[1], missile_pos[1] + a_lat_cmd[1]*a_scale])
        a_lat_cmd_line.set_3d_properties([missile_pos[2], missile_pos[2] + a_lat_cmd[2]*a_scale])

        # Update interception telemetry
        time = missile_hist["time"][frame_idx]
        phase = missile_hist["phase"][frame_idx]
        speed = np.linalg.norm(missile_hist["velocity"][frame_idx])
        mass = missile_hist["mass"][frame_idx]
        dist = np.linalg.norm(target_hist["position"][frame_idx] - missile_hist["position"][frame_idx])
        g = 9.81
        lateral_g = np.linalg.norm(missile_hist["a_lat_cmd"][frame_idx]) / g
        speed_of_sound = 343.0 # m/s at sea level
        speed_mach = speed / speed_of_sound

        interception_info = (r"$\bf{Interceptor\ Missile\ Telemetry}$" "\n"
                             f"Time:      {time:.1f} s\n"
                             f"Phase:     {phase}\n"
                             f"Speed:     Mach {speed_mach:.1f} ({speed:.0f} m/s)\n"
                             f"Mass:      {mass:.1f} kg\n"
                             f"Dist:      {dist:.1f} m\n"
                             f"Lateral G: {lateral_g:.1f} G")

        telemetry_text.set_text(interception_info)

        return target_line, missile_line, target_pt, missile_pt, los_line, target_vel_line, missile_vel_line, a_lat_cmd_line, telemetry_text

    anim = animation.FuncAnimation(fig, update, frames=frames, interval=10, blit=False, repeat=False)
    # # Save animation as a GIF file
    # anim.save('media/missile_interception_animation.gif', writer='pillow', fps=15)

    plt.show()

if __name__ == "__main__":
    missile_hist, target_hist, intercepted = run_simulation()
    plot_metrics(missile_hist, target_hist)
    animate_trajectories(missile_hist, target_hist)