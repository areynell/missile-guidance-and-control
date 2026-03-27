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

        self.phase = "STANDBY"
        self.N = 4.0

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

        self.last_a_lat_cmd = np.zeros(3, dtype=float)

    def position(self):
        return self.state[MissileState.X:MissileState.Z+1]

    def velocity(self):
        return self.state[MissileState.VX:MissileState.VZ+1]

    def speed(self):
        return np.linalg.norm(self.velocity())

    def mass(self):
        return self.state[MissileState.M]

    def current_phase(self):
        if self.speed() < 1.0e-6:
            return "STANDBY"
        elif self.mass() > self.m_dry:
            return "BOOST"
        else:
            return "COAST"

    def lateral_g(self):
        return np.linalg.norm(self.last_a_lat_cmd) / self.g0

    def pure_pronav_cmd(self, missile_pos, missile_vel, target_pos, target_vel, N=4.0):
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

        return a_lat

    def limit_lateral_accel(self, a_lat):
        """Limits the lateral acceleration to the missile's structural G-limits."""

        lat_accel_mag = np.linalg.norm(a_lat)
        if lat_accel_mag > self.max_lat_accel:
            a_lat = a_lat * (self.max_lat_accel / lat_accel_mag)
        return a_lat

    def dynamics(self,  missile_state, target_pos, target_vel):
        """Missile dynamics with aerodynamic drag, thrust, gravity, and pure PN guidance."""

        p = missile_state[MissileState.X:MissileState.Z+1]
        v = missile_state[MissileState.VX:MissileState.VZ+1]
        m = missile_state[MissileState.M]

        v_mag = np.linalg.norm(v)
        thrust = self.T if m > self.m_dry else 0.0

        # Drag model
        h = max(p[MissileState.Z], 0.0) # Altitude, clamped to zero
        rho = self.rho0 * np.exp(-h / self.H_scale)
        Fd = 0.5 * rho * self.Cd * self.Aref * v_mag * v

        # Thrust force along missile's velocity vector
        if v_mag < 1.0e-6:
            v_dir = self.v_dir_init
        else:
            v_dir = v / v_mag
        Ft = thrust * v_dir

        # Lateral steering from pure PN guidance law
        a_lat = self.pure_pronav_cmd(p, v, target_pos, target_vel, self.N)
        a_lat = self.limit_lateral_accel(a_lat)
        self.last_a_lat_cmd = a_lat

        # Constant gravity in inertial frame
        a_grav = np.array([0.0, 0.0, -self.g0])

        # State derivatives
        dpdt = v
        dvdt = Ft / m - Fd / m + a_lat + a_grav
        dmdt = -thrust / (self.Isp * self.g0)

        return np.hstack((dpdt, dvdt, dmdt))

class Target:
    """Represents a hostile target with its state and simple 3D evasive maneuvering."""

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

    joint_state = np.hstack((missile.state, target.state))
    next_state = rk4_update(combined_dynamics, joint_state, dt, missile, target)

    # TODO: Remove magic numbers by having missile and target classes provide state size and indexing information
    missile.state = next_state[:7]
    target.state = next_state[7:]
    # missile.phase = "BOOST" if missile.mass() > missile.m_dry else "COAST"

def combined_dynamics(state: np.ndarray, missile: Missile, target: Target) -> np.ndarray:
    """Combines the missile and target dynamics into a single state derivative vector."""

    # TODO: Remove magic numbers by having missile and target classes provide state size and indexing information
    missile_state = state[:7]
    target_state = state[7:]

    target_pos = target_state[TargetState.X:TargetState.Z+1]
    target_vel = target_state[TargetState.VX:TargetState.VZ+1]

    missile_dynamics = missile.dynamics(missile_state, target_pos, target_vel)
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
    t_max = 250.0

    # MIM-104 Patriot parameters (PAC-2 variant)
    missile_params = {
        'T': 70.0e3, # Newtons
        # 'T': 200.0e3, # Newtons
        'Isp': 240.0, # seconds
        'Cd': 0.6,
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
        'z': np.random.uniform(5000.0, 10000.0),
        'vx': target_velocity_xy[0] + np.random.uniform(-100.0, 100.0),
        'vy': target_velocity_xy[1] + np.random.uniform(-100.0, 100.0),
        'vz':  np.random.uniform(-50.0, 50.0),
        'wx': np.random.uniform(-0.02, 0.02),
        'wy': np.random.uniform(-0.02, 0.02),
        'wz': np.random.uniform(-0.02, 0.02),
    }

    target = Target(target_initial_state)

    # Data collection
    missile_history = []
    target_history = []
    info_history = []
    missile_vel_history = []
    target_vel_history = []
    intercepted = False

    print("Simulating missile interception...")

    while t < t_max:
        print(f"Time: {t:.2f} s, Missile Mass: {missile.mass():.1f} kg, Speed: {missile.speed():.1f} m/s, Distance to Target: {np.linalg.norm(target.position() - missile.position()):.1f} m, Phase: {missile.current_phase()}, Lateral G: {missile.lateral_g():.1f} G")

        target_history.append(target.position().copy())
        missile_history.append(missile.position().copy())
        target_vel_history.append(target.velocity().copy())
        missile_vel_history.append(missile.velocity().copy())

        rel_pos = target.position() - missile.position()
        rel_range = np.linalg.norm(rel_pos)

        if rel_range < missile.kill_radius:
            print(f"Proximity detonation! Target destroyed at {t:.2f} s. Distance: {rel_range:.1f} m")
            intercepted = True
            info_history.append((t, missile.mass(), missile.speed(), missile.current_phase(), rel_range, missile.lateral_g()))
            break
        if target.position()[TargetState.Z] < 0.0:
            print(f"Target impacted the ground at {t:.2f} s. Distance to missile: {rel_range:.1f} m")
            intercepted = True
            info_history.append((t, missile.mass(), missile.speed(), missile.current_phase(), rel_range, missile.lateral_g()))
            break
        if missile.position()[MissileState.Z] < 0.0:
            print(f"Missile impacted the ground at {t:.2f} s. Distance to target: {rel_range:.1f} m")
            intercepted = False
            info_history.append((t, missile.mass(), missile.speed(), missile.current_phase(), rel_range, missile.lateral_g()))
            break

        info_history.append((t, missile.mass(), missile.speed(), missile.current_phase(), rel_range, missile.lateral_g()))

        update_sim_states(missile, target, dt)
        t += dt

    if not intercepted:
        print("Target evaded interception.")

    return np.array(target_history), np.array(missile_history), np.array(target_vel_history), np.array(missile_vel_history), info_history, intercepted

def plot_metrics(info_history, intercepted):
    # TODO: Add plots for states vs. time
    # TODO: Add a plot for the missile's lateral G-load over time to visualize maneuvering demands
    pass

def animate_trajectories(target_pos_history, missile_pos_history, target_vel_history, missile_vel_history, info_history):
    """Animates the missile and target trajectories and overlays interception telemetry info."""

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Static axis scaling (creates a 1:1:1 cubic aspect ratio)
    all_data = np.concatenate((target_pos_history, missile_pos_history), axis=0)
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
    target_line, = ax.plot([], [], [], color='red', linewidth=1.5, label='Hostile Target')
    target_vel_line, = ax.plot([], [], [], color='black', linewidth=2.0)
    target_pt = ax.plot([], [], [], marker='o', color='red', markersize=6)[0]

    missile_line, = ax.plot([], [], [], color='blue', linewidth=1.5, label='Interceptor Missile')
    missile_vel_line, = ax.plot([], [], [], color='black', linewidth=2.0)
    missile_pt = ax.plot([], [], [], marker='o', color='blue', markersize=6)[0]

    los_line = ax.plot([], [], [], color='black', linestyle='--', linewidth=1.0, alpha=0.7)[0]

    # Interception telemetry text box
    telemetry_text = ax.text2D(0.05, 0.90, "", transform=ax.transAxes, fontsize=11,
                               color='black', fontfamily='monospace', verticalalignment='top',
                               bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))

    ax.legend(loc="upper right")

    # Frame downsampling (~250 frames rendered for smooth playback)
    total_steps = len(target_pos_history)
    # frame_skip = max(1, total_steps // 250)
    frame_skip = 1
    frames = list(range(0, total_steps, frame_skip))
    if frames[-1] != total_steps - 1:
        frames.append(total_steps - 1)

    def update(frame_idx):
        # Update target
        target_line.set_data(target_pos_history[:frame_idx, 0], target_pos_history[:frame_idx, 1])
        target_line.set_3d_properties(target_pos_history[:frame_idx, 2])
        target_pt.set_data([target_pos_history[frame_idx, 0]], [target_pos_history[frame_idx, 1]])
        target_pt.set_3d_properties([target_pos_history[frame_idx, 2]])

        # Update missile
        missile_line.set_data(missile_pos_history[:frame_idx, 0], missile_pos_history[:frame_idx, 1])
        missile_line.set_3d_properties(missile_pos_history[:frame_idx, 2])
        missile_pt.set_data([missile_pos_history[frame_idx, 0]], [missile_pos_history[frame_idx, 1]])
        missile_pt.set_3d_properties([missile_pos_history[frame_idx, 2]])

        # Update line of sight (LOS)
        los_line.set_data([target_pos_history[frame_idx, 0], missile_pos_history[frame_idx, 0]],
                          [target_pos_history[frame_idx, 1], missile_pos_history[frame_idx, 1]])
        los_line.set_3d_properties([target_pos_history[frame_idx, 2], missile_pos_history[frame_idx, 2]])

        # Update velocity vectors (scaled for visibility)
        v_scale = 3.0
        target_pos = target_pos_history[frame_idx]
        target_vel = target_vel_history[frame_idx]
        target_vel_line.set_data([target_pos[0], target_pos[0] + target_vel[0]*v_scale], [target_pos[1], target_pos[1] + target_vel[1]*v_scale])
        target_vel_line.set_3d_properties([target_pos[2], target_pos[2] + target_vel[2]*v_scale])

        missile_pos = missile_pos_history[frame_idx]
        missile_vel = missile_vel_history[frame_idx]
        missile_vel_line.set_data([missile_pos[0], missile_pos[0] + missile_vel[0]*v_scale], [missile_pos[1], missile_pos[1] + missile_vel[1]*v_scale])
        missile_vel_line.set_3d_properties([missile_pos[2], missile_pos[2] + missile_vel[2]*v_scale])

        # Update interception telemetry
        t, mass, speed, phase, dist, lateral_g = info_history[frame_idx]
        speed_mach = speed / 343.0 # Speed of sound at sea level approx

        interception_info = (r"$\bf{Interceptor\ Missile\ Telemetry}$" "\n"
                             f"Time:      {t:04.1f} s\n"
                             f"Phase:     {phase}\n"
                             f"Speed:     Mach {speed_mach:.1f} ({speed:.0f} m/s)\n"
                             f"Mass:      {mass:06.1f} kg\n"
                             f"Dist:      {dist/1000:04.1f} km\n"
                             f"Lateral G: {lateral_g:05.1f} G")

        telemetry_text.set_text(interception_info)

        return target_line, missile_line, target_pt, missile_pt, los_line, target_vel_line, missile_vel_line, telemetry_text

    anim = animation.FuncAnimation(fig, update, frames=frames, interval=30, blit=False, repeat=False)
    plt.show()

if __name__ == "__main__":
    target_hist, missile_hist, target_vel_hist, missile_vel_hist, info_hist, hit = run_simulation()
    animate_trajectories(target_hist, missile_hist, target_vel_hist, missile_vel_hist, info_hist)