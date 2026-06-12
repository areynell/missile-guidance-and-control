import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import utils

def plot_metrics(missile_log, target_log):
    fig, axes = plt.subplots(4, 3, figsize=(16, 8), constrained_layout=True)
    fig.suptitle('Missile Interception Metrics', fontsize=14, weight='bold')

    # Relative range vs. time
    ax = axes[0, 0]
    target_range = np.linalg.norm(target_log["position"] - missile_log["position"], axis=1)
    ax.plot(missile_log["time"], target_range)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Target Range (m)')
    ax.grid()

    # Missile lateral G-load vs. time
    structural_g_limit = 35.0
    g = 9.81
    ax = axes[0, 1]
    ax.plot(missile_log["time"], np.linalg.norm(missile_log["a_lat_desired"], axis=1) / g, label='Desired')
    ax.plot(missile_log["time"], np.linalg.norm(missile_log["a_lat_achieved"], axis=1) / g, label='Achieved')
    ax.axhline(y=structural_g_limit, color='red', linestyle='--', label=f'Structural limit ({structural_g_limit} G)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Lateral G-Load (G)')
    ax.grid()
    ax.legend()

     # Missile aerodynamic forces in wind frame vs. time
    ax = axes[0, 2]
    ax.plot(missile_log["time"], missile_log["Fw_aero"][:, 0], label='Drag (Wind Frame)')
    ax.plot(missile_log["time"], missile_log["Fw_aero"][:, 1], label='Side Force (Wind Frame)')
    ax.plot(missile_log["time"], missile_log["Fw_aero"][:, 2], label='Lift (Wind Frame)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Aerodynamic Force (N)')
    ax.grid()
    ax.legend()

    # Missile speed vs. time
    ax = axes[1, 0]
    ax.plot(missile_log["time"], np.linalg.norm(missile_log["velocity"], axis=1))
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Speed (m/s)')
    ax.grid()

    # Missile body velocity vs. time
    ax = axes[1, 1]
    ax.plot(missile_log["time"], missile_log["velocity"][:, 0], label='vx')
    ax.plot(missile_log["time"], missile_log["velocity"][:, 1], label='vy')
    ax.plot(missile_log["time"], missile_log["velocity"][:, 2], label='vz')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Body Velocity (m/s)')
    ax.grid()
    ax.legend()

    # Missile dynamic pressure vs. time
    ax = axes[1, 2]
    ax.plot(missile_log["time"], missile_log["dynamic_pressure"])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Dynamic Pressure (Pa)')
    ax.grid()

    # Missile orientation vs. time
    ax = axes[2, 0]
    ax.plot(missile_log["time"], missile_log["rpy_deg"][:, 0], label='Roll')
    ax.plot(missile_log["time"], missile_log["rpy_deg"][:, 1], label='Pitch')
    ax.plot(missile_log["time"], missile_log["rpy_deg"][:, 2], label='Yaw')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Orientation (deg)')
    ax.grid()
    ax.legend()

    # Missile alpha and beta vs. time
    ax = axes[2, 1]
    ax.plot(missile_log["time"], np.degrees(missile_log["alpha"]), label='$\\alpha$')
    ax.plot(missile_log["time"], np.degrees(missile_log["beta"]), label='$\\beta$')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Angle (deg)')
    ax.grid()
    ax.legend()

    # Missile angular velocity vs. time
    ax = axes[2, 2]
    ax.plot(missile_log["time"], missile_log["angular_velocity"][:, 0], label='wx')
    ax.plot(missile_log["time"], missile_log["angular_velocity"][:, 1], label='wy')
    ax.plot(missile_log["time"], missile_log["angular_velocity"][:, 2], label='wz')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Angular Velocity (rad/s)')
    ax.grid()
    ax.legend()

    # Missile control surface deflections vs. time
    ax = axes[3, 0]
    ax.plot(missile_log["time"], np.degrees(missile_log["control_deltas"][:, 0]), label='Aileron')
    ax.plot(missile_log["time"], np.degrees(missile_log["control_deltas"][:, 1]), label='Elevator')
    ax.plot(missile_log["time"], np.degrees(missile_log["control_deltas"][:, 2]), label='Rudder')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Control Deflection (deg)')
    ax.grid()
    ax.legend()

    # Missile thrust vs. time
    ax = axes[3, 1]
    ax.plot(missile_log["time"], np.linalg.norm(missile_log["thrust"], axis=1))
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Thrust (N)')
    ax.grid()

    # Missile mass vs. time
    ax = axes[3, 2]
    ax.plot(missile_log["time"], missile_log["mass"])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Mass (kg)')
    ax.grid()

    plt.show()
    # # Save the figure as a PNG file
    # fig.savefig('media/missile_interception_metrics.png', dpi=300)

def animate_trajectories(missile_log, target_log):
    """Animates the missile and target trajectories and overlays interception telemetry info."""

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Static axis scaling (creates a 1:1:1 cubic aspect ratio)
    all_data = np.concatenate((np.array(missile_log["position"]), np.array(target_log["position"])), axis=0)
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

    # Find the index where the flight phase switches to coast
    flight_phases = missile_log["flight_phase"]
    transition_idx = len(flight_phases)
    for i, flight_phase in enumerate(flight_phases):
        if flight_phase == "Coast":
            transition_idx = i
            break

    # Initialize drawing objects
    target_line, = ax.plot([], [], [], color='red', label='Target')
    target_vel_line, = ax.plot([], [], [], color='black')
    target_pt = ax.plot([], [], [], marker='o', color='red')[0]

    # Split missile trajectory into two lines based on flight phase
    missile_line_boost, = ax.plot([], [], [], color='orange', linewidth=2, label='Missile (Boost Phase)')
    missile_line_coast, = ax.plot([], [], [], color='blue', linewidth=2, label='Missile (Coast Phase)')
    missile_vel_line, = ax.plot([], [], [], color='black')
    missile_pt = ax.plot([], [], [], marker='o', color='orange')[0]
    a_lat_desired_line = ax.plot([], [], [], color='magenta', label='Desired Lateral Accel (Guidance)')[0]
    a_lat_achieved_line = ax.plot([], [], [], color='cyan', label='Achieved Lateral Accel (Controller)')[0]

    los_line = ax.plot([], [], [], color='black', linestyle='--')[0]

    # Interception telemetry text box
    telemetry_text = fig.text(
        0.02, 0.87, "",
        fontsize=11,
        fontfamily='monospace',
        verticalalignment='top',
        bbox=dict(facecolor='white')
    )

    ax.legend(loc="upper right")

    # Frame downsampling (~250 frames rendered for smooth playback)
    total_steps = len(target_log["position"])
    frame_skip = max(1, total_steps // 250)
    frames = list(range(0, total_steps, frame_skip))
    if frames[-1] != total_steps - 1:
        frames.append(total_steps - 1)

    def update(frame_idx):
        target_pos = target_log["position"]
        target_vel = target_log["velocity"]

        target_line.set_data(target_pos[:frame_idx, 0], target_pos[:frame_idx, 1])
        target_line.set_3d_properties(target_pos[:frame_idx, 2])
        target_pt.set_data([target_pos[frame_idx, 0]], [target_pos[frame_idx, 1]])
        target_pt.set_3d_properties([target_pos[frame_idx, 2]])

        missile_pos = missile_log["position"]
        missile_vel_body = missile_log["velocity"]
        missile_orientation = missile_log["orientation"]

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
        los_line.set_data([target_log["position"][frame_idx, 0], missile_pos[frame_idx, 0]],
                          [target_log["position"][frame_idx, 1], missile_pos[frame_idx, 1]])
        los_line.set_3d_properties([target_log["position"][frame_idx, 2], missile_pos[frame_idx, 2]])

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
        a_lat_desired = missile_log["a_lat_desired"][frame_idx]
        a_lat_desired_line.set_data([missile_pos[frame_idx, 0], missile_pos[frame_idx, 0] + a_lat_desired[0]*a_scale], [missile_pos[frame_idx, 1], missile_pos[frame_idx, 1] + a_lat_desired[1]*a_scale])
        a_lat_desired_line.set_3d_properties([missile_pos[frame_idx, 2], missile_pos[frame_idx, 2] + a_lat_desired[2]*a_scale])
        a_lat_achieved = missile_log["a_lat_achieved"][frame_idx]
        a_lat_achieved_line.set_data([missile_pos[frame_idx, 0], missile_pos[frame_idx, 0] + a_lat_achieved[0]*a_scale], [missile_pos[frame_idx, 1], missile_pos[frame_idx, 1] + a_lat_achieved[1]*a_scale])
        a_lat_achieved_line.set_3d_properties([missile_pos[frame_idx, 2], missile_pos[frame_idx, 2] + a_lat_achieved[2]*a_scale])

        # Update interception telemetry
        time = missile_log["time"][frame_idx]
        flight_phase = missile_log["flight_phase"][frame_idx]
        speed = np.linalg.norm(missile_vel_body[frame_idx])
        mass = missile_log["mass"][frame_idx]
        dist = np.linalg.norm(target_log["position"][frame_idx] - missile_log["position"][frame_idx])
        g = 9.81
        lateral_g = np.linalg.norm(missile_log["a_lat_achieved"][frame_idx]) / g
        speed_of_sound = 343.0 # m/s at sea level
        speed_mach = speed / speed_of_sound

        interception_info = (r"$\bf{Interceptor\ Missile\ Telemetry}$" "\n"
                             f"Time:         {time:.1f} s\n"
                             f"Flight Phase: {flight_phase}\n"
                             f"Speed:        Mach {speed_mach:.1f} ({speed:.0f} m/s)\n"
                             f"Mass:         {mass:.1f} kg\n"
                             f"Dist:         {dist:.1f} m\n"
                             f"Lateral G:    {lateral_g:.1f} G\n"
                             f"Alpha:        {np.degrees(missile_log['alpha'][frame_idx]):.1f} deg\n"
                             f"Beta:         {np.degrees(missile_log['beta'][frame_idx]):.1f} deg")

        telemetry_text.set_text(interception_info)

        return target_line, missile_line_boost, missile_line_coast, target_pt, missile_pt, los_line, target_vel_line, missile_vel_line, a_lat_desired_line, a_lat_achieved_line, telemetry_text

    anim = animation.FuncAnimation(fig, update, frames=frames, interval=10, blit=False, repeat=False)
    # # Save animation as a GIF file
    # anim.save('media/missile_interception_animation.gif', writer='pillow', fps=15)

    plt.show()

def animate_6dof_missile(missile_log, target_log, length, diameter,
                         vi_wind=np.array([0.0, 0.0, 0.0]),
                         v_scale=0.01, f_scale=0.0005):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Frame downsampling (~250 frames rendered for smooth playback)
    total_steps = len(missile_log["time"])
    frame_skip = max(1, total_steps // 250)
    frames = list(range(0, total_steps, frame_skip))
    if frames[-1] != total_steps - 1:
        frames.append(total_steps - 1)

    # Generate missile surface mesh
    theta = np.linspace(0, 2 * np.pi, 20)
    nose_length = 0.2 * length

    x_cylinder = np.linspace(-length / 2.0, length / 2.0 - nose_length, 10)
    Theta_cylinder, X_cylinder = np.meshgrid(theta, x_cylinder)
    Y_cylinder = (diameter / 2.0) * np.cos(Theta_cylinder)
    Z_cylinder = (diameter / 2.0) * np.sin(Theta_cylinder)
    missile_points_cylinder = np.vstack([X_cylinder.flatten(), Y_cylinder.flatten(), Z_cylinder.flatten()])

    x_cone = np.linspace(length / 2.0 - nose_length, length / 2.0, 10)
    Theta_cone, X_cone = np.meshgrid(theta, x_cone)
    R_cone = (diameter / 2.0) * ((length / 2.0 - X_cone) / nose_length)
    Y_cone = R_cone * np.cos(Theta_cone)
    Z_cone = R_cone * np.sin(Theta_cone)
    missile_points_cone = np.vstack([X_cone.flatten(), Y_cone.flatten(), Z_cone.flatten()])

    # Interception telemetry text box
    telemetry_text = fig.text(
        0.02, 0.87, "",
        fontsize=11,
        fontfamily='monospace',
        verticalalignment='top',
        bbox=dict(facecolor='white')
    )

    def update(frame_idx):
        ax.clear()

        missile_position = missile_log["position"][frame_idx]
        missile_vel_body = missile_log["velocity"]
        missile_orientation = missile_log["orientation"][frame_idx]
        R_ib = utils.quaternion_to_rotation_matrix(missile_orientation)  # Body -> inertial

        target_position = target_log["position"][frame_idx]

        # Missile geometry
        for missile_points, shape, color in [
            (missile_points_cylinder, X_cylinder.shape, 'darkgrey'),
            (missile_points_cone, X_cone.shape, 'dimgrey'),
        ]:
            missile_points_inertial = R_ib @ missile_points
            ax.plot_surface(
                (missile_points_inertial[0].reshape(shape) + missile_position[0]),
                (missile_points_inertial[1].reshape(shape) + missile_position[1]),
                (missile_points_inertial[2].reshape(shape) + missile_position[2]),
                color=color, alpha=0.3, edgecolor='gray', linewidth=0.5)

        # Body axes
        axis_length = length * 0.8
        for label, color, body_axes in [
            ('Body X (Forward)', 'red', np.array([axis_length, 0.0, 0.0])),
            ('Body Y (Left)',    'green', np.array([0.0, axis_length, 0.0])),
            ('Body Z (Up)',      'blue', np.array([0.0, 0.0, axis_length])),
        ]:
            axis_end = missile_position + R_ib @ body_axes
            ax.plot([missile_position[0], axis_end[0]], [missile_position[1], axis_end[1]], [missile_position[2], axis_end[2]],
                    color=color, linewidth=3, label=label)

        # Velocity and wind vectors
        v_body = missile_log["velocity"][frame_idx]
        v_inertial = R_ib @ v_body
        v_norm = np.linalg.norm(v_inertial)
        if v_norm > 1e-3:
            v_end = missile_position + v_inertial * v_scale
            ax.plot([missile_position[0], v_end[0]], [missile_position[1], v_end[1]], [missile_position[2], v_end[2]],
                    color='black', linewidth=2, linestyle='-', label='Velocity')

        if np.linalg.norm(vi_wind) > 1e-3:
            w_end = missile_position + vi_wind * v_scale
            ax.plot([missile_position[0], w_end[0]], [missile_position[1], w_end[1]], [missile_position[2], w_end[2]],
                    color='magenta', linewidth=2, linestyle=':', label='Wind')

        alpha = missile_log["alpha"][frame_idx]
        beta = missile_log["beta"][frame_idx]
        Fw_aero = missile_log["Fw_aero"][frame_idx]
        R_bw = utils.wind_to_body_rotation_matrix(alpha, beta) # Body <- wind

        # Isolate each wind-frame component, rotate to body, then to inertial
        for label, color, Fw_component in [
            ('Drag (Wind Frame)',       'orange', np.array([Fw_aero[0], 0.0,       0.0      ])),
            ('Side Force (Wind Frame)', 'purple', np.array([0.0,        Fw_aero[1], 0.0     ])),
            ('Lift (Wind Frame)',       'yellow', np.array([0.0,        0.0,       Fw_aero[2]])),
        ]:
            Fi = R_ib @ R_bw @ Fw_component # Inertial <- body <- wind
            if np.linalg.norm(Fi) > 1e-3:
                force_end = missile_position + Fi * f_scale
                ax.plot([missile_position[0], force_end[0]], [missile_position[1], force_end[1]], [missile_position[2], force_end[2]],
                        color=color, linewidth=2.5, linestyle='-', label=label)

        target_range = np.linalg.norm(target_position - missile_position)

        # Camera and labels
        window = length * 2.0
        ax.set_xlim(missile_position[0] - window, missile_position[0] + window)
        ax.set_ylim(missile_position[1] - window, missile_position[1] + window)
        ax.set_zlim(missile_position[2] - window, missile_position[2] + window)
        ax.set_xlabel('Inertial X (m)')
        ax.set_ylabel('Inertial Y (m)')
        ax.set_zlabel('Inertial Z (m)')
        ax.set_title('Missile Orientation and Forces', weight='bold')
        ax.set_box_aspect([1, 1, 1])
        ax.legend(loc='upper right')

        # Update interception telemetry
        time = missile_log["time"][frame_idx]
        flight_phase = missile_log["flight_phase"][frame_idx]
        speed = np.linalg.norm(missile_vel_body[frame_idx])
        mass = missile_log["mass"][frame_idx]
        dist = np.linalg.norm(target_log["position"][frame_idx] - missile_log["position"][frame_idx])
        g = 9.81
        lateral_g = np.linalg.norm(missile_log["a_lat_achieved"][frame_idx]) / g
        speed_of_sound = 343.0 # m/s at sea level
        speed_mach = speed / speed_of_sound

        interception_info = (r"$\bf{Interceptor\ Missile\ Telemetry}$" "\n"
                             f"Time:         {time:.1f} s\n"
                             f"Flight Phase: {flight_phase}\n"
                             f"Speed:        Mach {speed_mach:.1f} ({speed:.0f} m/s)\n"
                             f"Mass:         {mass:.1f} kg\n"
                             f"Dist:         {dist:.1f} m\n"
                             f"Lateral G:    {lateral_g:.1f} G\n"
                             f"Alpha:        {np.degrees(missile_log['alpha'][frame_idx]):.1f} deg\n"
                             f"Beta:         {np.degrees(missile_log['beta'][frame_idx]):.1f} deg")

        telemetry_text.set_text(interception_info)

    anim = animation.FuncAnimation(fig, update, frames=frames, interval=30, blit=False, repeat=False)
    # # Save animation as a GIF file
    # anim.save('media/missile_orientation_and_forces_animation.gif', writer='pillow', fps=15)

    plt.show()