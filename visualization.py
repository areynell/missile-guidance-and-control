import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtGui
from PIL import Image
import os

import utils


class SimulationVisualizer:
    """Encapsulates pyqtgraph 3D visualization for live simulation monitoring."""

    def __init__(self, missile, target, record=False):
        self.app = pg.mkQApp("Missile Interception Simulation")

        # Main window setup
        self.main_window = pg.QtWidgets.QWidget()
        self.main_window.setWindowTitle('Missile Interception - Live View')

        self.record = record
        self.frames = []

        self.view_elevation = 20.0
        self.view_azimuth = 0.0
        self.azimuth_rotation_rate = 0.5  # degrees per update cycle

        # Creating two windows, a "far" trajectory view and a "close" missile orientation view
        self.view_far = gl.GLViewWidget()
        self.view_far.setCameraPosition(distance=30000, elevation=self.view_elevation, azimuth=self.view_azimuth)
        self.view_close = gl.GLViewWidget()
        self.view_close.setCameraPosition(distance=25, elevation=self.view_elevation, azimuth=self.view_azimuth)

        self.layout = pg.QtWidgets.QHBoxLayout(self.main_window)
        self.layout.addWidget(self.view_far)
        self.layout.addWidget(self.view_close)

        screen = QtGui.QGuiApplication.primaryScreen().geometry()
        self.main_window.resize(int(screen.width() * 0.8), int(screen.height() * 0.6))
        self.main_window.show()

        self.items_far = self._setup_view_items(self.view_far, missile, target)
        self.items_close = self._setup_view_items(self.view_close, missile, target)

        self._setup_overlays()

        self.force_scale = 0.001
        self.acceleration_scale = 0.1

        # Pre-allocate buffers for trajectory visualization
        self.buffer_capacity = 1000
        self.missile_positions = np.zeros((self.buffer_capacity, 3))
        self.target_positions = np.zeros((self.buffer_capacity, 3))
        self.point_count = 0
        self.flight_phase_change_idx = None

    def update(self, t, missile, target):
        """Main update loop called by the simulation to refresh the 3D scene."""

        missile_position = missile.position()
        missile_velocity = missile.velocity()

        target_position = target.position()
        target_velocity = target.velocity()

        rel_range = np.linalg.norm(target_position - missile_position)

        # Grow buffers if they reach capacity
        if self.point_count >= self.buffer_capacity:
            self.buffer_capacity *= 2
            self.missile_positions.resize((self.buffer_capacity, 3), refcheck=False)
            self.target_positions.resize((self.buffer_capacity, 3), refcheck=False)

        # Insert new data at the current pointer
        self.missile_positions[self.point_count] = missile_position
        self.target_positions[self.point_count] = target_position
        self.point_count += 1

        # Detect the transition point from boost to coast phase
        if self.flight_phase_change_idx is None and missile.current_flight_phase() == "Coast":
            self.flight_phase_change_idx = self.point_count - 1

        # Processing and plotting trajectories
        missile_trajectory = self.missile_positions[:self.point_count]
        target_trajectory = self.target_positions[:self.point_count]
        los_points = np.linspace(missile_position, target_position, max(4, int(rel_range/500)*2))

        if self.flight_phase_change_idx is not None:
            boost_points = missile_trajectory[:self.flight_phase_change_idx + 1]
            coast_points = missile_trajectory[self.flight_phase_change_idx:]
        else:
            boost_points = missile_trajectory
            coast_points = None

        for view_items in [self.items_far, self.items_close]:
            view_items["missile_boost_trajectory"].setData(pos=boost_points)
            view_items["missile_coast_trajectory"].setData(pos=coast_points)
            view_items["target_trajectory"].setData(pos=target_trajectory)
            view_items["los_line"].setData(pos=los_points, mode='lines')

        # Orientations and state calculations
        R_ib = utils.quaternion_to_rotation_matrix(missile.orientation())
        R_bi = R_ib.T
        alpha = missile.alpha(missile.velocity(), missile.vi_wind, R_bi)
        beta = missile.beta(missile.velocity(), missile.vi_wind, R_bi)
        R_iw = R_ib @ utils.wind_to_body_rotation_matrix(alpha, beta)

        _, Fw_aero = missile.compute_aerodynamic_forces(missile_position[2], missile_velocity, missile.vi_wind, R_bi, missile.virtual_control_deltas)
        a_lat_desired = missile.desired_lateral_accel()
        a_lat_achieved = missile.achieved_lateral_accel()

        self._update_telemetry(t, rel_range, missile, alpha, beta)

        for view_items in [self.items_far, self.items_close]:
            view_items["missile_velocity"].setData(pos=np.vstack((missile_position, missile_position + R_ib @ missile_velocity)))
            view_items["target_velocity"].setData(pos=np.vstack((target_position, target_position + target_velocity)))
            view_items["missile_x_axis"].setData(pos=np.vstack((missile_position, missile_position + R_ib @ [missile.L, 0, 0])))
            view_items["missile_y_axis"].setData(pos=np.vstack((missile_position, missile_position + R_ib @ [0, missile.L, 0])))
            view_items["missile_z_axis"].setData(pos=np.vstack((missile_position, missile_position + R_ib @ [0, 0, missile.L])))
            view_items["drag_force"].setData(pos=np.vstack((missile_position, missile_position + (R_iw @ [Fw_aero[0], 0, 0]) * self.force_scale)))
            view_items["side_force"].setData(pos=np.vstack((missile_position, missile_position + (R_iw @ [0, Fw_aero[1], 0]) * self.force_scale)))
            view_items["lift_force"].setData(pos=np.vstack((missile_position, missile_position + (R_iw @ [0, 0, Fw_aero[2]]) * self.force_scale)))
            view_items["desired_lateral_accel"].setData(pos=np.vstack((missile_position, missile_position + a_lat_desired * self.acceleration_scale)))
            view_items["achieved_lateral_accel"].setData(pos=np.vstack((missile_position, missile_position + a_lat_achieved * self.acceleration_scale)))

        # Missile Mesh Transformations
        # Construct the base 4x4 transformation matrix for the missile
        missile_base_transform = self._homogeneous_transform(R_ib, missile_position)

        # Missile cylinder (main body) offset and orientation
        missile_center_offset = -missile.L / 2.0
        missile_body_transform = QtGui.QMatrix4x4(missile_base_transform)
        missile_body_transform.translate(missile_center_offset, 0.0, 0.0)
        missile_body_transform.rotate(90.0, 0.0, 1.0, 0.0)

        # Missile cone (nose) offset and orientation
        missile_cone_offset = missile.L / 2.0 - 0.2 * missile.L
        missile_nose_transform = QtGui.QMatrix4x4(missile_base_transform)
        missile_nose_transform.translate(missile_cone_offset, 0.0, 0.0)
        missile_nose_transform.rotate(90.0, 0.0, 1.0, 0.0)

        # Target Mesh Transformations
        # Compute target orientation based on its current velocity vector
        R_target = utils.direction_to_rotation_matrix(target_velocity)
        # Construct the base 4x4 transformation matrix for the target
        target_base_transform = self._homogeneous_transform(R_target, target_position)

        # Target cylinder (main body) offset and orientation
        target_center_offset = -target.L / 2.0
        target_body_transform = QtGui.QMatrix4x4(target_base_transform)
        target_body_transform.translate(target_center_offset, 0.0, 0.0)
        target_body_transform.rotate(90.0, 0.0, 1.0, 0.0)

        # Target cone (nose) offset and orientation
        target_cone_offset = target.L / 2.0 - 0.2 * target.L
        target_nose_transform = QtGui.QMatrix4x4(target_base_transform)
        target_nose_transform.translate(target_cone_offset, 0.0, 0.0)
        target_nose_transform.rotate(90.0, 0.0, 1.0, 0.0)

        # Updating the missile and target items with new transformations
        fin_angles = [45, 135, 225, 315]
        for view_items in [self.items_far, self.items_close]:
            # Update missile body and nose
            view_items["missile_cylinder"].setTransform(missile_body_transform)
            view_items["missile_cone"].setTransform(missile_nose_transform)

            # Update missile fins
            for fin_idx, angle_deg in enumerate(fin_angles):
                deflection_deg = np.rad2deg(missile.fin_deflections[fin_idx])
                missile_fin_transform = self._transform_fin(angle_deg, missile_base_transform, missile.L, missile.D_ref/2, deflection_deg=deflection_deg)
                view_items[f"missile_fin{fin_idx+1}"].setTransform(missile_fin_transform)

            # Update target body and nose
            view_items["target_cylinder"].setTransform(target_body_transform)
            view_items["target_cone"].setTransform(target_nose_transform)

            # Update target fins
            for fin_idx, angle_deg in enumerate(fin_angles):
                target_fin_transform = self._transform_fin(angle_deg, target_base_transform, target.L, target.D_ref/2, fin_scale_xy=1.0, fin_scale_z=0.8)
                view_items[f"target_fin{fin_idx+1}"].setTransform(target_fin_transform)

        self.view_close.opts['center'] = pg.QtGui.QVector3D(*missile_position)
        self.view_far.opts['center'] = pg.QtGui.QVector3D(*((missile_position + target_position)/2))

        self._position_overlays()
        self._sync_view_angles()
        self.app.processEvents()

        if self.record:
            # Capture the entire main window
            qpixmap = self.main_window.grab()
            if qpixmap.isNull():
                return

            # Fast QImage to PIL Image conversion using numpy memory mapping
            qimage = qpixmap.toImage().convertToFormat(QtGui.QImage.Format.Format_RGBA8888)
            ptr = qimage.bits()

            # For PyQt-based bindings, the void pointer needs its size set before conversion
            if hasattr(ptr, 'setsize'):
                ptr.setsize(qimage.height() * qimage.width() * 4)

            arr = np.frombuffer(ptr, np.uint8).reshape((qimage.height(), qimage.width(), 4))
            self.frames.append(Image.fromarray(arr, 'RGBA').convert('RGB'))

    def finalize(self):
        """Handles post-simulation window management."""
        if self.record and self.frames:
            gif_path = os.path.join('media', 'live_sim_visualization.gif')
            print(f"Saving {len(self.frames)} frames to {gif_path}...")
            self.frames[0].save(gif_path, save_all=True, append_images=self.frames[1:], duration=50, loop=0)
            print(f"GIF saved as {gif_path}")

        if self.main_window.isVisible():
            print("Simulation finished. Close the window to continue.")
            self.app.exec()

    def _setup_view_items(self, view, missile, target):
        """Initializes the 3D items for a given view."""

        grid = gl.GLGridItem()
        grid.scale(5000, 5000, 1)
        view.addItem(grid)

        # Fin mesh data
        vertices = np.array([[-0.5,-0.5,-0.5],[0.5,-0.5,-0.5],[0.5,0.5,-0.5],[-0.5,0.5,-0.5],[-0.5,-0.5,0.5],[-0.2,-0.5,0.5],[-0.2,0.5,0.5],[-0.5,0.5,0.5]])
        faces = np.array([[0,1,2],[0,2,3],[4,5,6],[4,6,7],[0,1,5],[0,5,4],[1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7]])
        fin_meshdata = gl.MeshData(vertexes=vertices, faces=faces)

        items = {
            "missile_boost_trajectory": gl.GLLinePlotItem(color=(1, 0.5, 0, 1), width=2.0, antialias=True),
            "missile_coast_trajectory": gl.GLLinePlotItem(color=(0, 0.4, 1, 1), width=2.0, antialias=True),
            "target_trajectory": gl.GLLinePlotItem(color=(1, 0, 0, 1), width=2.0, antialias=True),
            "los_line": gl.GLLinePlotItem(color=(0.5, 0.5, 0.5, 1), width=2.0, antialias=True),
            "missile_velocity": gl.GLLinePlotItem(color=(1, 1, 1, 1), width=2.0, antialias=True),
            "target_velocity": gl.GLLinePlotItem(color=(1, 1, 1, 1), width=2.0, antialias=True),
            "missile_x_axis": gl.GLLinePlotItem(color=(1, 0, 0, 1), width=2.0, antialias=True),
            "missile_y_axis": gl.GLLinePlotItem(color=(0, 1, 0, 1), width=2.0, antialias=True),
            "missile_z_axis": gl.GLLinePlotItem(color=(0, 0, 1, 1), width=2.0, antialias=True),
            "drag_force": gl.GLLinePlotItem(color=(1, 0.6, 0, 1), width=2.0, antialias=True),
            "side_force": gl.GLLinePlotItem(color=(0.5, 0, 0.5, 1), width=2.0, antialias=True),
            "lift_force": gl.GLLinePlotItem(color=(1, 1, 0, 1), width=2.0, antialias=True),
            "desired_lateral_accel": gl.GLLinePlotItem(color=(1, 0, 1, 1), width=2.0, antialias=True),
            "achieved_lateral_accel": gl.GLLinePlotItem(color=(0, 1, 1, 1), width=2.0, antialias=True),
            "missile_cylinder": gl.GLMeshItem(meshdata=gl.MeshData.cylinder(rows=10, cols=40, radius=[missile.D_ref/2, missile.D_ref/2], length=missile.L*0.8), color=(0.7,0.7,0.7,1.0), shader='shaded'),
            "missile_cone": gl.GLMeshItem(meshdata=gl.MeshData.cylinder(rows=10, cols=40, radius=[missile.D_ref/2, 0.0], length=missile.L*0.2), color=(0.9,0.9,0.9,1.0), shader='shaded'),
            "missile_fin1": gl.GLMeshItem(meshdata=fin_meshdata, color=(0.9,0.9,0.9,1.0), shader='shaded'),
            "missile_fin2": gl.GLMeshItem(meshdata=fin_meshdata, color=(0.9,0.9,0.9,1.0), shader='shaded'),
            "missile_fin3": gl.GLMeshItem(meshdata=fin_meshdata, color=(0.9,0.9,0.9,1.0), shader='shaded'),
            "missile_fin4": gl.GLMeshItem(meshdata=fin_meshdata, color=(0.9,0.9,0.9,1.0), shader='shaded'),
            "target_cylinder": gl.GLMeshItem(meshdata=gl.MeshData.cylinder(rows=10, cols=40, radius=[target.D_ref/2, target.D_ref/2], length=target.L*0.8), color=(0.5,0.1,0.1,1.0), shader='shaded'),
            "target_cone": gl.GLMeshItem(meshdata=gl.MeshData.cylinder(rows=10, cols=40, radius=[target.D_ref/2, 0.0], length=target.L*0.2), color=(0.7,0.2,0.2,1.0), shader='shaded'),
            "target_fin1": gl.GLMeshItem(meshdata=fin_meshdata, color=(0.6,0.2,0.2,1.0), shader='shaded'),
            "target_fin2": gl.GLMeshItem(meshdata=fin_meshdata, color=(0.6,0.2,0.2,1.0), shader='shaded'),
            "target_fin3": gl.GLMeshItem(meshdata=fin_meshdata, color=(0.6,0.2,0.2,1.0), shader='shaded'),
            "target_fin4": gl.GLMeshItem(meshdata=fin_meshdata, color=(0.6,0.2,0.2,1.0), shader='shaded')
        }
        for ax in ["missile_x_axis", "missile_y_axis", "missile_z_axis"]: items[ax].setGLOptions('additive')
        for item in items.values(): view.addItem(item)
        return items

    def _setup_overlays(self):
        """Initializes the legend and telemetry overlays."""

        trajectory_legend_entries = [
            ((1, 0, 0), "Target Trajectory"),
            ((1, 0.5, 0), "Missile Trajectory (Boost Phase)"),
            ((0, 0.4, 1), "Missile Trajectory (Coast Phase)")
        ]
        missile_legend_entries = [
            ((1, 0, 0), "Missile Body X"),
            ((0, 1, 0), "Missile Body Y"),
            ((0, 0, 1), "Missile Body Z"),
            ((1, 1, 1), "Velocity Vector (m/s)"),
            ((1, 0, 1), "Desired Lateral Acceleration (m/s^2)"),
            ((0, 1, 1), "Achieved Lateral Acceleration (m/s^2)"),
            ((1, 0.6, 0), "Drag (Wind Frame) (N)"),
            ((0.5, 0, 0.5), "Side Force (Wind Frame) (N)"),
            ((1, 1, 0), "Lift (Wind Frame) (N)")
        ]

        self.legend_far = self._create_legend_widget(self.view_far, trajectory_legend_entries)
        self.legend_far.show()

        self.legend_close = self._create_legend_widget(self.view_close, missile_legend_entries)
        self.legend_close.show()

        self.telemetry_label = pg.QtWidgets.QLabel(self.view_far)
        self.telemetry_label.setStyleSheet("background-color: rgba(255, 255, 255, 180); padding: 8px; border: 1px solid black; font-family: monospace; font-size: 11pt;")
        self.telemetry_label.move(10, 10)
        self.telemetry_label.show()

        self._position_overlays()

    def _position_overlays(self):
        """Recalculates positions for floating UI overlays to handle window resizing."""

        # Right-align the far trajectory view legend
        x_pos_far = self.view_far.width() - self.legend_far.width() - 10
        self.legend_far.move(max(10, x_pos_far), 10)

        # Left-align the close missile view legend
        self.legend_close.move(10, 10)

    def _create_legend_widget(self, parent, entries):
        """Creates legend overlay."""

        widget = pg.QtWidgets.QWidget(parent)
        widget.setStyleSheet("background-color: rgba(255, 255, 255, 180); padding: 5px; border: 1px solid black;")
        layout = pg.QtWidgets.QGridLayout(widget)
        for row_idx, (color, text) in enumerate(entries):
            color_box = pg.QtWidgets.QLabel()
            color_box.setFixedSize(20, 10)
            color_box.setStyleSheet(f"background-color: rgb({int(color[0]*255)}, {int(color[1]*255)}, {int(color[2]*255)});")
            label = pg.QtWidgets.QLabel(text)
            label.setStyleSheet("border: none; background-color: transparent; font-weight: bold; font-family: sans-serif; font-size: 9pt;")
            layout.addWidget(color_box, row_idx, 0)
            layout.addWidget(label, row_idx, 1)
        widget.adjustSize()
        return widget

    def _update_telemetry(self, t, rel_range, missile, alpha, beta):
        """Updates the telemetry overlay."""

        speed = missile.speed()
        lateral_g = np.linalg.norm(missile.achieved_lateral_accel()) / 9.81
        speed_mach = speed / 343.0

        telemetry = (
            f"<b><font size='+1'>Missile Telemetry</font></b>"
            f"<table style='margin-top: 5px;'>"
            f"<tr><td>Time:</td><td style='padding-left: 10px;'>{t: >6.1f} s</td></tr>"
            f"<tr><td>Phase:</td><td style='padding-left: 10px;'>{missile.current_flight_phase()}</td></tr>"
            f"<tr><td>Speed:</td><td style='padding-left: 10px;'>Mach {speed_mach: >6.1f} ({speed:.0f} m/s)</td></tr>"
            f"<tr><td>Mass:</td><td style='padding-left: 10px;'>{missile.mass(): >6.1f} kg</td></tr>"
            f"<tr><td>Range:</td><td style='padding-left: 10px;'>{rel_range: >6.1f} m</td></tr>"
            f"<tr><td>Altitude:</td><td style='padding-left: 10px;'>{missile.position()[2]: >6.1f} m</td></tr>"
            f"<tr><td>Lateral G:</td><td style='padding-left: 10px;'>{lateral_g: >6.1f} G</td></tr>"
            f"<tr><td>Alpha:</td><td style='padding-left: 10px;'>{np.rad2deg(alpha): >6.1f} deg</td></tr>"
            f"<tr><td>Beta:</td><td style='padding-left: 10px;'>{np.rad2deg(beta): >6.1f} deg</td></tr>"
            f"</table>"
        )
        self.telemetry_label.setText(telemetry)
        self.telemetry_label.adjustSize()

    def _sync_view_angles(self):
        """Synchronizes view angles between the far and close view windows."""

        # Increment azimuth at a constant rate if rotation is enabled
        if self.record:
            self.view_azimuth += self.azimuth_rotation_rate

            # Wrap azimuth to keep it in [-180, 180] range
            if self.view_azimuth > 180:
                self.view_azimuth -= 360
            elif self.view_azimuth < -180:
                self.view_azimuth += 360

            # Apply the rotating view to both windows
            self.view_far.setCameraPosition(elevation=self.view_elevation, azimuth=self.view_azimuth)
            self.view_close.setCameraPosition(elevation=self.view_elevation, azimuth=self.view_azimuth)
        else:
            # Setting view_close to match view_far
            if self.view_far.opts['elevation'] != self.view_elevation or self.view_far.opts['azimuth'] != self.view_azimuth:
                self.view_elevation = self.view_far.opts['elevation']
                self.view_azimuth = self.view_far.opts['azimuth']
                self.view_close.setCameraPosition(elevation=self.view_elevation, azimuth=self.view_azimuth)
            # Setting view_far to match view_close
            elif self.view_close.opts['elevation'] != self.view_elevation or self.view_close.opts['azimuth'] != self.view_azimuth:
                self.view_elevation = self.view_close.opts['elevation']
                self.view_azimuth = self.view_close.opts['azimuth']
                self.view_far.setCameraPosition(elevation=self.view_elevation, azimuth=self.view_azimuth)

    def _transform_fin(self, rotation_angle, base_transform, body_length, fin_radius, fin_scale_xy=0.5, fin_scale_z=0.4, deflection_deg=0.0):
        """Creates fin transformation matrices."""

        fin_transform = QtGui.QMatrix4x4(base_transform)
        fin_transform.rotate(rotation_angle, 1, 0, 0)
        fin_transform.translate(-body_length/2 + fin_scale_xy/2.0, 0, fin_radius + fin_scale_z/2)
        fin_transform.rotate(deflection_deg, 0, 0, 1)
        fin_transform.scale(fin_scale_xy, 0.03, fin_scale_z)
        return fin_transform

    def _homogeneous_transform(self, rotation_matrix, translation_vector):
        """Creates a homogeneous transformation matrix from a rotation matrix and a translation vector."""

        transform = QtGui.QMatrix4x4(rotation_matrix[0, 0], rotation_matrix[0, 1], rotation_matrix[0, 2], translation_vector[0],
                                     rotation_matrix[1, 0], rotation_matrix[1, 1], rotation_matrix[1, 2], translation_vector[1],
                                     rotation_matrix[2, 0], rotation_matrix[2, 1], rotation_matrix[2, 2], translation_vector[2],
                                     0.0,                0.0,                0.0,                1.0)

        return transform

def plot_metrics(missile_log, target_log):
    # Convert all python lists to numpy arrays for missile_log and target_log
    missile_log = utils.convert_dict_list_to_dict_array(missile_log)
    target_log = utils.convert_dict_list_to_dict_array(target_log)

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
    ax.plot(missile_log["time"], np.degrees(missile_log["virtual_control_deltas"][:, 0]), label='Aileron (Virtual)', linestyle='--')
    ax.plot(missile_log["time"], np.degrees(missile_log["virtual_control_deltas"][:, 1]), label='Elevator (Virtual)', linestyle='--')
    ax.plot(missile_log["time"], np.degrees(missile_log["virtual_control_deltas"][:, 2]), label='Rudder (Virtual)', linestyle='--')
    ax.plot(missile_log["time"], np.degrees(missile_log["fin_deflections"][:, 0]), label='Fin 1')
    ax.plot(missile_log["time"], np.degrees(missile_log["fin_deflections"][:, 1]), label='Fin 2')
    ax.plot(missile_log["time"], np.degrees(missile_log["fin_deflections"][:, 2]), label='Fin 3')
    ax.plot(missile_log["time"], np.degrees(missile_log["fin_deflections"][:, 3]), label='Fin 4')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Control Deflection (deg)')
    ax.grid()
    ax.legend(loc='lower center', fontsize='small')

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
        axis_length = length
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
