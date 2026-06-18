# Missile Guidance and Control
Implements and simulates the guidance and control of a 6-DoF surface-to-air missile.

*   **Non-Linear Rigid Body Dynamics:** Full 6-DoF Newton-Euler equations of motion with quaternion-based kinematics to ensure singularity-free attitude tracking.
*   **Aerodynamic Modeling:** Calculation of force and moment coefficients based on angle of attack, sideslip angle and fin deflection control inputs, and atmospheric density modeling using an exponential scale height.
*   **Guidance:** Implementation of pure PN guidance law to generate acceleration commands based on the line-of-sight (LOS) rate between the interceptor and the target.
*   **Feedforward Control:** Conversion of required acceleration commands from guidance system into feedforward fin deflection control inputs to improve transient response.
*   **Feedback Control:** Implementation of a 3-loop cascaded flight controller, which includes a roll PI controller and acceleration-based pitch/yaw controllers with dynamic pressure-based gain scheduling.
*   **Numerical Simulation:** Implementation of a 4th-order Runge-Kutta (RK4) integration approach for joint state propagation of both interceptor and maneuvering targets.
*   **Visualization:** Live 3D interception data is visualized using `pyqtgraph` in the main simulation loop.

<p align="center">
  <img src="media/missile_interception_animation.gif" alt="Missile Interception Animation" width="70%">
</p>

<p align="center">
  <img src="media/missile_interception_metrics.png" alt="Missile Interception Metrics" width="100%">
</p>

<p align="center">
  <img src="media/missile_orientation_and_forces_animation.gif" alt="Missile Interception Orientation and Forces" width="70%">
</p>

## Future Improvements
1) Add changing moment arm to pitch and yaw due to changing CG from mass flow
2) Add variable inertia rate due to mass flow in dwbdt equations
3) Make force and moment aerodynamic coefficients functions of Mach
4) Add cross-coupling to force and moment aerodynamic coefficients
5) Perform rigorous controller design

6) Improve controller's integral anti-windup mechanism