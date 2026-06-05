# Missile Guidance and Control
Implements and simulates 6-DoF Newton-Euler equations of motion for a missile running a proportional navigation (PN) guidance law and a 3-loop cascaded flight controller, for surface-to-air interception scenarios.

<p align="center">
  <img src="media/missile_interception_animation.gif" alt="Missile Interception Animation" width="70%">
</p>

<p align="center">
  <img src="media/missile_interception_metrics.png" alt="Missile Interception Metrics" width="100%">
</p>

<p align="center">
  <img src="media/missile_orientation_and_forces_animation.gif" alt="Missile Interception Orientation and Forces" width="70%">
</p>

## TODO
1) Add cross-coupling between forces and moments along multiple axes
2) Add variable inertia rate due to mass flow in dwbdt equations
3) Add changing moment arm to pitch due to changing CG from mass flow
4) Make force and moment coefficients functions of Mach instead of constants
5) Perform rigorous controller design
6) Make data logging neater in simulation loop
7) Add feedforward guidance commands to flight controller