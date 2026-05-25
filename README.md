# PN-Based Missile Guidance
Implementation and simulation of a proportional navigation guidance law and a 6-DoF flight controller for surface-to-air missile interception.

![missile interception animation](media/missile_interception_animation.gif)

![missile interception metrics](media/missile_interception_metrics.png)


## TODO:
1) Add cross-coupling between forces and moments along multiple axes
2) Add variable inertia rate due to mass flow in dwbdt equations
3) Add changing moment arm to pitch due to changing CG from mass flow
4) Make force and moment coefficients functions of Mach instead of constants
5) Perform more rigorous controller design
6) Re-order functions in code to make logic flow easier to understand
7) Make data logging neater in simulation loop