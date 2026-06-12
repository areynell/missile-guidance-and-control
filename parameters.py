from dataclasses import dataclass, field
import numpy as np

@dataclass
class AerodynamicParams:
    CD_0: float      # Base drag coefficient at zero angle of attack, based on parasitic/skin friction drag of the missile's body and fins
    CD_alpha: float  # Drag static stability derivative (dCD/dAlpha), which captures how the drag coefficient increases with angle of attack
    CD_delta: float  # Drag control derivative (dCD/dDelta), which captures how the drag coefficient changes with control surface deflection
    CY_0: float      # Base side-force coefficient at zero sideslip, which is typically 0 for a symmetric missile body
    CY_beta: float   # Side-force static stability derivative (dCY/dBeta), which captures how the missile generates side force to counteract sideslip and maintain directional stability.
    CY_delta: float  # Side-force control derivative (dCY/dDelta), which captures how effective the rudder is at generating side force for yaw control
    CL_0: float      # Base lift coefficient at zero angle of attack, which is typically 0 for a symmetric missile body
    CL_alpha: float  # Lift static stability derivative (dCL/dAlpha), which captures how effective the missile's body and fins are at generating lift to achieve maneuvering
    CL_delta: float  # Lift control derivative (dCL/dDelta), which captures how effective the elevator is at generating lift for pitch control
    Cl_0: float      # Base rolling moment coefficient at zero aileron deflection, which is typically 0 for a symmetric missile body
    Cl_p: float      # Base roll damping dynamic stability derivative (dCl/dp), which captures how the missile's roll rate generates a restoring rolling moment to stabilize roll oscillations
    Cl_delta: float  # Roll control derivative (dCl/dDelta), which captures how effective the aileron is at generating rolling moment for roll control
    Cm_0: float      # Base pitching moment coefficient at zero angle of attack, which is typically 0 for a symmetric missile body
    Cm_alpha: float  # Pitch static stability derivative (dCm/dAlpha), which captures how the missile's angle of attack generates a restoring pitching moment to stabilize pitch oscillations
    Cm_q: float      # Pitch damping dynamic stability derivative (dCm/dq), which captures how the missile's pitch rate generates a restoring pitching moment to stabilize pitch oscillations
    Cm_delta: float  # Pitch control derivative (dCm/dDelta), which captures how effective the elevator is at generating pitching moment for pitch control
    Cn_0: float      # Base yawing moment coefficient at zero sideslip, which is typically 0 for a symmetric missile body
    Cn_beta: float   # Yaw static stability derivative (dCn/dBeta), which captures how the missile's sideslip angle generates a restoring yawing moment to stabilize directional oscillations
    Cn_r: float      # Yaw damping dynamic stability derivative (dCn/dr), which captures how the missile's yaw rate generates a restoring yawing moment to stabilize directional oscillations
    Cn_delta: float  # Yaw control derivative (dCn/dDelta), which captures how effective the rudder is at generating yawing moment for yaw control

@dataclass
class PropulsionParams:
    thrust: float  # N, missile engine thrust
    Isp: float     # s, specific impulse

@dataclass
class StructuralParams:
    diameter: float           # m, reference diameter of the missile, which is used to non-dimensionalize aerodynamic coefficients
    length: float             # m, length for computing inertia matrix
    dry_mass: float           # kg, mass of the engine and associated hardware, excluding propellant
    total_mass: float         # kg, mass of the missile at launch, including propellant
    max_lateral_accel: float  # m/s^2, structural G-limits for lateral acceleration

@dataclass
class WarheadParams:
    kill_radius: float  # m, radius within which the missile is considered to have successfully intercepted the target

@dataclass
class MissileParams:
    aero: AerodynamicParams
    propulsion: PropulsionParams
    structural: StructuralParams
    warhead: WarheadParams

@dataclass
class GuidanceParams:
    N: float                 # Proportional navigation constant (typically between 3 and 5)
    max_lateral_accel: float # m/s^2, maximum lateral acceleration command limit for the guidance law

@dataclass
class ControllerParams:
    # Roll control gains
    Kp_roll: float
    Kp_roll_rate: float
    Ki_roll_rate: float

    # Pitch control gains
    Kdc_pitch: float
    Ka_pitch_rate: float
    Ki_pitch_rate: float
    Kr_pitch_rate: float

    # Yaw control gains
    Kdc_yaw: float
    Ka_yaw_rate: float
    Ki_yaw_rate: float
    Kr_yaw_rate: float

    # Additional parameters for gain scheduling and anti-windup
    v_ref: float           # m/s, reference speed for gain scheduling
    P_dyn_ref: float       # Pa, reference dynamic pressure for gain scheduling
    P_dyn_min: float       # Pa, minimum dynamic pressure for gain scheduling to avoid excessive control deflections at very low dynamic pressures
    integral_limit: float  # rad, anti-windup limit for integral terms
    delta_limit: float     # rad, maximum control surface deflection

@dataclass
class AtmosphericParams:
    sea_level_density: float  # kg/m^3, density of air at sea level, used for calculating dynamic pressure and aerodynamic forces
    scale_height: float       # m, scale height for atmospheric model, used for calculating how air density decreases with altitude
    wind_vector: np.ndarray = field(default_factory=lambda: np.zeros(3))  # m/s, wind velocity vector in the inertial frame, defaults to zero for no wind conditions