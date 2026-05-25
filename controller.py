import numpy as np
import utils

class MissileController:
    """
    Implements a 3-axis missile autopilot:
    - Roll: Cascaded attitude/rate PI control
    - Pitch: 3-loop acceleration control
    - Yaw: 3-loop acceleration control
    """
    def __init__(self, roll_gains: dict, pitch_gains: dict, yaw_gains: dict, P_dyn_min: float, P_dyn_ref: float, integral_limit: float, delta_limit: float):
        # Roll gains (cascaded attitude/rate PI control)
        self.Kp_roll = roll_gains['Kp_roll'] # Roll angle proportional gain to generate a roll rate command from roll angle error
        self.Kp_roll_rate = roll_gains['Kp_roll_rate'] # Roll rate proportional gain to generate a roll control surface deflection command from roll rate error
        self.Ki_roll_rate = roll_gains['Ki_roll_rate'] # Roll rate integral gain to eliminate steady-state error in roll rate tracking

        # Pitch gains (3-loop acceleration control)
        self.Kdc_pitch = pitch_gains['Kdc_pitch'] # DC gain to scale guidance acceleration command into expected acceleration
        self.Ka_pitch_rate = pitch_gains['Ka_pitch_rate'] # Accleration loop gain to convert acceleration error into a pitch rate command
        self.Ki_pitch_rate = pitch_gains['Ki_pitch_rate'] # Integral loop gain to eliminate steady-state error in pitch rate tracking
        self.Kr_pitch_rate = pitch_gains['Kr_pitch_rate'] # Rate loop gain to convert pitch rate error into a pitch control surface deflection command

        # Yaw gains (3-loop acceleration control)
        self.Kdc_yaw = yaw_gains['Kdc_yaw'] # DC gain to scale guidance acceleration command into expected acceleration
        self.Ka_yaw_rate = yaw_gains['Ka_yaw_rate'] # Accleration loop gain to convert acceleration error into a yaw rate command
        self.Ki_yaw_rate = yaw_gains['Ki_yaw_rate'] # Integral loop gain to eliminate steady-state error in yaw rate tracking
        self.Kr_yaw_rate = yaw_gains['Kr_yaw_rate'] # Rate loop gain to convert yaw rate error into a yaw control surface deflection command

        # TODO: Implement better integrator anti-windup strategy than just clamping the integral term
        # Integrator states
        self.int_roll_rate_error = 0.0
        self.int_pitch_rate_error = 0.0
        self.int_yaw_rate_error = 0.0
        self.integral_limit = integral_limit # Anti-windup clamp limit (rad)

        self.P_dyn_min = P_dyn_min # Minimum dynamic pressure for gain scheduling to avoid excessive control deflections at very low dynamic pressures (Pa)
        self.P_dyn_ref = P_dyn_ref # Reference dynamic pressure for gain scheduling (Pa)
        self.delta_limit = delta_limit # Max control surface deflection (rad)

    def update(self, roll_cmd: float, a_cmd_body: np.ndarray, a_body: np.ndarray, w: np.ndarray, q: np.ndarray, P_dyn: float, dt: float) -> np.ndarray:
        """
        Computes the virtual control effector commands (aileron, elevator, rudder).

        Inputs:
            roll_cmd: Desired roll angle command (rad)
            a_cmd_body: Guidance commands in body frame (lateral acceleration commands from the guidance law, rotated from inertial frame to body frame)
            a_body: Specific force in body frame (what an accelerometer would measure)
            w: Angular velocity in body frame (what a gyroscope would measure)
            q: Attitude quaternion (body to inertial)
            P_dyn: Current dynamic pressure for gain scheduling (Pa)
            dt: Time step for integration (sec)
        Outputs:
            control_deltas: Control surface deflection commands for roll, pitch, yaw (rad)
        """

        # NOTE: Pure PN guidance command should generate ax_cmd ~= 0 relative to missile's velocity vector
        # TODO: Double check sign convention for az
        ax_cmd, ay_cmd, az_cmd = a_cmd_body

        # TODO: Double check sign conventions for az
        ax, ay, az = a_body

        wx, wy, wz = w

        roll = utils.quaternion_to_roll(q)

        # Compute control deflections for each axis
        delta_roll = self.roll_feedback_control(roll_cmd, roll, wx, dt)
        delta_pitch = self.pitch_feedback_control(az_cmd, az, wy, dt)
        delta_yaw = self.yaw_feedback_control(ay_cmd, ay, wz, dt)

        control_deltas = np.array([delta_roll, delta_pitch, delta_yaw])

        # Scale deflections inversely with dynamic pressure: More deflection needed at low speed, less deflection needed at high speed, for the same aerodynamic effect
        P_dyn_eff = max(P_dyn, self.P_dyn_min) # Avoid division by zero or very small dynamic pressures for gain scheduling
        P_dyn_ratio = self.P_dyn_ref / P_dyn_eff
        control_deltas = P_dyn_ratio * control_deltas # Scale control deflections inversely with dynamic pressure for gain scheduling

        control_deltas = np.clip(control_deltas, -self.delta_limit, self.delta_limit)

        return control_deltas

    def roll_feedback_control(self, roll_cmd: float, roll: float, wx: float, dt: float) -> float:
        """
        Implements a cascaded PI roll control loop:
        - Outer loop: Roll angle error is scaled by a proportional gain to generate a roll rate command
        - Inner loop: Roll rate error is computed by comparing the roll rate command to the actual roll
        rate measurement from the gyro. The roll rate error is then fed into a PI controller to generate the final roll control surface deflection command.

        Inputs:
            roll_cmd: Desired roll angle command (rad)
            roll: Measured roll angle (rad)
            wx: Measured roll rate from gyro (rad/s)
            dt: Time step for integration (sec)
        Outputs:
            delta_roll: Roll control surface deflection command (rad)
        """

        # Outer Loop Error: Scale roll angle error to generate a targeted body roll rate command (rad/s)
        roll_rate_cmd = self.Kp_roll * (roll_cmd - roll)

        # Inner Loop Error: Compute deviation between requested roll rate and actual gyro measurement
        roll_rate_error = roll_rate_cmd - wx

        # Inner Loop Integral: Accumulate roll rate error over time step to eliminate steady-state tracking offsets
        self.int_roll_rate_error += self.Ki_roll_rate * roll_rate_error * dt
        self.int_roll_rate_error = np.clip(self.int_roll_rate_error, -self.integral_limit, self.integral_limit) # Integral anti-windup clamp

        # Output: Compute contol deflection for roll by combining proportional roll rate tracking with the integrated roll rate error to eliminate steady-state offsets
        delta_roll = (self.Kp_roll_rate * roll_rate_error) + self.int_roll_rate_error

        return delta_roll

    def pitch_feedback_control(self, az_cmd: float, az: float, wy: float, dt: float) -> float:
        """
        Controller implementation based on the block-diagram depicted in figure 6 in paper titled "Overview of Missile Flight Control Systems" by Paul B. Jackson

        Implements a 3-loop pitch acceleration control:
        - Outer loop: Compare measured linear acceleration against scaled guidance acceleration command to compute an acceleration error
        - Mid loop: Scale linear acceleration error into a body pitch rate command using a proportional gain
        - Inner loop: Compute error between the requested body pitch rate and actual gyro pitch rate, then feed the pitch rate error into a PI controller to generate the final pitch control surface deflection command.

        Inputs:
            az_cmd: Desired acceleration command in body z-axis from guidance law (m/s^2)
            az: Measured acceleration in body z-axis from accelerometer (m/s^2)
            wy: Measured pitch rate from gyro (rad/s)
            dt: Time step for integration (sec)
        Outputs:
            delta_pitch: Pitch control surface deflection command (rad)
        """

        # NOTE: If +pitch results in -az, we either need -ve pitch gains, or we need to reverse the error calculation to: az_error = az - (K_dcp * az_cmd)
        # Outer Loop Error: Compare measured linear acceleration against scaled guidance acceleration command
        az_error = az - (self.Kdc_pitch * az_cmd)

        # Mid-Loop: Scale linear acceleration error into a body pitch rate (rad/s)
        pitch_rate_cmd = self.Ka_pitch_rate * az_error

        # Inner Loop Error: Compute error between the requested body pitch rate and actual gyro pitch rate
        pitch_rate_error = pitch_rate_cmd - wy

        # Inner Loop Integral: Accumulate pitch rate error over time step to eliminate steady-state tracking offsets
        self.int_pitch_rate_error += self.Ki_pitch_rate * pitch_rate_error * dt
        self.int_pitch_rate_error = np.clip(self.int_pitch_rate_error, -self.integral_limit, self.integral_limit) # Integral anti-windup clamp

        # Output: Compute control deflection for pitch based on difference between integrated pitch rate error and current pitch rate
        delta_pitch = self.Kr_pitch_rate * (self.int_pitch_rate_error - wy)

        return delta_pitch

    def yaw_feedback_control(self, ay_cmd: float, ay: float, wz: float, dt: float) -> float:
        """
        Controller implementation based on the block-diagram depicted in figure 6 in paper titled "Overview of Missile Flight Control Systems" by Paul B. Jackson

        Implements a 3-loop yaw acceleration control:
        - Outer loop: Compare measured linear acceleration against scaled guidance acceleration command to compute an acceleration error
        - Mid loop: Scale linear acceleration error into a body yaw rate command using a proportional gain
        - Inner loop: Compute error between the requested body yaw rate and actual gyro yaw rate, then feed the yaw rate error into a PI controller to generate the final yaw control surface deflection command.

        Inputs:
            ay_cmd: Desired acceleration command in body y-axis from guidance law (m/s^2)
            ay: Measured acceleration in body y-axis from accelerometer (m/s^2)
            wz: Measured yaw rate from gyro (rad/s)
            dt: Time step for integration (sec)
        Outputs:
            delta_yaw: Yaw control surface deflection command (rad)
        """

        # Outer Loop Error: Compare measured linear acceleration against scaled guidance acceleration command
        ay_error = (self.Kdc_yaw * ay_cmd) - ay

        # Mid-Loop: Scale linear acceleration error into a body yaw rate (rad/s)
        yaw_rate_cmd = self.Ka_yaw_rate * ay_error

        # Inner Loop Error: Compute error between the requested body yaw rate and actual gyro yaw rate
        yaw_rate_error = yaw_rate_cmd - wz

        # Inner Loop Integral: Accumulate yaw rate error over time step to eliminate steady-state tracking offsets
        self.int_yaw_rate_error += self.Ki_yaw_rate * yaw_rate_error * dt
        self.int_yaw_rate_error = np.clip(self.int_yaw_rate_error, -self.integral_limit, self.integral_limit) # Integral anti-windup clamp

        # Output: Compute control deflection for yaw based on difference between integrated yaw rate error and current yaw rate
        delta_yaw = self.Kr_yaw_rate * (self.int_yaw_rate_error - wz)

        return delta_yaw