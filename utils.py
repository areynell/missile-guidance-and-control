import numpy as np

def quaternion_multiply(q1, q2):
    """Multiplies two quaternions q1 and q2 and returns the resulting quaternion."""

    qw1 = q1[0]
    qx1 = q1[1]
    qy1 = q1[2]
    qz1 = q1[3]

    qw2 = q2[0]
    qx2 = q2[1]
    qy2 = q2[2]
    qz2 = q2[3]

    qw = qw1*qw2 - qx1*qx2 - qy1*qy2 - qz1*qz2
    qx = qw1*qx2 + qx1*qw2 + qy1*qz2 - qz1*qy2
    qy = qw1*qy2 - qx1*qz2 + qy1*qw2 + qz1*qx2
    qz = qw1*qz2 + qx1*qy2 - qy1*qx2 + qz1*qw2

    return np.array([qw, qx, qy, qz], dtype=float)

def quaternion_normalization(q):
    """Normalizes a quaternion to have unit magnitude."""
    return q / np.linalg.norm(q)

def quaternion_to_rotation_matrix(q):
    """Converts a quaternion into a 3x3 rotation matrix."""

    qw, qx, qy, qz = q

    R = np.array([
        [1.0 - 2.0*(qy**2 + qz**2),     2.0*(qx*qy - qz*qw),     2.0*(qx*qz + qy*qw)],
        [    2.0*(qx*qy + qz*qw), 1.0 - 2.0*(qx**2 + qz**2),     2.0*(qy*qz - qx*qw)],
        [    2.0*(qx*qz - qy*qw),     2.0*(qy*qz + qx*qw), 1.0 - 2.0*(qx**2 + qy**2)]
    ], dtype=float)

    return R

def quaternion_to_rpy_deg(q):
    """Converts a quaternion into roll, pitch, and yaw angles (in degrees)."""

    qw, qx, qy, qz = q

    roll = np.arctan2(2.0*(qw*qx + qy*qz), 1.0 - 2.0*(qx**2 + qy**2))
    pitch = np.arcsin(np.clip(2.0*(qw*qy - qz*qx), -1.0, 1.0))
    yaw = np.arctan2(2.0*(qw*qz + qx*qy), 1.0 - 2.0*(qy**2 + qz**2))

    return np.degrees(np.array([roll, pitch, yaw], dtype=float))

def quaternion_to_roll(q):
    """Extracts the roll angle (in radians) from a quaternion."""
    qw, qx, qy, qz = q
    return np.arctan2(2.0*(qw*qx + qy*qz), 1.0 - 2.0*(qx**2 + qy**2))

def quaternion_to_pitch(q):
    """Extracts the pitch angle (in radians) from a quaternion."""
    qw, qx, qy, qz = q
    return np.arcsin(np.clip(2.0*(qw*qy - qz*qx), -1.0, 1.0))

def quaternion_to_yaw(q):
    """Extracts the yaw angle (in radians) from a quaternion."""
    qw, qx, qy, qz = q
    return np.arctan2(2.0*(qw*qz + qx*qy), 1.0 - 2.0*(qy**2 + qz**2))

def vector_to_skew_symmetric_matrix(v):
    """Converts a 3-element vector into a 3x3 skew-symmetric matrix."""
    return np.array([
        [0.0,  -v[2], v[1]],
        [v[2],  0.0, -v[0]],
        [-v[1], v[0], 0.0]
    ], dtype=float)

def wind_to_body_rotation_matrix(alpha, beta):
    """Creates a rotation matrix that transforms vectors from the wind frame to the body frame based on angle of attack (alpha) and sideslip angle (beta)."""

    # Angle of attack rotation matrix (about the y-axis)
    Ry_alpha = np.array([[np.cos(alpha), 0.0, -np.sin(alpha)],
                         [0.0,           1.0,  0.0           ],
                         [np.sin(alpha), 0.0,  np.cos(alpha) ]], dtype=float)

    # Sideslip angle rotation matrix (about the z-axis)
    Rz_beta = np.array([[np.cos(beta), -np.sin(beta), 0.0],
                        [np.sin(beta),  np.cos(beta), 0.0],
                        [0.0,           0.0,          1.0]], dtype=float)

    # Combined rotation from wind frame to body frame
    R_bw = Rz_beta @ Ry_alpha  # body <- wind
    return R_bw

def compute_air_density(altitude, rho0, H_scale):
    """Calculates air density at a given altitude using an exponential model."""

    altitude = max(altitude, 0.0)
    return rho0 * np.exp(-altitude / H_scale)

def convert_dict_list_to_dict_array(data_dict):
    """Converts all lists within a dictionary to numpy arrays."""

    for key in data_dict:
        if isinstance(data_dict[key], list):
            data_dict[key] = np.array(data_dict[key])

    return data_dict

def direction_to_rotation_matrix(direction_vector: np.ndarray) -> np.ndarray:
    """
    Calculates a 3x3 rotation matrix that aligns the local X-axis (forward)
    with a given 3D direction vector (like velocity).
    """

    magnitude = np.linalg.norm(direction_vector)

    # If there is no movement, return identity (no rotation)
    if magnitude < 1e-3:
        return np.eye(3)

    # 1. Determine the forward axis (X)
    # Normalize the input vector to create a unit vector for the X direction.
    forward_axis = direction_vector / magnitude

    # 2. Determine the left axis (Y)
    # Find a vector perpendicular to both the global "up" and forward directions.
    global_up = np.array([0.0, 0.0, 1.0])
    # NOTE: This is not yet a unit vector, since global_up and forward_axis are generally not orthogonal.
    left_axis = np.cross(global_up, forward_axis)

    # Singularity check: If flying straight up/down, the cross product will fail.
    # In this case, manually pick a default "left" direction.
    left_mag = np.linalg.norm(left_axis)
    if left_mag < 1e-3:
        left_axis = np.array([0.0, 1.0, 0.0])
    # Normalize the left_axis vector to create a unit vector for the Y direction
    else:
        left_axis = left_axis / left_mag

    # 3. Determine the up axis (Z)
    # The Z axis must be perpendicular to both X and Y to complete the coordinate system
    # NOTE: No need for normalization as forward_axis and left_axis vectors are already unit vectors and orthogonal.
    up_axis = np.cross(forward_axis, left_axis)

    # Combine the three unit vectors as columns of the rotation matrix
    return np.column_stack((forward_axis, left_axis, up_axis))