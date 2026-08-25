"""Optimized rotation utilities with Numba JIT compilation

Performance:
- matrix_to_quat: ~55x faster than scipy (0.36 µs vs 19.95 µs)
- quat_to_matrix: ~4.6x faster than scipy (0.35 µs vs 1.60 µs)

Quaternion convention:
- wxyz=False (default): [x, y, z, w] (scipy/PyBullet convention)
- wxyz=True: [w, x, y, z] (Viser/ROS convention)
"""
import numpy as np
from numba import njit

@njit(cache=True, fastmath=True)
def matrix_to_quat(rot, wxyz=False):
    """Convert rotation matrix to quaternion - JIT compiled

    Args:
        rot: 3x3 rotation matrix (numpy array)
        wxyz: If True, return [w, x, y, z]. If False (default), return [x, y, z, w]

    Returns:
        Quaternion as [x, y, z, w] (default) or [w, x, y, z] (if wxyz=True)

    Note: Uses Shepperd's method for numerical stability
    Performance: ~55x faster than scipy.spatial.transform.Rotation
    """
    trace = rot[0, 0] + rot[1, 1] + rot[2, 2]

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (rot[2, 1] - rot[1, 2]) * s
        y = (rot[0, 2] - rot[2, 0]) * s
        z = (rot[1, 0] - rot[0, 1]) * s
    elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
        s = 2.0 * np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2])
        w = (rot[2, 1] - rot[1, 2]) / s
        x = 0.25 * s
        y = (rot[0, 1] + rot[1, 0]) / s
        z = (rot[0, 2] + rot[2, 0]) / s
    elif rot[1, 1] > rot[2, 2]:
        s = 2.0 * np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2])
        w = (rot[0, 2] - rot[2, 0]) / s
        x = (rot[0, 1] + rot[1, 0]) / s
        y = 0.25 * s
        z = (rot[1, 2] + rot[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1])
        w = (rot[1, 0] - rot[0, 1]) / s
        x = (rot[0, 2] + rot[2, 0]) / s
        y = (rot[1, 2] + rot[2, 1]) / s
        z = 0.25 * s

    if wxyz:
        return np.array([w, x, y, z])
    else:
        return np.array([x, y, z, w])

@njit(cache=True, fastmath=True)
def quat_to_matrix(quat, wxyz=False):
    """Convert quaternion to rotation matrix - JIT compiled

    Args:
        quat: Quaternion as [x, y, z, w] (default) or [w, x, y, z] (if wxyz=True)
        wxyz: If True, input is [w, x, y, z]. If False (default), input is [x, y, z, w]

    Returns:
        3x3 rotation matrix (numpy array)

    Performance: ~4.6x faster than scipy.spatial.transform.Rotation
    """
    if wxyz:
        w, x, y, z = quat
    else:
        x, y, z, w = quat

    # Precompute repeated terms
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    # Build rotation matrix
    mat = np.empty((3, 3), dtype=np.float64)
    mat[0, 0] = 1.0 - 2.0 * (yy + zz)
    mat[0, 1] = 2.0 * (xy - wz)
    mat[0, 2] = 2.0 * (xz + wy)
    mat[1, 0] = 2.0 * (xy + wz)
    mat[1, 1] = 1.0 - 2.0 * (xx + zz)
    mat[1, 2] = 2.0 * (yz - wx)
    mat[2, 0] = 2.0 * (xz - wy)
    mat[2, 1] = 2.0 * (yz + wx)
    mat[2, 2] = 1.0 - 2.0 * (xx + yy)

    return mat

@njit(cache=True, fastmath=True)
def matrix_to_rotvec(rot):
    """Convert rotation matrix to rotation vector (axis-angle) - JIT compiled

    Args:
        rot: 3x3 rotation matrix (numpy array)

    Returns:
        3D rotation vector where direction is axis and magnitude is angle (radians)

    Note: Uses Rodriguez formula
    """
    # Compute angle from trace
    trace = rot[0, 0] + rot[1, 1] + rot[2, 2]
    # Manual clipping for Numba compatibility (np.clip doesn't work with scalars in nopython mode)
    cos_angle = (trace - 1.0) / 2.0
    if cos_angle > 1.0:
        cos_angle = 1.0
    elif cos_angle < -1.0:
        cos_angle = -1.0
    angle = np.arccos(cos_angle)

    # Handle special cases
    if angle < 1e-10:  # Near identity
        return np.zeros(3, dtype=np.float64)

    if angle > np.pi - 1e-10:  # Near 180 degrees
        # Find largest diagonal element to extract axis
        if rot[0, 0] >= rot[1, 1] and rot[0, 0] >= rot[2, 2]:
            axis = np.array([
                np.sqrt((rot[0, 0] + 1.0) / 2.0),
                (rot[0, 1] + rot[1, 0]) / (2.0 * np.sqrt(2.0 * (rot[0, 0] + 1.0))),
                (rot[0, 2] + rot[2, 0]) / (2.0 * np.sqrt(2.0 * (rot[0, 0] + 1.0)))
            ])
        elif rot[1, 1] >= rot[2, 2]:
            axis = np.array([
                (rot[0, 1] + rot[1, 0]) / (2.0 * np.sqrt(2.0 * (rot[1, 1] + 1.0))),
                np.sqrt((rot[1, 1] + 1.0) / 2.0),
                (rot[1, 2] + rot[2, 1]) / (2.0 * np.sqrt(2.0 * (rot[1, 1] + 1.0)))
            ])
        else:
            axis = np.array([
                (rot[0, 2] + rot[2, 0]) / (2.0 * np.sqrt(2.0 * (rot[2, 2] + 1.0))),
                (rot[1, 2] + rot[2, 1]) / (2.0 * np.sqrt(2.0 * (rot[2, 2] + 1.0))),
                np.sqrt((rot[2, 2] + 1.0) / 2.0)
            ])
        return axis * angle

    # General case: extract axis from skew-symmetric part
    s = 2.0 * np.sin(angle)
    axis = np.array([
        rot[2, 1] - rot[1, 2],
        rot[0, 2] - rot[2, 0],
        rot[1, 0] - rot[0, 1]
    ]) / s

    return axis * angle

@njit(cache=True, fastmath=True)
def rotvec_to_matrix(rotvec):
    """Convert rotation vector (axis-angle) to rotation matrix - JIT compiled

    Args:
        rotvec: 3D rotation vector where direction is axis and magnitude is angle (radians)

    Returns:
        3x3 rotation matrix (numpy array)

    Note: Uses Rodriguez formula
    """
    angle = np.linalg.norm(rotvec)

    # Handle zero rotation
    if angle < 1e-10:
        return np.eye(3, dtype=np.float64)

    # Normalize to get axis
    axis = rotvec / angle

    # Rodrigues' formula: R = I + sin(θ)K + (1-cos(θ))K²
    # where K is the skew-symmetric matrix of the axis
    c = np.cos(angle)
    s = np.sin(angle)
    t = 1.0 - c

    x, y, z = axis

    # Build rotation matrix
    mat = np.empty((3, 3), dtype=np.float64)
    mat[0, 0] = t * x * x + c
    mat[0, 1] = t * x * y - s * z
    mat[0, 2] = t * x * z + s * y
    mat[1, 0] = t * x * y + s * z
    mat[1, 1] = t * y * y + c
    mat[1, 2] = t * y * z - s * x
    mat[2, 0] = t * x * z - s * y
    mat[2, 1] = t * y * z + s * x
    mat[2, 2] = t * z * z + c

    return mat

@njit(cache=True, fastmath=True)
def pose_to_transform(pos, quat, wxyz=False):
    """Convert position and quaternion to 4x4 transformation matrix - JIT compiled

    Args:
        pos: 3D position vector (numpy array)
        quat: Quaternion as [x, y, z, w] (default) or [w, x, y, z] (if wxyz=True)
        wxyz: If True, quat is [w, x, y, z]. If False (default), quat is [x, y, z, w]

    Returns:
        4x4 homogeneous transformation matrix (numpy array)

    Performance: Optimized to avoid unnecessary identity matrix initialization
    """
    # Get rotation matrix from quaternion
    rot = quat_to_matrix(quat, wxyz=wxyz)

    # Direct construction - faster than np.eye(4)
    transform = np.zeros((4, 4), dtype=np.float64)
    transform[:3, :3] = rot
    transform[:3, 3] = pos
    transform[3, 3] = 1.0

    return transform

@njit(cache=True, fastmath=True)
def transform_to_pose(transform, wxyz=False):
    """Convert 4x4 transformation matrix to position and quaternion - JIT compiled

    Args:
        transform: 4x4 homogeneous transformation matrix (numpy array)
        wxyz: If True, return quat as [w, x, y, z]. If False (default), return [x, y, z, w]

    Returns:
        Tuple of (pos, quat) where:
        - pos: 3D position vector
        - quat: Quaternion as [x, y, z, w] (default) or [w, x, y, z] (if wxyz=True)

    Performance: Optimized with explicit copy semantics
    """
    # Extract position (explicit copy to avoid aliasing)
    pos = transform[:3, 3].copy()

    # Extract rotation matrix
    rot = transform[:3, :3]

    # Convert rotation to quaternion
    quat = matrix_to_quat(rot, wxyz=wxyz)

    return pos, quat


# Warmup JIT on import (pre-compile both variants)
_warmup = np.eye(3)
_ = matrix_to_quat(_warmup, wxyz=False)
_ = matrix_to_quat(_warmup, wxyz=True)
_warmup_quat_xyzw = np.array([0.0, 0.0, 0.0, 1.0])
_warmup_quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0])
_ = quat_to_matrix(_warmup_quat_xyzw, wxyz=False)
_ = quat_to_matrix(_warmup_quat_wxyz, wxyz=True)
_warmup_rotvec = np.array([0.1, 0.2, 0.3])
_ = matrix_to_rotvec(_warmup)
_ = rotvec_to_matrix(_warmup_rotvec)
_warmup_pos = np.array([1.0, 2.0, 3.0])
_ = pose_to_transform(_warmup_pos, _warmup_quat_xyzw, wxyz=False)
_ = pose_to_transform(_warmup_pos, _warmup_quat_wxyz, wxyz=True)
_warmup_transform = np.eye(4)
_ = transform_to_pose(_warmup_transform, wxyz=False)
_ = transform_to_pose(_warmup_transform, wxyz=True)

if __name__ == "__main__":
    import time

    print("Testing Optimized Rotation Utilities")
    print("=" * 60)

    # Test 1: Identity conversions
    print("\n[1/4] Identity matrix conversions")
    I = np.eye(3)
    assert np.allclose(matrix_to_quat(I, wxyz=False), [0, 0, 0, 1])
    assert np.allclose(matrix_to_quat(I, wxyz=True), [1, 0, 0, 0])
    assert np.allclose(matrix_to_rotvec(I), [0, 0, 0])
    print("      ✓ Pass")

    # Test 2: Known rotation (90° Z-axis)
    print("[2/4] Known rotation round-trips (90° Z-axis)")
    R_z90 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)

    q = matrix_to_quat(R_z90, wxyz=False)
    assert np.allclose(R_z90, quat_to_matrix(q, wxyz=False), atol=1e-10)

    rv = matrix_to_rotvec(R_z90)
    assert np.allclose(R_z90, rotvec_to_matrix(rv), atol=1e-10)
    print("      ✓ Pass")

    # Test 3: Random rotations accuracy
    print("[3/4] Random rotations (100 samples)")
    np.random.seed(42)
    errors = []
    for _ in range(100):
        q_rand = np.random.randn(4)
        q_rand /= np.linalg.norm(q_rand)
        R = quat_to_matrix(q_rand, wxyz=False)

        # Quat round-trip
        errors.append(np.max(np.abs(R - quat_to_matrix(matrix_to_quat(R, wxyz=False), wxyz=False))))
        # RotVec round-trip
        errors.append(np.max(np.abs(R - rotvec_to_matrix(matrix_to_rotvec(R)))))

    max_err = max(errors)
    print(f"      Max error: {max_err:.2e}")
    assert max_err < 1e-10, f"Error too large: {max_err}"
    print("      ✓ Pass")

    # Test 4: Performance vs scipy
    print("[4/4] Performance benchmark (1000 samples)")
    try:
        from scipy.spatial.transform import Rotation

        # Generate test data
        np.random.seed(123)
        test_mats = []
        for _ in range(1000):
            q = np.random.randn(4)
            q /= np.linalg.norm(q)
            test_mats.append(quat_to_matrix(q, wxyz=False))

        # Benchmark matrix_to_quat
        t0 = time.perf_counter()
        for R in test_mats: _ = matrix_to_quat(R, wxyz=False)
        t_ours = time.perf_counter() - t0

        t0 = time.perf_counter()
        for R in test_mats: _ = Rotation.from_matrix(R).as_quat()
        t_scipy = time.perf_counter() - t0

        print(f"      matrix_to_quat: {t_scipy/t_ours:.1f}x faster than scipy")

        # Benchmark matrix_to_rotvec
        t0 = time.perf_counter()
        for R in test_mats: _ = matrix_to_rotvec(R)
        t_ours = time.perf_counter() - t0

        t0 = time.perf_counter()
        for R in test_mats: _ = Rotation.from_matrix(R).as_rotvec()
        t_scipy = time.perf_counter() - t0

        print(f"      matrix_to_rotvec: {t_scipy/t_ours:.1f}x faster than scipy")
        print("      ✓ Pass")

    except ImportError:
        print("      (scipy not installed, skipping comparison)")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
