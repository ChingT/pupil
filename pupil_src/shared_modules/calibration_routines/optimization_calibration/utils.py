"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2020 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import functools
import json
import os
import time

import cv2
import numpy as np


def transform_points_by_extrinsic(points_3d_cam1, extrinsic_cam2_cam1):
    """
    Transform 3d points from cam1 coordinate to cam2 coordinate

    :param points_3d_cam1: 3d points in cam1 coordinate, shape: (N x 3)
    :param extrinsic_cam2_cam1: extrinsic of cam2 in cam1 coordinate, shape: (6,)
    :return: 3d points in cam2 coordinate, shape: (N x 3)
    """

    rotation_cam2_cam1, translation_cam2_cam1 = split_extrinsic(extrinsic_cam2_cam1)
    points_3d_cam1 = np.asarray(points_3d_cam1, dtype=np.float64)
    points_3d_cam1.shape = -1, 3
    rotation_matrix_cam2_cam1 = cv2.Rodrigues(rotation_cam2_cam1)[0]
    points_3d_cam2 = (
        np.dot(rotation_matrix_cam2_cam1, points_3d_cam1.T).T + translation_cam2_cam1
    )
    return points_3d_cam2


def transform_points_by_pose(points_3d_cam1, pose_cam2_cam1):
    """
    Transform 3d points from cam1 coordinate to cam2 coordinate

    :param points_3d_cam1: 3d points in cam1 coordinate, shape: (N x 3)
    :param pose_cam2_cam1: camera pose of cam2 in cam1 coordinate, shape: (6,)
    :return: 3d points in cam2 coordinate, shape: (N x 3)
    """

    rotation_cam2_cam1, translation_cam2_cam1 = split_extrinsic(pose_cam2_cam1)
    points_3d_cam1 = np.asarray(points_3d_cam1, dtype=np.float64)
    points_3d_cam1.shape = -1, 3

    rotation_matrix_cam2_cam1 = cv2.Rodrigues(rotation_cam2_cam1)[0]
    rotation_matrix_cam1_cam2 = rotation_matrix_cam2_cam1.T
    translation_cam1_cam2 = np.dot(-rotation_matrix_cam1_cam2, translation_cam2_cam1)
    points_3d_cam2 = (
        np.dot(rotation_matrix_cam1_cam2, points_3d_cam1.T).T + translation_cam1_cam2
    )
    return points_3d_cam2


def inverse_extrinsic(extrinsic):
    rotation_ext, translation_ext = split_extrinsic(extrinsic)
    rotation_inv = -rotation_ext
    translation_inv = np.dot(-cv2.Rodrigues(rotation_inv)[0], translation_ext)
    return merge_extrinsic(rotation_inv, translation_inv)


def convert_extrinsic_to_matrix(extrinsic):
    rotation, translation = split_extrinsic(extrinsic)
    extrinsic_matrix = np.eye(4, dtype=np.float64)
    extrinsic_matrix[0:3, 0:3] = cv2.Rodrigues(rotation)[0]
    extrinsic_matrix[0:3, 3] = translation
    return extrinsic_matrix


def convert_matrix_to_extrinsic(extrinsic_matrix):
    extrinsic_matrix = np.asarray(extrinsic_matrix, dtype=np.float64)
    rotation = cv2.Rodrigues(extrinsic_matrix[0:3, 0:3])[0]
    translation = extrinsic_matrix[0:3, 3]
    return merge_extrinsic(rotation, translation)


def split_extrinsic(extrinsic):
    extrinsic = np.asarray(extrinsic, dtype=np.float64)
    assert extrinsic.size == 6
    rotation = extrinsic.ravel()[0:3]
    translation = extrinsic.ravel()[3:6]
    return rotation, translation


def merge_extrinsic(rotation, translation):
    assert rotation.size == 3 and translation.size == 3
    extrinsic = np.concatenate((rotation.ravel(), translation.ravel()))
    return extrinsic


def extrinsic_mul(*args):
    return convert_matrix_to_extrinsic(
        multiple_matmul(*[convert_extrinsic_to_matrix(extrinsic) for extrinsic in args])
    )


def multiple_matmul(*args):
    assert args[0].shape in [(3, 3), (4, 4)]
    return functools.reduce(lambda x, y: np.dot(x, y), args)


def find_rigid_transform(A, B):
    """Calculates the transformation between two coordinate systems using SVD.
    This function determines the rotation matrix (R) and the translation vector
    (L) for a rigid body after the following transformation [1]_, [2]_:
    B = R*A + L + err, where A and B represents the rigid body in different instants
    and err is an aleatory noise (which should be zero for a perfect rigid body).

    Adapted from: https://github.com/demotu/BMC/blob/master/functions/svdt.py
    """

    assert A.shape == B.shape and A.ndim == 2 and A.shape[1] == 3

    A_centroid = np.mean(A, axis=0)
    B_centroid = np.mean(B, axis=0)
    M = np.dot((B - B_centroid).T, (A - A_centroid))
    U, S, Vt = np.linalg.svd(M)

    rotation_matrix = np.dot(
        U, np.dot(np.diag([1, 1, np.linalg.det(np.dot(U, Vt))]), Vt)
    )

    translation_vector = B_centroid - np.dot(rotation_matrix, A_centroid)
    return rotation_matrix, translation_vector


def timer(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        t1 = time.perf_counter()
        output = func(*args, **kwargs)
        t2 = time.perf_counter()
        run_time = t2 - t1

        if run_time > 1:
            unit = "s"
            show_time = run_time
        elif run_time > 1e-3:
            unit = "ms"
            show_time = run_time * 1e3
        else:
            unit = "µs"
            show_time = run_time * 1e6

        print(f"{func.__name__} took {show_time:.2f} {unit}")

        file_name = "binocular-" + func.__module__.split(".")[-1]
        function_name = func.__name__
        save_time(file_name, function_name, kwargs, run_time)
        if isinstance(output, dict) and "residual" in output:
            save_time(file_name, "residual", kwargs, output["residual"])
        elif function_name == "finish_calibration":
            save_time(file_name, "accuracy", kwargs, float(output))

        return output

    return wrapper_timer


def save_time(file_name, function_name, kwargs, number):
    root = "/cluster/users/Ching/codebase/calibration-refactor/timer_functions"
    os.makedirs(root, exist_ok=True)

    path = os.path.join(root, f"{file_name}.json")
    if os.path.isfile(path):
        with open(path, "r") as file:
            time_dict = json.load(file)
    else:
        time_dict = {}

    if function_name not in time_dict:
        time_dict[function_name] = {}

    key_words = {
        "initial_points",
        "matched_monocular_data",
        "matched_binocular_data",
        "result",
    }
    kwargs_keys = set(kwargs.keys())
    try:
        key_word = (key_words & kwargs_keys).pop()
    except KeyError:
        return

    if key_word == "result":
        n_points = str(len(kwargs["result"]["cal_points_3d"]))
    else:
        n_points = str(len(kwargs[key_word]))
    if n_points in time_dict[function_name]:
        time_dict[function_name][n_points].append(number)
    else:
        time_dict[function_name][n_points] = [number]

    with open(path, "w") as file:
        json.dump(time_dict, file)


def draw_cpu_time(function_name, _locals, **kwargs):
    import cProfile
    import subprocess

    kwargs_list = ", ".join(kwargs.keys())
    statement = f"{function_name}({kwargs_list})"
    cProfile.runctx(statement, kwargs, _locals, "temp.pstats")
    gprof2dot_loc = (
        "/cluster/users/Ching/codebase/pupil/pupil_src/shared_modules/gprof2dot.py"
    )
    folder = "/cluster/users/Ching/codebase/calibration-refactor"
    now = time.strftime("%y_%m_%d_%H_%M_%S", time.localtime(time.time()))
    file_path = os.path.join(folder, "time", f"cpu_time-{function_name}-{now}.png")
    subprocess.call(
        f"python3 {gprof2dot_loc} -f pstats temp.pstats | dot -Tpng -o {file_path}",
        shell=True,
    )


def AngleAxisToQuaternion(angle_axis):
    a0, a1, a2 = angle_axis
    theta_squared = a0 * a0 + a1 * a1 + a2 * a2

    if theta_squared > 0.0:
        theta = np.sqrt(theta_squared)
        half_theta = theta * 0.5
        k = np.sin(half_theta) / theta
        quaternion = np.array([np.cos(half_theta), a0 * k, a1 * k, a2 * k])
    else:
        k = 0.5
        quaternion = np.array([1.0, a0 * k, a1 * k, a2 * k])

    return quaternion


def QuaternionToAngleAxis(quaternion):
    _, q1, q2, q3 = quaternion
    sin_squared = q1 * q1 + q2 * q2 + q3 * q3

    if sin_squared > 0.0:
        sin_theta = np.sqrt(sin_squared)
        k = 2.0 * np.arctan2(sin_theta, quaternion[0]) / sin_theta
        angle_axis = np.array([q1, q2, q3]) * k
    else:
        k = 2.0
        angle_axis = np.array([q1, q2, q3]) * k

    return angle_axis


def QuaternionToScaledRotation(q):
    a, b, c, d = q

    aa = a * a
    ab = a * b
    ac = a * c
    ad = a * d
    bb = b * b
    bc = b * c
    bd = b * d
    cc = c * c
    cd = c * d
    dd = d * d

    R = np.zeros(9, dtype=np.float64)
    R[0] = aa + bb - cc - dd
    R[1] = 2 * (bc - ad)
    R[2] = 2 * (ac + bd)
    R[3] = 2 * (ad + bc)
    R[4] = aa - bb + cc - dd
    R[5] = 2 * (cd - ab)
    R[6] = 2 * (bd - ac)
    R[7] = 2 * (ab + cd)
    R[8] = aa - bb - cc + dd
    R.shape = 3, 3
    return R


def QuaternionToRotation(q):
    R = QuaternionToScaledRotation(q)

    normalizer = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]
    assert normalizer != 0

    R /= normalizer
    return R


def RotationMatrixToAngleAxis(R):
    R = R[:3, :3]
    assert R.shape == (3, 3)
    R = R.ravel()
    angle_axis = np.array([R[5] - R[7], R[6] - R[2], R[1] - R[3]])

    costheta = min((max(((R[0] + R[4] + R[8] - 1.0) / 2.0, -1.0)), 1.0))
    sintheta = min((np.sqrt(np.dot(angle_axis, angle_axis)) / 2.0, 1.0))
    theta = np.arctan2(sintheta, costheta)

    kThreshold = 1e-12
    if sintheta > kThreshold:
        r = theta / (2.0 * sintheta)
        angle_axis *= r
        return angle_axis

    if costheta > 0.0:
        angle_axis *= 0.5
        return angle_axis

    inv_one_minus_costheta = 1.0 / (1.0 - costheta)

    for i in range(3):
        angle_axis[i] = theta * np.sqrt((R[i * 4] - costheta) * inv_one_minus_costheta)
        if (sintheta < 0 and angle_axis[i] > 0) or (sintheta > 0 and angle_axis[i] < 0):
            angle_axis[i] *= -1
    return angle_axis


def AngleAxisToRotationMatrix(angle_axis):
    angle_axis = np.array(angle_axis, dtype=np.float64).reshape(3)
    theta2 = np.dot(angle_axis, angle_axis)
    R = np.zeros(9)
    if theta2 > 0.0:
        theta = np.sqrt(theta2)
        wx, wy, wz = angle_axis / theta

        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        R[0] = costheta + wx * wx * (1.0 - costheta)
        R[1] = wz * sintheta + wx * wy * (1.0 - costheta)
        R[2] = -wy * sintheta + wx * wz * (1.0 - costheta)
        R[3] = wx * wy * (1.0 - costheta) - wz * sintheta
        R[4] = costheta + wy * wy * (1.0 - costheta)
        R[5] = wx * sintheta + wy * wz * (1.0 - costheta)
        R[6] = wy * sintheta + wx * wz * (1.0 - costheta)
        R[7] = -wx * sintheta + wy * wz * (1.0 - costheta)
        R[8] = costheta + wz * wz * (1.0 - costheta)
    else:
        R[0] = 1.0
        R[1] = -angle_axis[2]
        R[2] = angle_axis[1]
        R[3] = angle_axis[2]
        R[4] = 1.0
        R[5] = -angle_axis[0]
        R[6] = -angle_axis[1]
        R[7] = angle_axis[0]
        R[8] = 1.0

    R.shape = 3, 3
    return R


def calculate_eye_camera_to_world(rotation, translation, sphere_pos):
    # eye_camera_to_world_matrix is the camera pose of eye in world coordinate
    eye_camera_to_world_matrix = np.eye(4)
    eye_camera_to_world_matrix[0:3, 0:3] = cv2.Rodrigues(rotation)[0]
    eye_camera_to_world_matrix[0:3, 3] = transform_points_by_extrinsic(
        -np.asarray(sphere_pos), merge_extrinsic(rotation, translation)
    )
    return eye_camera_to_world_matrix


def nearest_linepoints_to_points(ref_points, lines):
    p1, p2 = lines
    direction = p2 - p1
    denom = np.linalg.norm(direction, axis=1)
    delta = np.diag(np.dot(ref_points - p1, direction.T)) / (denom * denom)
    nearest_linepoints = p1 + direction * delta[:, np.newaxis]
    return nearest_linepoints


def calculate_nearest_points_to_lines(
    observed_normals, poses_in_world, points_in_world
):
    all_nearest_points = []
    for observations, pose in zip(observed_normals, poses_in_world):
        lines_start = transform_points_by_extrinsic(np.zeros(3), pose)
        lines_end = transform_points_by_extrinsic(observations, pose)
        nearest_points = nearest_linepoints_to_points(
            points_in_world, (lines_start, lines_end)
        )
        all_nearest_points.append(nearest_points)

    return all_nearest_points
