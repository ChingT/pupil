import functools
import logging
import os
import time

import cv2
import numpy as np

import recorder
from marker_tracker_3d import math

logger = logging.getLogger(__name__)


def get_marker_vertex_coord(marker_extrinsics, camera_model):
    marker_extrinsics = np.array(marker_extrinsics)
    marker_points_3d = camera_model.params_to_points_3d(marker_extrinsics)
    marker_points_3d.shape = 4, 3
    return marker_points_3d


def split_param(param):
    assert param.size == 6
    return param.ravel()[0:3], param.ravel()[3:6]


def merge_param(rvec, tvec):
    assert rvec.size == 3 and tvec.size == 3
    return np.concatenate((rvec.ravel(), tvec.ravel()))


def to_camera_coordinate(pts_3d_world, rvec, tvec):
    pts_3d_cam = [
        cv2.Rodrigues(rvec)[0] @ p + tvec.ravel() for p in pts_3d_world.reshape(-1, 3)
    ]
    pts_3d_cam = np.array(pts_3d_cam)

    return pts_3d_cam


def check_camera_extrinsics(pts_3d_world, rvec, tvec):
    assert rvec.size == 3 and tvec.size == 3
    if (np.abs(rvec) > np.pi * 2).any():
        return False

    pts_3d_camera = to_camera_coordinate(pts_3d_world, rvec, tvec)
    if (pts_3d_camera.reshape(-1, 3)[:, 2] < 1).any():
        return False

    return True


def get_extrinsic_matrix(camera_extrinsics):
    rvec, tvec = split_param(camera_extrinsics)
    extrinsic_matrix = np.eye(4, dtype=np.float32)
    extrinsic_matrix[0:3, 0:3] = cv2.Rodrigues(rvec)[0]
    extrinsic_matrix[0:3, 3] = tvec
    return extrinsic_matrix


def get_camera_pose_matrix(camera_extrinsics):
    rvec, tvec = split_param(camera_extrinsics)
    camera_pose_matrix = np.eye(4, dtype=np.float32)
    camera_pose_matrix[0:3, 0:3] = cv2.Rodrigues(rvec)[0].T
    camera_pose_matrix[0:3, 3] = -camera_pose_matrix[0:3, 0:3] @ tvec
    return camera_pose_matrix


def get_camera_trace(camera_pose_matrix):
    return camera_pose_matrix[0:3, 3]


def get_camera_trace_from_camera_extrinsics(camera_extrinsics):
    rvec, tvec = split_param(camera_extrinsics)
    return -cv2.Rodrigues(rvec)[0].T @ tvec


def compute_camera_trace_distance(previous_camera_trace, current_camera_trace):
    return np.linalg.norm(current_camera_trace - previous_camera_trace)


def params_to_points_3d(params):
    params = np.asarray(params).reshape(-1, 6)
    marker_points_3d = []
    for param in params:
        mat = get_extrinsic_matrix(param)
        marker_transformed_h = mat @ marker_df_h.T
        marker_transformed = cv2.convertPointsFromHomogeneous(
            marker_transformed_h.T
        ).reshape(4, 3)
        marker_points_3d.append(marker_transformed)

    marker_points_3d = np.array(marker_points_3d)
    return marker_points_3d


def point_3d_to_param(marker_points_3d):
    rotation_matrix, translation_vector, _ = math.svdt(A=marker_df, B=marker_points_3d)

    rvec = cv2.Rodrigues(rotation_matrix)[0]
    tvec = translation_vector
    marker_extrinsics = merge_param(rvec, tvec)
    return marker_extrinsics


marker_df = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32)
marker_df_h = cv2.convertPointsToHomogeneous(marker_df).reshape(4, 4)
marker_extrinsics_origin = point_3d_to_param(marker_df)


def save_array(path, file_name, data):
    try:
        np.save(os.path.join(path, file_name), data)
    except FileNotFoundError:
        os.makedirs(path)
        np.save(os.path.join(path, file_name), data)


def save_dict_to_pkl(path, file_name, data):
    import pickle

    try:
        f = open(os.path.join(path, file_name), "wb")
    except FileNotFoundError:
        os.makedirs(path)
        f = open(os.path.join(path, file_name), "wb")

    pickle.dump(data, f)
    f.close()


def timer(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        t1 = time.perf_counter()
        value = func(*args, **kwargs)
        t2 = time.perf_counter()
        run_time = t2 - t1
        if run_time > 1:
            logger.debug("{0} took {1:.2f} s".format(func.__name__, run_time))
        elif run_time > 1e-3:
            logger.debug("{0} took {1:.2f} ms".format(func.__name__, run_time * 1e3))
        else:
            logger.debug("{0} took {1:.2f} µs".format(func.__name__, run_time * 1e6))

        return value

    return wrapper_timer


def get_save_path(root):
    now = recorder.get_auto_name()
    counter = 0
    while True:
        save_path = os.path.join(root, now, "{:03d}".format(counter))
        if os.path.exists(save_path):
            counter += 1
        else:
            break
    return save_path
