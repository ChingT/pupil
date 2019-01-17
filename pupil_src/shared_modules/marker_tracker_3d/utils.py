import functools
import logging
import time

import cv2
import numpy as np

from marker_tracker_3d import math

logger = logging.getLogger(__name__)


def split_extrinsics(extrinsics):
    assert extrinsics.size == 6
    # extrinsics could be of shape (6,) or (1, 6), so ravel() is needed.
    rotation = extrinsics.ravel()[0:3]
    translation = extrinsics.ravel()[3:6]
    return rotation, translation


def merge_extrinsics(rotation, translation):
    assert rotation.size == 3 and translation.size == 3
    # rotation and translation could be of shape (3,) or (1, 3), so ravel() is needed.
    extrinsics = np.concatenate((rotation.ravel(), translation.ravel()))
    return extrinsics


def to_camera_coordinate(pts_3d_world, rotation, translation):
    pts_3d_cam = [
        cv2.Rodrigues(rotation)[0] @ p + translation.ravel()
        for p in pts_3d_world.reshape(-1, 3)
    ]
    pts_3d_cam = np.array(pts_3d_cam)

    return pts_3d_cam


def get_extrinsic_matrix(camera_extrinsics):
    rotation, translation = split_extrinsics(camera_extrinsics)
    extrinsic_matrix = np.eye(4, dtype=np.float32)
    extrinsic_matrix[0:3, 0:3] = cv2.Rodrigues(rotation)[0]
    extrinsic_matrix[0:3, 3] = translation
    return extrinsic_matrix


def get_camera_pose_matrix(camera_extrinsics):
    rotation, translation = split_extrinsics(camera_extrinsics)
    camera_pose_matrix = np.eye(4, dtype=np.float32)
    camera_pose_matrix[0:3, 0:3] = cv2.Rodrigues(rotation)[0].T
    camera_pose_matrix[0:3, 3] = -camera_pose_matrix[0:3, 0:3] @ translation
    return camera_pose_matrix


def get_camera_trace(camera_pose_matrix):
    return camera_pose_matrix[0:3, 3]


def get_camera_trace_from_camera_extrinsics(camera_extrinsics):
    rotation, translation = split_extrinsics(camera_extrinsics)
    camera_trace = -cv2.Rodrigues(rotation)[0].T @ translation
    return camera_trace


def extrinsics_to_marker_points_3d(marker_extrinsics):
    marker_extrinsics = np.asarray(marker_extrinsics).reshape(-1, 6)
    marker_points_4d_origin = cv2.convertPointsToHomogeneous(
        get_marker_points_3d_origin()
    ).reshape(4, 4)

    marker_points_3d = []
    for extrinsics in marker_extrinsics:
        mat = get_extrinsic_matrix(extrinsics)
        marker_transformed_h = mat @ marker_points_4d_origin.T
        marker_transformed = cv2.convertPointsFromHomogeneous(
            marker_transformed_h.T
        ).reshape(4, 3)
        marker_points_3d.append(marker_transformed)

    marker_points_3d = np.array(marker_points_3d)
    return marker_points_3d


def marker_points_3d_to_extrinsics(marker_points_3d):
    rotation_matrix, translation, _ = math.svdt(
        A=get_marker_points_3d_origin(), B=marker_points_3d
    )
    rotation = cv2.Rodrigues(rotation_matrix)[0]
    marker_extrinsics = merge_extrinsics(rotation, translation)
    return marker_extrinsics


def get_marker_points_3d_origin():
    marker_points_3d_origin = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32
    )
    return marker_points_3d_origin


def get_marker_extrinsics_origin():
    marker_extrinsics_origin = marker_points_3d_to_extrinsics(
        get_marker_points_3d_origin()
    )
    return marker_extrinsics_origin


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
