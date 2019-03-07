"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import functools
import logging
import time

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def split_extrinsics(extrinsics):
    extrinsics = np.array(extrinsics)
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
        np.matmul(cv2.Rodrigues(rotation)[0], p) + translation.ravel()
        for p in pts_3d_world.reshape(-1, 3)
    ]
    pts_3d_cam = np.array(pts_3d_cam)

    return pts_3d_cam


def convert_extrinsic_to_matrix(extrinsics):
    rotation, translation = split_extrinsics(extrinsics)
    extrinsic_matrix = np.eye(4, dtype=np.float32)
    extrinsic_matrix[0:3, 0:3] = cv2.Rodrigues(rotation)[0]
    extrinsic_matrix[0:3, 3] = translation
    return extrinsic_matrix


def convert_matrix_to_extrinsic(extrinsic_matrix):
    rotation = cv2.Rodrigues(extrinsic_matrix[0:3, 0:3])[0]
    translation = extrinsic_matrix[0:3, 3]
    return merge_extrinsics(rotation, translation)


def get_camera_pose(camera_extrinsics):
    if camera_extrinsics is None:
        return get_none_camera_extrinsics()

    camera_extrinsics = np.array(camera_extrinsics)
    rotation_ext, translation_ext = split_extrinsics(camera_extrinsics)
    rotation_pose = -rotation_ext
    translation_pose = np.matmul(-cv2.Rodrigues(rotation_ext)[0].T, translation_ext)
    camera_pose = merge_extrinsics(rotation_pose, translation_pose)
    return camera_pose


def convert_marker_extrinsics_to_points_3d(marker_extrinsics):
    mat = convert_extrinsic_to_matrix(marker_extrinsics)
    marker_transformed_h = np.matmul(mat, get_marker_points_4d_origin().T)
    marker_points_3d = cv2.convertPointsFromHomogeneous(marker_transformed_h.T)
    marker_points_3d.shape = 4, 3

    return marker_points_3d


def find_origin_marker_id(marker_id_to_extrinsics):
    for marker_id, extrinsics in marker_id_to_extrinsics.items():
        if np.allclose(extrinsics, get_marker_extrinsics_origin()):
            return marker_id
    return None


def get_marker_points_3d_origin():
    return np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32)


def get_marker_points_4d_origin():
    return np.array(
        [[0, 0, 0, 1], [1, 0, 0, 1], [1, 1, 0, 1], [0, 1, 0, 1]], dtype=np.float32
    )


def get_marker_extrinsics_origin():
    return np.array([0, 0, 0, 0, 0, 0.0], dtype=np.float32)


def get_none_camera_extrinsics():
    return np.full((6,), np.nan)


def timer(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        t1 = time.perf_counter()
        value = func(*args, **kwargs)
        t2 = time.perf_counter()
        run_time = t2 - t1
        if run_time > 1:
            logger.info("{0} took {1:.2f} s".format(func.__name__, run_time))
        elif run_time > 1e-3:
            logger.info("{0} took {1:.2f} ms".format(func.__name__, run_time * 1e3))
        else:
            logger.info("{0} took {1:.2f} µs".format(func.__name__, run_time * 1e6))

        return value

    return wrapper_timer
