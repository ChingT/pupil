import collections
import os

import cv2
import numpy as np

from marker_tracker_3d import math

DataForOptimization = collections.namedtuple(
    "DataForOptimization",
    [
        "camera_indices",
        "marker_indices",
        "markers_points_2d_detected",
        "camera_extrinsics_prv",
        "marker_extrinsics_prv",
    ],
)

ResultOfOptimization = collections.namedtuple(
    "ResultOfOptimization",
    [
        "camera_extrinsics_opt",
        "marker_extrinsics_opt",
        "camera_keys_failed",
        "marker_keys_failed",
    ],
)


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
    if (np.abs(rvec) > np.pi * 2).any():
        return False

    pts_3d_camera = to_camera_coordinate(pts_3d_world, rvec, tvec)
    if (pts_3d_camera.reshape(-1, 3)[:, 2] < 1).any():
        return False

    return True


def params_to_points_3d(params):
    params = np.asarray(params).reshape(-1, 6)
    marker_points_3d = list()
    for param in params:
        rvec, tvec = split_param(param)
        mat = np.eye(4, dtype=np.float32)
        mat[0:3, 0:3] = cv2.Rodrigues(rvec)[0]
        mat[0:3, 3] = tvec
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


marker_df = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float)
marker_df_h = cv2.convertPointsToHomogeneous(marker_df).reshape(4, 4)
marker_extrinsics_origin = point_3d_to_param(marker_df)


# For experiments
def save_params_dicts(save_path, dicts):
    if not os.path.exists(os.path.join(save_path)):
        os.makedirs(os.path.join(save_path))
    for k, v in dicts.items():
        if isinstance(v, dict):
            _save_dict_to_pkl(v, os.path.join(save_path, k))
        elif isinstance(v, np.ndarray) or isinstance(v, list):
            np.save(os.path.join(save_path, k), v)


def _save_dict_to_pkl(d, dict_name):
    import pickle

    f = open(dict_name, "wb")
    pickle.dump(d, f)
    f.close()
