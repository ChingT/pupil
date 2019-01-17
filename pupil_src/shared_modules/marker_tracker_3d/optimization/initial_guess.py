import random

import cv2
import numpy as np

from marker_tracker_3d import localize_camera
from marker_tracker_3d import utils


def get(
    camera_model,
    frame_indices,
    marker_indices,
    markers_points_2d_detected,
    camera_extrinsics_prv_dict,
    marker_extrinsics_prv_dict,
):
    """
    get marker_and camera extrinsics initial guess for bundle adjustment

    :param camera_model: 
    :param frame_indices: array_like with shape (n, ), frame indices
    :param marker_indices: array_like with shape (n, ), marker indices
    :param markers_points_2d_detected: np.ndarray with shape (n x 4 x 2),
    markers points from image
    :param camera_extrinsics_prv_dict: dict, previous camera extrinsics
    :param marker_extrinsics_prv_dict: dict, previous marker extrinsics
    """

    camera_extrinsics_init_dict = camera_extrinsics_prv_dict
    marker_extrinsics_init_dict = marker_extrinsics_prv_dict

    n_frames = len(set(frame_indices))
    n_markers = len(set(marker_indices))

    # The function _calculate_extrinsics calculates camera extrinsics and marker
    # extrinsics iteratively. It is possible that not all of them can be calculated
    # after one run of _calculate_extrinsics, so we need to run it twice.
    for _ in range(2):
        camera_extrinsics_init_dict, marker_extrinsics_init_dict = _calculate_extrinsics(
            camera_model,
            frame_indices,
            marker_indices,
            markers_points_2d_detected,
            camera_extrinsics_init_dict,
            marker_extrinsics_init_dict,
        )

        try:
            camera_extrinsics_init_array = np.array(
                [camera_extrinsics_init_dict[i] for i in range(n_frames)]
            )
            marker_extrinsics_init_array = np.array(
                [marker_extrinsics_init_dict[i] for i in range(n_markers)]
            )
        except KeyError:
            pass
        else:
            return camera_extrinsics_init_array, marker_extrinsics_init_array

    return None, None


def _calculate_extrinsics(
    camera_model,
    frame_indices,
    marker_indices,
    markers_points_2d_detected,
    camera_extrinsics_init_dict,
    marker_extrinsics_init_dict,
):
    """ The function calculates camera extrinsics based on the known marker extrinsics
    and then calculates marker extrinsics based on the known camera extrinsics.
    """

    camera_extrinsics_init_dict = _get_camera_extrinsics_init_dict(
        camera_model,
        frame_indices,
        marker_indices,
        markers_points_2d_detected,
        camera_extrinsics_init_dict,
        marker_extrinsics_init_dict,
    )
    marker_extrinsics_init_dict = _get_marker_extrinsics_init(
        camera_model,
        frame_indices,
        marker_indices,
        markers_points_2d_detected,
        camera_extrinsics_init_dict,
        marker_extrinsics_init_dict,
    )
    return camera_extrinsics_init_dict, marker_extrinsics_init_dict


def _get_camera_extrinsics_init_dict(
    camera_model,
    frame_indices,
    marker_indices,
    markers_points_2d_detected,
    camera_extrinsics_init_dict,
    marker_extrinsics_init_dict,
):
    frames_index_not_computed = set(frame_indices) - set(
        camera_extrinsics_init_dict.keys()
    )

    for frame_index in frames_index_not_computed:
        data = (
            frame_index,
            frame_indices,
            marker_indices,
            markers_points_2d_detected,
            marker_extrinsics_init_dict,
        )
        camera_extrinsics = localize_camera.get(camera_model, data)

        if camera_extrinsics is not None:
            camera_extrinsics_init_dict[frame_index] = camera_extrinsics

    return camera_extrinsics_init_dict


def _get_marker_extrinsics_init(
    camera_model,
    frame_indices,
    marker_indices,
    markers_points_2d_detected,
    camera_extrinsics_prv_dict,
    marker_extrinsics_prv_dict,
):
    marker_extrinsics_init = marker_extrinsics_prv_dict
    markers_index_not_computed = set(marker_indices) - set(
        marker_extrinsics_init.keys()
    )

    for marker_index in markers_index_not_computed:
        frames_index_available = list(
            set(camera_extrinsics_prv_dict.keys())
            & set(frame_indices[marker_indices == marker_index])
        )
        try:
            camera_index_0, camera_index_1 = random.sample(frames_index_available, 2)
        except ValueError:
            pass
        else:
            data_for_run_triangulation = (
                frame_indices,
                marker_indices,
                markers_points_2d_detected,
                camera_extrinsics_prv_dict,
                camera_index_0,
                camera_index_1,
                marker_index,
            )
            marker_extrinsics_init[marker_index] = _calculate_marker_extrinsics(
                camera_model, data_for_run_triangulation
            )

    return marker_extrinsics_init


def _calculate_marker_extrinsics(camera_model, data_for_run_triangulation):
    data_for_cv2_triangulate_points = _prepare_data_for_cv2_triangulate_points(
        camera_model, *data_for_run_triangulation
    )

    points_4d = cv2.triangulatePoints(*data_for_cv2_triangulate_points)

    return _convert_to_marker_extrinsics(points_4d)


def _prepare_data_for_cv2_triangulate_points(
    camera_model,
    frame_indices,
    marker_indices,
    markers_points_2d_detected,
    camera_extrinsics,
    frame_index_0,
    frame_index_1,
    marker_index,
):
    proj_mat1 = utils.get_extrinsic_matrix(camera_extrinsics[frame_index_0])[:3, :4]
    proj_mat2 = utils.get_extrinsic_matrix(camera_extrinsics[frame_index_1])[:3, :4]

    points1 = markers_points_2d_detected[
        np.bitwise_and(frame_indices == frame_index_0, marker_indices == marker_index)
    ]

    points2 = markers_points_2d_detected[
        np.bitwise_and(frame_indices == frame_index_1, marker_indices == marker_index)
    ]
    undistort_points1 = camera_model.undistortPoints(points1)
    undistort_points2 = camera_model.undistortPoints(points2)

    return proj_mat1, proj_mat2, undistort_points1, undistort_points2


def _convert_to_marker_extrinsics(points_4d):
    marker_points_3d = cv2.convertPointsFromHomogeneous(points_4d.T).reshape(4, 3)
    marker_extrinsics = utils.marker_points_3d_to_extrinsics(marker_points_3d)

    return marker_extrinsics
