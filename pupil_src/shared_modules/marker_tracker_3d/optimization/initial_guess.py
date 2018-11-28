import random

import cv2
import numpy as np

from marker_tracker_3d import math
from marker_tracker_3d import utils


class InitialGuess:
    def __init__(self, camera_model):
        """ get marker_and camera extrinsics initial guess for bundle adjustment """

        self.camera_model = camera_model

    def get(
        self,
        camera_indices,
        marker_indices,
        markers_points_2d_detected,
        camera_extrinsics_prv,
        marker_extrinsics_prv,
    ):
        """
        calculate initial guess of marker and camera poses

        :param camera_indices: array_like with shape (n, ), camera indices
        :param marker_indices: array_like with shape (n, ), marker indices
        :param markers_points_2d_detected: np.ndarray with shape (n x 4 x 2), markers points from image
        :param camera_extrinsics_prv: dict, previous camera extrinsics
        :param marker_extrinsics_prv: dict, previous marker extrinsics
        """

        camera_extrinsics_init = self._get_camera_extrinsics_initial_guess(
            camera_extrinsics_prv
        )
        marker_extrinsics_init = self._get_marker_extrinsics_initial_guess(
            camera_indices,
            marker_indices,
            markers_points_2d_detected,
            camera_extrinsics_prv,
            marker_extrinsics_prv,
        )
        return camera_extrinsics_init, marker_extrinsics_init

    @staticmethod
    def _get_camera_extrinsics_initial_guess(camera_extrinsics_prv):
        # no need to calculate cameras_extrinsics initial guess since we have got them when picking the keyframe

        camera_extrinsics_init = np.array(
            [camera_extrinsics_prv[i] for i in range(len(camera_extrinsics_prv))]
        )
        return camera_extrinsics_init

    def _get_marker_extrinsics_initial_guess(
        self,
        camera_indices,
        marker_indices,
        markers_points_2d_detected,
        camera_extrinsics_prv,
        marker_extrinsics_prv,
    ):

        marker_extrinsics_init = marker_extrinsics_prv
        marker_keys_not_computed = set(marker_indices) - set(
            marker_extrinsics_init.keys()
        )

        for marker_idx in marker_keys_not_computed:
            camera_keys_available = list(
                set(camera_extrinsics_prv.keys())
                & set(camera_indices[marker_indices == marker_idx])
            )
            try:
                camera_idx0, camera_idx1 = random.sample(camera_keys_available, 2)
            except ValueError:
                return
            else:
                points_4d = self._run_triangulation(
                    markers_points_2d_detected,
                    camera_indices,
                    marker_indices,
                    camera_extrinsics_prv,
                    camera_idx0,
                    camera_idx1,
                    marker_idx,
                )
                marker_extrinsics_init[marker_idx] = self._convert_to_marker_extrinsics(
                    points_4d
                )

        marker_extrinsics_init = np.array(
            [marker_extrinsics_init[i] for i in range(len(marker_extrinsics_init))]
        )
        return marker_extrinsics_init

    def _run_triangulation(
        self,
        markers_points_2d_detected,
        camera_indices,
        marker_indices,
        camera_extrinsics,
        camera_idx0,
        camera_idx1,
        marker_idx,
    ):
        """ triangulate points """

        proj_mat1, proj_mat2, undistort_points1, undistort_points2 = self._prepare_data_for_triangulation(
            markers_points_2d_detected,
            camera_indices,
            marker_indices,
            camera_extrinsics,
            camera_idx0,
            camera_idx1,
            marker_idx,
        )

        points_4d = cv2.triangulatePoints(
            proj_mat1, proj_mat2, undistort_points1, undistort_points2
        )

        return points_4d

    def _prepare_data_for_triangulation(
        self,
        markers_points_2d_detected,
        camera_indices,
        marker_indices,
        camera_extrinsics,
        camera_idx0,
        camera_idx1,
        marker_idx,
    ):
        proj_mat1 = math.get_transform_mat(camera_extrinsics[camera_idx0])[:3, :4]
        proj_mat2 = math.get_transform_mat(camera_extrinsics[camera_idx1])[:3, :4]

        points1 = markers_points_2d_detected[
            np.bitwise_and(camera_indices == camera_idx0, marker_indices == marker_idx)
        ]

        points2 = markers_points_2d_detected[
            np.bitwise_and(camera_indices == camera_idx1, marker_indices == marker_idx)
        ]
        undistort_points1 = self.camera_model.undistortPoints(points1)
        undistort_points2 = self.camera_model.undistortPoints(points2)

        return proj_mat1, proj_mat2, undistort_points1, undistort_points2

    @staticmethod
    def _convert_to_marker_extrinsics(points_4d):
        marker_points_3d = cv2.convertPointsFromHomogeneous(points_4d.T).reshape(4, 3)
        marker_extrinsics = utils.point_3d_to_param(marker_points_3d)

        return marker_extrinsics
