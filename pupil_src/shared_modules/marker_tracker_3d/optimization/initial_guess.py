import random

import cv2
import numpy as np

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

        camera_extrinsics_init = camera_extrinsics_prv
        marker_extrinsics_init = marker_extrinsics_prv

        for ii in range(5):
            camera_extrinsics_init = self._get_camera_extrinsics_initial_guess(
                camera_indices,
                marker_indices,
                markers_points_2d_detected,
                camera_extrinsics_init,
                marker_extrinsics_init,
            )
            marker_extrinsics_init = self._get_marker_extrinsics_initial_guess(
                camera_indices,
                marker_indices,
                markers_points_2d_detected,
                camera_extrinsics_init,
                marker_extrinsics_init,
            )

            try:
                camera_extrinsics_init_array = np.array(
                    [camera_extrinsics_init[i] for i in range(len(set(camera_indices)))]
                )
                marker_extrinsics_init_array = np.array(
                    [marker_extrinsics_init[i] for i in range(len(set(marker_indices)))]
                )
            except KeyError:
                continue
            else:
                return camera_extrinsics_init_array, marker_extrinsics_init_array

        return None, None

    def _get_camera_extrinsics_initial_guess(
        self,
        camera_indices,
        marker_indices,
        markers_points_2d_detected,
        camera_extrinsics_init,
        marker_extrinsics_init,
    ):
        camera_keys_not_computed = set(camera_indices) - set(
            camera_extrinsics_init.keys()
        )

        for camera_idx in camera_keys_not_computed:
            marker_keys_available = list(
                set(marker_extrinsics_init.keys())
                & set(marker_indices[camera_indices == camera_idx])
            )
            try:
                marker_idx = marker_keys_available[0]
            except IndexError:
                continue

            marker_points_3d = utils.params_to_points_3d(
                marker_extrinsics_init[marker_idx]
            )
            marker_points_2d = markers_points_2d_detected[
                np.bitwise_and(
                    camera_indices == camera_idx, marker_indices == marker_idx
                )
            ]

            retval, rvec, tvec = self.camera_model.solvePnP(
                marker_points_3d, marker_points_2d
            )
            if retval:
                if utils.check_camera_extrinsics(marker_points_3d, rvec, tvec):
                    camera_extrinsics_init[camera_idx] = utils.merge_param(rvec, tvec)
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
                continue
            else:
                data_for_run_triangulation = (
                    camera_indices,
                    marker_indices,
                    markers_points_2d_detected,
                    camera_extrinsics_prv,
                    camera_idx0,
                    camera_idx1,
                    marker_idx,
                )
                marker_extrinsics_init[marker_idx] = self._calculate_marker_extrinsics(
                    data_for_run_triangulation
                )

        return marker_extrinsics_init

    def _calculate_marker_extrinsics(self, data_for_run_triangulation):

        data_for_cv2_triangulate_points = self._prepare_data_for_cv2_triangulate_points(
            *data_for_run_triangulation
        )

        points_4d = cv2.triangulatePoints(*data_for_cv2_triangulate_points)

        return self._convert_to_marker_extrinsics(points_4d)

    def _prepare_data_for_cv2_triangulate_points(
        self,
        camera_indices,
        marker_indices,
        markers_points_2d_detected,
        camera_extrinsics,
        camera_idx0,
        camera_idx1,
        marker_idx,
    ):
        proj_mat1 = utils.get_extrinsic_matrix(camera_extrinsics[camera_idx0])[:3, :4]
        proj_mat2 = utils.get_extrinsic_matrix(camera_extrinsics[camera_idx1])[:3, :4]

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
