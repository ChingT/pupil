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
        frame_indices,
        marker_indices,
        markers_points_2d_detected,
        camera_extrinsics_prv,
        marker_extrinsics_prv,
    ):
        """
        calculate initial guess of camera and marker poses

        :param frame_indices: array_like with shape (n, ), camera indices
        :param marker_indices: array_like with shape (n, ), marker indices
        :param markers_points_2d_detected: np.ndarray with shape (n x 4 x 2), markers points from image
        :param camera_extrinsics_prv: dict, previous camera extrinsics
        :param marker_extrinsics_prv: dict, previous marker extrinsics
        """

        camera_extrinsics_init = camera_extrinsics_prv
        marker_extrinsics_init = marker_extrinsics_prv

        n_frames = len(set(frame_indices))
        n_markers = len(set(marker_indices))

        for _ in range(2):
            camera_extrinsics_init = self._get_camera_extrinsics_initial_guess(
                frame_indices,
                marker_indices,
                markers_points_2d_detected,
                camera_extrinsics_init,
                marker_extrinsics_init,
            )
            marker_extrinsics_init = self._get_marker_extrinsics_initial_guess(
                frame_indices,
                marker_indices,
                markers_points_2d_detected,
                camera_extrinsics_init,
                marker_extrinsics_init,
            )

            try:
                camera_extrinsics_init_array = np.array(
                    [camera_extrinsics_init[i] for i in range(n_frames)]
                )
                marker_extrinsics_init_array = np.array(
                    [marker_extrinsics_init[i] for i in range(n_markers)]
                )
            except KeyError:
                continue
            else:
                return camera_extrinsics_init_array, marker_extrinsics_init_array

        return None, None

    def _get_camera_extrinsics_initial_guess(
        self,
        frame_indices,
        marker_indices,
        markers_points_2d_detected,
        camera_extrinsics_init,
        marker_extrinsics_init,
    ):
        frames_index_not_computed = set(frame_indices) - set(
            camera_extrinsics_init.keys()
        )

        for frame_index in frames_index_not_computed:
            markers_index_available = list(
                set(marker_extrinsics_init.keys())
                & set(marker_indices[frame_indices == frame_index])
            )
            try:
                marker_index = markers_index_available[0]
            except IndexError:
                continue

            marker_points_3d = utils.params_to_points_3d(
                marker_extrinsics_init[marker_index]
            )
            marker_points_2d = markers_points_2d_detected[
                np.bitwise_and(
                    frame_indices == frame_index, marker_indices == marker_index
                )
            ]

            retval, rvec, tvec = self.camera_model.solvePnP(
                marker_points_3d, marker_points_2d
            )
            if retval:
                if utils.check_camera_extrinsics(marker_points_3d, rvec, tvec):
                    camera_extrinsics_init[frame_index] = utils.merge_param(rvec, tvec)
        return camera_extrinsics_init

    def _get_marker_extrinsics_initial_guess(
        self,
        frame_indices,
        marker_indices,
        markers_points_2d_detected,
        camera_extrinsics_prv,
        marker_extrinsics_prv,
    ):

        marker_extrinsics_init = marker_extrinsics_prv
        markers_index_not_computed = set(marker_indices) - set(
            marker_extrinsics_init.keys()
        )

        for marker_index in markers_index_not_computed:
            frames_index_available = list(
                set(camera_extrinsics_prv.keys())
                & set(frame_indices[marker_indices == marker_index])
            )
            try:
                camera_index_0, camera_index_1 = random.sample(
                    frames_index_available, 2
                )
            except ValueError:
                continue
            else:
                data_for_run_triangulation = (
                    frame_indices,
                    marker_indices,
                    markers_points_2d_detected,
                    camera_extrinsics_prv,
                    camera_index_0,
                    camera_index_1,
                    marker_index,
                )
                marker_extrinsics_init[
                    marker_index
                ] = self._calculate_marker_extrinsics(data_for_run_triangulation)

        return marker_extrinsics_init

    def _calculate_marker_extrinsics(self, data_for_run_triangulation):

        data_for_cv2_triangulate_points = self._prepare_data_for_cv2_triangulate_points(
            *data_for_run_triangulation
        )

        points_4d = cv2.triangulatePoints(*data_for_cv2_triangulate_points)

        return self._convert_to_marker_extrinsics(points_4d)

    def _prepare_data_for_cv2_triangulate_points(
        self,
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
            np.bitwise_and(
                frame_indices == frame_index_0, marker_indices == marker_index
            )
        ]

        points2 = markers_points_2d_detected[
            np.bitwise_and(
                frame_indices == frame_index_1, marker_indices == marker_index
            )
        ]
        undistort_points1 = self.camera_model.undistortPoints(points1)
        undistort_points2 = self.camera_model.undistortPoints(points2)

        return proj_mat1, proj_mat2, undistort_points1, undistort_points2

    @staticmethod
    def _convert_to_marker_extrinsics(points_4d):
        marker_points_3d = cv2.convertPointsFromHomogeneous(points_4d.T).reshape(4, 3)
        marker_extrinsics = utils.point_3d_to_param(marker_points_3d)

        return marker_extrinsics
