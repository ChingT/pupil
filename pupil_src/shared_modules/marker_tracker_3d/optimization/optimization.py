import logging
import time

import cv2
import numpy as np
import scipy

from marker_tracker_3d import math
from marker_tracker_3d import utils

logger = logging.getLogger(__name__)


class Optimization:
    def __init__(self, camera_model):
        self.camera_model = camera_model

        self.tol = 1e-3
        self.diff_step = 1e-3

        self.n_camera_params = 6
        self.n_marker_params = 6

        self.camera_indices = None
        self.marker_indices = None
        self.markers_points_2d_detected = None
        self.camera_extrinsics_prv = None
        self.marker_extrinsics_prv = None

        self.n_cameras = 0
        self.n_markers = 0

    def update_params(
        self,
        camera_indices,
        marker_indices,
        markers_points_2d_detected,
        camera_extrinsics_prv,
        marker_extrinsics_prv,
    ):
        """
        :param camera_indices: array_like with shape (n, ), camera indices
        :param marker_indices: array_like with shape (n, ), marker indices
        :param markers_points_2d_detected: np.ndarray with shape (n x 4 x 2), markers points from image
        :param camera_extrinsics_prv: dict, previous camera extrinsics
        :param marker_extrinsics_prv: dict, previous marker extrinsics
        """
        self.camera_indices = camera_indices
        self.marker_indices = marker_indices
        self.markers_points_2d_detected = markers_points_2d_detected
        self.camera_extrinsics_prv = camera_extrinsics_prv
        self.marker_extrinsics_prv = marker_extrinsics_prv

        self.n_cameras = len(set(self.camera_indices))
        self.n_markers = len(
            set(self.marker_extrinsics_prv.keys()) | set(self.marker_indices)
        )

    def _reconstruction(self):
        """ reconstruct camera extrinsics and markers extrinsics iteratively
        the results are used as the initial guess for bundle adjustment
        """

        camera_extrinsics_init = self.camera_extrinsics_prv.copy()
        marker_extrinsics_init = self.marker_extrinsics_prv.copy()

        # no need to reconstruct cameras since we have got initial guess when picking the keyframe

        marker_index_not_computed = set(self.marker_indices) - set(
            marker_extrinsics_init.keys()
        )

        # reconstruct markers
        marker_extrinsics_init = self._reconstruct_markers(
            camera_extrinsics_init, marker_extrinsics_init, marker_index_not_computed
        )

        return camera_extrinsics_init, marker_extrinsics_init

    def _reconstruct_markers(
        self, camera_extrinsics_init, marker_extrinsics_init, marker_index_not_computed
    ):
        for marker_idx in marker_index_not_computed:
            camera_index_available = list(
                set(camera_extrinsics_init.keys())
                & set(self.camera_indices[self.marker_indices == marker_idx])
            )
            try:
                camera_idx0, camera_idx1 = np.random.choice(
                    camera_index_available, 2, replace=False
                )
            except ValueError:
                return marker_extrinsics_init
            else:
                points4D = self._run_triangulation(
                    camera_extrinsics_init, camera_idx0, camera_idx1, marker_idx
                )
                marker_extrinsics_init[marker_idx] = self._convert_to_marker_extrinsics(
                    points4D
                )

        return marker_extrinsics_init

    def _run_triangulation(
        self, camera_extrinsics, camera_idx0, camera_idx1, marker_idx
    ):
        """ triangulate points """

        proj_mat1, proj_mat2, undistort_points1, undistort_points2 = self._prepare_data_for_triangulation(
            camera_extrinsics, camera_idx0, camera_idx1, marker_idx
        )

        points4D = cv2.triangulatePoints(
            proj_mat1, proj_mat2, undistort_points1, undistort_points2
        )

        return points4D

    def _prepare_data_for_triangulation(
        self, camera_extrinsics, camera_idx0, camera_idx1, marker_idx
    ):
        proj_mat1 = math.get_transform_mat(camera_extrinsics[camera_idx0])[:3, :4]
        proj_mat2 = math.get_transform_mat(camera_extrinsics[camera_idx1])[:3, :4]

        points1 = self.markers_points_2d_detected[
            np.bitwise_and(
                self.camera_indices == camera_idx0, self.marker_indices == marker_idx
            )
        ]

        points2 = self.markers_points_2d_detected[
            np.bitwise_and(
                self.camera_indices == camera_idx1, self.marker_indices == marker_idx
            )
        ]
        undistort_points1 = self.camera_model.undistortPoints(points1)
        undistort_points2 = self.camera_model.undistortPoints(points2)

        return proj_mat1, proj_mat2, undistort_points1, undistort_points2

    def _convert_to_marker_extrinsics(self, points4D):
        marker_points_3d = cv2.convertPointsFromHomogeneous(points4D.T).reshape(4, 3)
        marker_extrinsics = utils.point_3d_to_param(marker_points_3d)

        return marker_extrinsics

    def _find_sparsity(self):
        """
        Defines the sparsity structure of the Jacobian matrix for finite difference estimation.
        If the Jacobian has only few non-zero elements in each row, providing the sparsity structure will greatly speed
        up the computations. A zero entry means that a corresponding element in the Jacobian is identically zero.
        """

        n_residuals = self.markers_points_2d_detected.size
        n_params = (
            self.n_camera_params * self.n_cameras
            + self.n_marker_params * self.n_markers
        )
        logger.debug(
            "n_cameras {0} n_markers {1} n_residuals {2} n_params {3}".format(
                self.n_cameras, self.n_markers, n_residuals, n_params
            )
        )

        sparsity_mat = scipy.sparse.lil_matrix((n_residuals, n_params), dtype=int)
        i = np.arange(self.camera_indices.size)

        for s in range(self.n_camera_params):
            for j in range(8):
                sparsity_mat[
                    8 * i + j, self.camera_indices * self.n_camera_params + s
                ] = 1

        for s in range(self.n_marker_params):
            for j in range(8):
                sparsity_mat[
                    8 * i + j,
                    self.n_cameras * self.n_camera_params
                    + self.marker_indices * self.n_marker_params
                    + s,
                ] = 1

        return sparsity_mat

    # Fix first marker
    def _cal_bounds(self, x, epsilon=1e-8):
        """ calculate the lower and upper bounds on independent variables """

        camera_params_size = self.n_cameras * self.n_camera_params
        lower_bound = np.full_like(x, -np.inf)
        lower_bound[camera_params_size : camera_params_size + self.n_marker_params] = (
            utils.marker_extrinsics_origin - epsilon
        )
        upper_bound = np.full_like(x, np.inf)
        upper_bound[camera_params_size : camera_params_size + self.n_marker_params] = (
            utils.marker_extrinsics_origin + epsilon
        )
        assert (
            (x > lower_bound)[
                camera_params_size : camera_params_size + self.n_marker_params
            ]
        ).all(), "lower_bound hit"
        assert (
            (x < upper_bound)[
                camera_params_size : camera_params_size + self.n_marker_params
            ]
        ).all(), "upper_bound hit"

        return lower_bound, upper_bound

    def _func(self, params):
        """
        Function which computes the vector of residuals, with the signature fun(x, *args, **kwargs),
        i.e., the minimization proceeds with respect to its first argument.
        The argument x passed to this function is an ndarray of shape (n,)
        """

        camera_extrinsics, marker_extrinsics = self._reshape_params(params)
        proj_error = self.cal_proj_error(
            camera_extrinsics,
            marker_extrinsics,
            self.camera_indices,
            self.marker_indices,
            self.markers_points_2d_detected,
        )
        return proj_error

    def cal_proj_error(
        self,
        camera_extrinsics,
        marker_extrinsics,
        camera_indices,
        marker_indices,
        markers_points_2d_detected,
    ):
        markers_points_3d = utils.params_to_points_3d(marker_extrinsics.reshape(-1, 6))
        markers_points_2d_projected = self._project_markers(
            camera_extrinsics[camera_indices], markers_points_3d[marker_indices]
        )
        diff = markers_points_2d_projected - markers_points_2d_detected
        return diff.ravel()

    def _project_markers(self, camera_extrinsics, markers_points_3d):
        camera_extrinsics = camera_extrinsics.reshape(-1, 6).copy()
        markers_points_3d = markers_points_3d.reshape(-1, 4, 3).copy()
        markers_points_2d_projected = [
            self.camera_model.projectPoints(points, cam[0:3], cam[3:6])
            for cam, points in zip(camera_extrinsics, markers_points_3d)
        ]
        markers_points_2d_projected = np.array(
            markers_points_2d_projected, dtype=np.float32
        )
        return markers_points_2d_projected

    def _reshape_params(self, params):
        """ reshape camera_extrinsics and marker_extrinsics into original shape"""

        camera_params_size = self.n_cameras * self.n_camera_params
        camera_extrinsics = params[:camera_params_size].reshape(
            self.n_cameras, self.n_camera_params
        )
        marker_extrinsics = params[camera_params_size:].reshape(
            self.n_markers, self.n_marker_params
        )
        return camera_extrinsics, marker_extrinsics

    def bundle_adjustment(
        self, camera_extrinsics_init, marker_extrinsics_init, verbose=False
    ):
        """ run bundle adjustment given the result of reconstruction """

        # initial guess
        camera_extrinsics_0 = np.array(
            [camera_extrinsics_init[i] for i in sorted(camera_extrinsics_init.keys())]
        )
        marker_extrinsics_0 = np.array(
            [marker_extrinsics_init[i] for i in sorted(marker_extrinsics_init.keys())]
        )

        x0 = np.hstack((camera_extrinsics_0.ravel(), marker_extrinsics_0.ravel()))

        bounds = self._cal_bounds(x0)
        A = self._find_sparsity()

        t0 = time.time()
        res = scipy.optimize.least_squares(
            self._func,
            x0,
            jac_sparsity=A,
            x_scale="jac",
            method="trf",
            bounds=bounds,
            diff_step=self.diff_step,
            ftol=self.tol,
            xtol=self.tol,
            gtol=self.tol,
            verbose=verbose,
        )
        t1 = time.time()
        logger.debug("bundle_adjustment took {0:.4f} seconds".format(t1 - t0))

        camera_extrinsics_opt, marker_extrinsics_opt = self._reshape_params(res.x)
        return camera_extrinsics_opt, marker_extrinsics_opt

    def run(self):
        """ run reconstruction and then bundle adjustment """

        # Reconstruction
        camera_extrinsics_init, marker_extrinsics_init = self._reconstruction()
        if len(marker_extrinsics_init) != self.n_markers:
            logger.debug("reconstruction failed")
            optimization_result = "failed"
            return optimization_result

        logger.debug("reconstruction done")

        # bundle adjustment
        camera_extrinsics_opt, marker_extrinsics_opt = self.bundle_adjustment(
            camera_extrinsics_init, marker_extrinsics_init
        )
        camera_index_failed, marker_index_failed = self._success_check(
            camera_extrinsics_opt,
            marker_extrinsics_opt,
            self.camera_indices,
            self.marker_indices,
            self.markers_points_2d_detected,
            12,
        )

        logger.debug("bundle adjustment done")
        optimization_result = {
            "camera_extrinsics_opt": camera_extrinsics_opt,
            "marker_extrinsics_opt": marker_extrinsics_opt,
            "camera_index_failed": camera_index_failed,
            "marker_index_failed": marker_index_failed,
        }
        return optimization_result

    def _success_check(
        self,
        camera_extrinsics,
        marker_extrinsics,
        camera_indices,
        marker_indices,
        markers_points_2d_detected,
        thres=5,
    ):
        """ check if the result of optimization is reasonable """

        camera_extrinsics = camera_extrinsics.reshape(-1, self.n_camera_params)
        marker_extrinsics = marker_extrinsics.reshape(-1, self.n_marker_params)
        markers_points_3d = utils.params_to_points_3d(marker_extrinsics)
        markers_points_2d_projected = self._project_markers(
            camera_extrinsics[camera_indices], markers_points_3d[marker_indices]
        )

        # check if the projected points are within reasonable range
        max_projected_points = np.max(np.abs(markers_points_2d_projected), axis=(1, 2))
        camera_index_failed = set(camera_indices[max_projected_points > 1e4])
        marker_index_failed = set(marker_indices[max_projected_points > 1e4])

        # check if the reprojection errors are small
        reprojection_errors = np.linalg.norm(
            (markers_points_2d_detected - markers_points_2d_projected), axis=2
        ).sum(axis=1)
        camera_index_failed |= set(camera_indices[reprojection_errors > thres])
        marker_index_failed |= set(marker_indices[reprojection_errors > thres])

        return camera_index_failed, marker_index_failed
