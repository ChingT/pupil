import collections
import logging

import cv2
import numpy as np
import scipy

from marker_tracker_3d import utils

logger = logging.getLogger(__name__)

ResultOfOptimization = collections.namedtuple(
    "ResultOfOptimization",
    [
        "camera_extrinsics_opt",
        "marker_extrinsics_opt",
        "camera_keys_failed",
        "marker_keys_failed",
    ],
)


class BundleAdjustment:
    def __init__(self, camera_model):
        self.camera_model = camera_model

        self.tol = 1e-3
        self.diff_step = 1e-3

    def run(
        self,
        camera_indices,
        marker_indices,
        markers_points_2d_detected,
        camera_extrinsics_init,
        marker_extrinsics_init,
    ):
        """ run bundle adjustment given the initial guess and then check the result of optimization """

        if marker_extrinsics_init is None:
            return

        self._update_data(
            camera_indices,
            marker_indices,
            markers_points_2d_detected,
            camera_extrinsics_init,
            marker_extrinsics_init,
        )

        initial_guess, bounds, sparsity_matrix = self._prepare_parameters(
            camera_extrinsics_init, marker_extrinsics_init
        )

        optimization_result = self._get_optimization_result(
            initial_guess, bounds, sparsity_matrix
        )
        return optimization_result

    def _update_data(
        self,
        camera_indices,
        marker_indices,
        markers_points_2d_detected,
        camera_extrinsics_init,
        marker_extrinsics_init,
    ):
        self.camera_indices = camera_indices
        self.marker_indices = marker_indices
        self.markers_points_2d_detected = markers_points_2d_detected

        self.camera_extrinsics_shape = camera_extrinsics_init.shape
        self.marker_extrinsics_shape = marker_extrinsics_init.shape

    def _prepare_parameters(self, camera_extrinsics_init, marker_extrinsics_init):
        initial_guess = np.vstack(
            (camera_extrinsics_init, marker_extrinsics_init)
        ).ravel()

        bounds = self._cal_bounds()

        sparsity_matrix = self._construct_sparsity_matrix()

        return initial_guess, bounds, sparsity_matrix

    def _cal_bounds(self, epsilon=1e-8):
        """ calculate the lower and upper bounds on independent variables
            fix the first marker at the origin of the coordinate system
        """

        camera_extrinsics_lower_bound = np.full(self.camera_extrinsics_shape, -np.inf)
        camera_extrinsics_upper_bound = np.full(self.camera_extrinsics_shape, np.inf)

        marker_extrinsics_lower_bound = np.full(self.marker_extrinsics_shape, -np.inf)
        marker_extrinsics_upper_bound = np.full(self.marker_extrinsics_shape, np.inf)
        marker_extrinsics_lower_bound[0] = utils.marker_extrinsics_origin - epsilon
        marker_extrinsics_upper_bound[0] = utils.marker_extrinsics_origin + epsilon

        lower_bound = np.vstack(
            (camera_extrinsics_lower_bound, marker_extrinsics_lower_bound)
        ).ravel()
        upper_bound = np.vstack(
            (camera_extrinsics_upper_bound, marker_extrinsics_upper_bound)
        ).ravel()

        return lower_bound, upper_bound

    def _construct_sparsity_matrix(self):
        """
        Construct the sparsity structure of the Jacobian matrix for finite difference estimation.
        If the Jacobian has only few non-zero elements in each row, providing the sparsity structure will greatly speed
        up the computations. A zero entry means that a corresponding element in the Jacobian is identically zero.

        :return: scipy.sparse.lil_matrix, with shape (n_residuals, n_variables), where
        n_residuals = markers_points_2d_detected.size,
        n_variables = camera_extrinsics.size + marker_extrinsics.size
        """

        n_samples = len(self.camera_indices)
        n_residual_per_sample = self.markers_points_2d_detected.size // n_samples

        mat_camera = np.zeros((n_samples, self.camera_extrinsics_shape[0]), dtype=int)
        mat_camera[np.arange(n_samples), self.camera_indices] = 1
        mat_camera = cv2.resize(
            mat_camera,
            fx=self.camera_extrinsics_shape[1],
            fy=n_residual_per_sample,
            dsize=(0, 0),
            interpolation=cv2.INTER_NEAREST,
        )

        mat_marker = np.zeros((n_samples, self.marker_extrinsics_shape[0]), dtype=int)
        mat_marker[np.arange(n_samples), self.marker_indices] = 1
        mat_marker = cv2.resize(
            mat_marker,
            fx=self.marker_extrinsics_shape[1],
            fy=n_residual_per_sample,
            dsize=(0, 0),
            interpolation=cv2.INTER_NEAREST,
        )

        sparsity_matrix = scipy.sparse.lil_matrix(np.hstack((mat_camera, mat_marker)))

        return sparsity_matrix

    def _get_optimization_result(self, initial_guess, bounds, sparsity_matrix):
        result = scipy.optimize.least_squares(
            fun=self._fun_compute_residuals,
            x0=initial_guess,
            bounds=bounds,
            ftol=self.tol,
            xtol=self.tol,
            gtol=self.tol,
            x_scale="jac",
            diff_step=self.diff_step,
            jac_sparsity=sparsity_matrix,
        )

        camera_extrinsics_opt, marker_extrinsics_opt = self._reshape_variables_to_extrinsics(
            result.x
        )

        camera_keys_failed, marker_keys_failed = self._find_failed_keys(result.fun)

        optimization_result = ResultOfOptimization(
            camera_extrinsics_opt=camera_extrinsics_opt,
            marker_extrinsics_opt=marker_extrinsics_opt,
            camera_keys_failed=camera_keys_failed,
            marker_keys_failed=marker_keys_failed,
        )
        return optimization_result

    def _fun_compute_residuals(self, variables):
        """ Function which computes the vector of residuals,
        i.e., the minimization proceeds with respect to params
        """

        camera_extrinsics, marker_extrinsics = self._reshape_variables_to_extrinsics(
            variables
        )

        markers_points_2d_projected = self._project_markers(
            camera_extrinsics, marker_extrinsics
        )
        residuals = markers_points_2d_projected - self.markers_points_2d_detected
        return residuals.ravel()

    def _reshape_variables_to_extrinsics(self, variables):
        """ reshape 1-dimensional vector into the original shape of camera_extrinsics and marker_extrinsics """

        camera_extrinsics = variables[: np.prod(self.camera_extrinsics_shape)]
        camera_extrinsics.shape = self.camera_extrinsics_shape

        marker_extrinsics = variables[np.prod(self.camera_extrinsics_shape) :]
        marker_extrinsics.shape = self.marker_extrinsics_shape

        return camera_extrinsics, marker_extrinsics

    def _project_markers(self, camera_extrinsics, marker_extrinsics):
        markers_points_3d = utils.params_to_points_3d(marker_extrinsics)

        markers_points_2d_projected = [
            self.camera_model.projectPoints(points, cam[0:3], cam[3:6])
            for cam, points in zip(
                camera_extrinsics[self.camera_indices],
                markers_points_3d[self.marker_indices],
            )
        ]
        markers_points_2d_projected = np.array(markers_points_2d_projected)
        return markers_points_2d_projected

    def _find_failed_keys(self, residuals, thres=6):
        """ find out those camera_keys and marker_keys which causes large reprojection errors """

        residuals.shape = -1, 4, 2
        reprojection_errors = np.linalg.norm(residuals, axis=2).sum(axis=1)
        camera_keys_failed = set(self.camera_indices[reprojection_errors > thres])
        marker_keys_failed = set(self.marker_indices[reprojection_errors > thres])
        return camera_keys_failed, marker_keys_failed
