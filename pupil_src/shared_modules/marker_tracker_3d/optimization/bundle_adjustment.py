import collections

import numpy as np
from scipy import misc as scipy_misc
from scipy import optimize as scipy_optimize
from scipy import sparse as scipy_sparse

from marker_tracker_3d import utils

OptimizationResult = collections.namedtuple(
    "OptimizationResult",
    [
        "camera_extrinsics_opt_array",
        "marker_extrinsics_opt_array",
        "frame_indices_failed",
        "marker_indices_failed",
    ],
)


# BundleAdjustment is a class instead of functions, since passing all the parameters
# would be inefficient.
# (especially true for _function_compute_residuals as a callback)
class BundleAdjustment:
    def __init__(self, camera_model):
        self._camera_model = camera_model

        self._tol = 1e-4
        self._diff_step = 1e-3

    def run(
        self,
        frame_indices,
        marker_indices,
        markers_points_2d_detected,
        camera_extrinsics_init_array,
        marker_extrinsics_init_array,
    ):
        """ run bundle adjustment given the initial guess and then check the result of
        optimization
        """

        if marker_extrinsics_init_array is None:
            return None

        self._update_data(
            frame_indices,
            marker_indices,
            markers_points_2d_detected,
            camera_extrinsics_init_array,
            marker_extrinsics_init_array,
        )

        initial_guess, bounds, sparsity_matrix = self._prepare_parameters(
            camera_extrinsics_init_array, marker_extrinsics_init_array
        )
        least_sq_result = self._least_squares(bounds, initial_guess, sparsity_matrix)

        optimization_result = self._get_optimization_result(least_sq_result)
        return optimization_result

    def _update_data(
        self,
        frame_indices,
        marker_indices,
        markers_points_2d_detected,
        camera_extrinsics_init_array,
        marker_extrinsics_init_array,
    ):
        self._frame_indices = frame_indices
        self._marker_indices = marker_indices
        self._markers_points_2d_detected = markers_points_2d_detected

        self._camera_extrinsics_shape = camera_extrinsics_init_array.shape
        self._marker_extrinsics_shape = marker_extrinsics_init_array.shape

    def _prepare_parameters(
        self, camera_extrinsics_init_array, marker_extrinsics_init_array
    ):
        initial_guess = np.vstack(
            (camera_extrinsics_init_array, marker_extrinsics_init_array)
        ).ravel()

        bounds = self._calculate_bounds()

        sparsity_matrix = self._construct_sparsity_matrix()

        return initial_guess, bounds, sparsity_matrix

    def _calculate_bounds(self, eps=1e-16):
        """ calculate the lower and upper bounds on independent variables
            fix the first marker at the origin of the coordinate system
        """

        camera_extrinsics_lower_bound = np.full(self._camera_extrinsics_shape, -np.inf)
        camera_extrinsics_upper_bound = np.full(self._camera_extrinsics_shape, np.inf)

        marker_extrinsics_lower_bound = np.full(self._marker_extrinsics_shape, -np.inf)
        marker_extrinsics_upper_bound = np.full(self._marker_extrinsics_shape, np.inf)
        marker_extrinsics_lower_bound[0] = utils.get_marker_extrinsics_origin() - eps
        marker_extrinsics_upper_bound[0] = utils.get_marker_extrinsics_origin() + eps

        lower_bound = np.vstack(
            (camera_extrinsics_lower_bound, marker_extrinsics_lower_bound)
        ).ravel()
        upper_bound = np.vstack(
            (camera_extrinsics_upper_bound, marker_extrinsics_upper_bound)
        ).ravel()

        return lower_bound, upper_bound

    def _construct_sparsity_matrix(self):
        """
        Construct the sparsity structure of the Jacobian matrix for finite difference
        estimation. If the Jacobian has only few non-zero elements in each row,
        providing the sparsity structure will greatly speed up the computations. A zero
        entry means that a corresponding element in the Jacobian is identically zero.

        :return: scipy.sparse.lil_matrix, with shape (n_residuals, n_variables), where
        n_residuals = markers_points_2d_detected.size,
        n_variables = camera_extrinsics.size + marker_extrinsics.size
        """

        n_samples = len(self._frame_indices)

        mat_camera = np.zeros((n_samples, self._camera_extrinsics_shape[0]), dtype=int)
        mat_camera[np.arange(n_samples), self._frame_indices] = 1
        mat_camera = scipy_misc.imresize(
            mat_camera,
            size=(
                self._markers_points_2d_detected.size,
                np.prod(self._camera_extrinsics_shape),
            ),
            interp="nearest",
        )

        mat_marker = np.zeros((n_samples, self._marker_extrinsics_shape[0]), dtype=int)
        mat_marker[np.arange(n_samples), self._marker_indices] = 1
        mat_marker = scipy_misc.imresize(
            mat_marker,
            size=(
                self._markers_points_2d_detected.size,
                np.prod(self._marker_extrinsics_shape),
            ),
            interp="nearest",
        )

        sparsity_matrix = np.hstack((mat_camera, mat_marker))
        sparsity_matrix = scipy_sparse.lil_matrix(sparsity_matrix)
        return sparsity_matrix

    def _least_squares(self, bounds, initial_guess, sparsity_matrix):
        result = scipy_optimize.least_squares(
            fun=self._function_compute_residuals,
            x0=initial_guess,
            bounds=bounds,
            ftol=self._tol,
            xtol=self._tol,
            gtol=self._tol,
            x_scale="jac",
            loss="soft_l1",
            diff_step=self._diff_step,
            jac_sparsity=sparsity_matrix,
        )
        return result

    def _get_optimization_result(self, least_sq_result):
        camera_extrinsics, marker_extrinsics = self._reshape_variables_to_extrinsics(
            least_sq_result.x
        )

        frame_indices_failed, marker_indices_failed = self._find_failed_indices(
            least_sq_result.fun
        )

        optimization_result = OptimizationResult(
            camera_extrinsics_opt_array=camera_extrinsics,
            marker_extrinsics_opt_array=marker_extrinsics,
            frame_indices_failed=frame_indices_failed,
            marker_indices_failed=marker_indices_failed,
        )
        return optimization_result

    def _function_compute_residuals(self, variables):
        """ Function which computes the vector of residuals,
        i.e., the minimization proceeds with respect to params
        """

        camera_extrinsics, marker_extrinsics = self._reshape_variables_to_extrinsics(
            variables
        )

        markers_points_2d_projected = self._project_markers(
            camera_extrinsics, marker_extrinsics
        )
        residuals = markers_points_2d_projected - self._markers_points_2d_detected
        return residuals.ravel()

    def _reshape_variables_to_extrinsics(self, variables):
        """ reshape 1-dimensional vector into the original shape of camera_extrinsics
        and marker_extrinsics
        """

        camera_extrinsics = variables[: np.prod(self._camera_extrinsics_shape)]
        camera_extrinsics.shape = self._camera_extrinsics_shape

        marker_extrinsics = variables[np.prod(self._camera_extrinsics_shape) :]
        marker_extrinsics.shape = self._marker_extrinsics_shape

        return camera_extrinsics, marker_extrinsics

    def _project_markers(self, camera_extrinsics, marker_extrinsics):
        markers_points_3d = utils.extrinsics_to_marker_points_3d(marker_extrinsics)

        markers_points_2d_projected = [
            self._camera_model.projectPoints(points, cam[0:3], cam[3:6])
            for cam, points in zip(
                camera_extrinsics[self._frame_indices],
                markers_points_3d[self._marker_indices],
            )
        ]
        markers_points_2d_projected = np.array(markers_points_2d_projected)
        return markers_points_2d_projected

    def _find_failed_indices(self, residuals, thres=20):
        """ find out those frames_id and markers_id which causes large reprojection
        errors
        """

        residuals.shape = -1, 4, 2
        reprojection_errors = np.linalg.norm(residuals, axis=2).sum(axis=1)
        frame_indices_failed = set(self._frame_indices[reprojection_errors > thres])
        marker_indices_failed = set(self._marker_indices[reprojection_errors > thres])
        return frame_indices_failed, marker_indices_failed
