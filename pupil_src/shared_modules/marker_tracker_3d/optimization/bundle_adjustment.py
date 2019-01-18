import collections

import numpy as np
from scipy import misc as scipy_misc
from scipy import optimize as scipy_optimize
from scipy import sparse as scipy_sparse

from marker_tracker_3d import utils

OptimizationResult = collections.namedtuple(
    "OptimizationResult",
    ["camera_extrinsics_opt", "marker_extrinsics_opt", "frame_ids_failed"],
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
        all_novel_markers,
        camera_extrinsics_init_dict,
        marker_extrinsics_init_dict,
    ):
        """ run bundle adjustment given the initial guess and then check the result of
        optimization
        """
        self._set_ids(camera_extrinsics_init_dict, marker_extrinsics_init_dict)

        if not self._check_enoug_data(all_novel_markers):
            return

        camera_extrinsics_init_array, marker_extrinsics_init_array = self._update_init_array(
            camera_extrinsics_init_dict, marker_extrinsics_init_dict
        )

        initial_guess, bounds, sparsity_matrix = self._prepare_parameters(
            camera_extrinsics_init_array, marker_extrinsics_init_array
        )
        least_sq_result = self._least_squares(bounds, initial_guess, sparsity_matrix)

        optimization_result = self._get_optimization_result(least_sq_result)
        return optimization_result

    def _set_ids(self, camera_extrinsics_init_dict, marker_extrinsics_init_dict):
        origin_marker_id = utils.find_origin_marker_id(marker_extrinsics_init_dict)
        self._marker_ids = [origin_marker_id] + list(
            set(marker_extrinsics_init_dict.keys()) - {origin_marker_id}
        )
        self._frame_ids = list(camera_extrinsics_init_dict.keys())

    def _check_enoug_data(self, all_novel_markers):
        frame_indices = []
        marker_indices = []
        markers_points_2d_detected = []

        for marker in all_novel_markers:
            if (
                marker.frame_id in self._frame_ids
                and marker.marker_id in self._marker_ids
            ):
                frame_indices.append(self._frame_ids.index(marker.frame_id))
                marker_indices.append(self._marker_ids.index(marker.marker_id))
                markers_points_2d_detected.append(marker.verts)

        if not markers_points_2d_detected:
            return False

        self._frame_indices = np.array(frame_indices)
        self._marker_indices = np.array(marker_indices)
        self._markers_points_2d_detected = np.array(markers_points_2d_detected)
        return True

    def _update_init_array(
        self, camera_extrinsics_init_dict, marker_extrinsics_init_dict
    ):
        camera_extrinsics_init_array = np.array(
            [camera_extrinsics_init_dict[frame_id] for frame_id in self._frame_ids]
        )
        marker_extrinsics_init_array = np.array(
            [marker_extrinsics_init_dict[marker_id] for marker_id in self._marker_ids]
        )

        self._camera_extrinsics_shape = camera_extrinsics_init_array.shape
        self._marker_extrinsics_shape = marker_extrinsics_init_array.shape

        return camera_extrinsics_init_array, marker_extrinsics_init_array

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
        camera_extrinsics_opt_array, marker_extrinsics_opt_array = self._reshape_variables_to_extrinsics(
            least_sq_result.x
        )

        frame_indices_failed, marker_indices_failed = self._find_failed_indices(
            least_sq_result.fun
        )

        camera_extrinsics_opt = {
            self._frame_ids[i]: camera_extrinsics
            for i, camera_extrinsics in enumerate(camera_extrinsics_opt_array)
            if i not in frame_indices_failed
        }
        marker_extrinsics_opt = {
            self._marker_ids[i]: marker_extrinsics
            for i, marker_extrinsics in enumerate(marker_extrinsics_opt_array)
            if i not in marker_indices_failed
        }
        frame_ids_failed = [self._frame_ids[i] for i in frame_indices_failed]

        optimization_result = OptimizationResult(
            camera_extrinsics_opt=camera_extrinsics_opt,
            marker_extrinsics_opt=marker_extrinsics_opt,
            frame_ids_failed=frame_ids_failed,
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
