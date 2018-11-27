import numpy as np

from marker_tracker_3d.optimization.bundle_adjustment import BundleAdjustment
from marker_tracker_3d.optimization.initial_guess import InitialGuess


class Optimization:
    def __init__(self, camera_model):
        self.initial_guess = InitialGuess(camera_model)
        self.bundle_adjustment = BundleAdjustment(camera_model)

    def run(
        self,
        camera_indices,
        marker_indices,
        markers_points_2d_detected,
        camera_extrinsics_prv,
        marker_extrinsics_prv,
    ):
        """
        calculate initial guess of marker and camera poses,
        run bundle adjustment,
        check the result of optimization

        :param camera_indices: array_like with shape (n, ), camera indices
        :param marker_indices: array_like with shape (n, ), marker indices
        :param markers_points_2d_detected: np.ndarray with shape (n x 4 x 2), markers points from image
        :param camera_extrinsics_prv: dict, previous camera extrinsics
        :param marker_extrinsics_prv: dict, previous marker extrinsics
        """

        camera_extrinsics_init, marker_extrinsics_init = self.initial_guess.get(
            camera_indices,
            marker_indices,
            markers_points_2d_detected,
            camera_extrinsics_prv,
            marker_extrinsics_prv,
        )

        if marker_extrinsics_init is None:
            return dict()

        camera_extrinsics_opt, marker_extrinsics_opt, residuals = self.bundle_adjustment.run(
            camera_indices,
            marker_indices,
            markers_points_2d_detected,
            camera_extrinsics_init,
            marker_extrinsics_init,
        )

        camera_keys_failed, marker_keys_failed = self._find_failed_keys(
            camera_indices, marker_indices, residuals
        )

        optimization_result = {
            "camera_extrinsics_opt": camera_extrinsics_opt,
            "marker_extrinsics_opt": marker_extrinsics_opt,
            "camera_keys_failed": camera_keys_failed,
            "marker_keys_failed": marker_keys_failed,
        }
        return optimization_result

    @staticmethod
    def _find_failed_keys(camera_indices, marker_indices, residuals, thres=6):
        """ find out those camera_keys and marker_keys which causes large reprojection errors """

        residuals.shape = -1, 4, 2
        reprojection_errors = np.linalg.norm(residuals, axis=2).sum(axis=1)
        camera_keys_failed = set(camera_indices[reprojection_errors > thres])
        marker_keys_failed = set(marker_indices[reprojection_errors > thres])
        return camera_keys_failed, marker_keys_failed
