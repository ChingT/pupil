import numpy as np

from marker_tracker_3d import utils


class CameraLocalizer:
    def __init__(
        self,
        camera_model,
        min_n_markers_per_frame_for_loc=1,
        max_camera_trace_distance=10.0,
    ):
        self.camera_model = camera_model
        self._min_n_markers_per_frame_for_loc = min_n_markers_per_frame_for_loc
        self.max_camera_trace_distance = max_camera_trace_distance

        self.previous_camera_extrinsics = None
        self.previous_camera_trace = np.full((3,), np.nan)

    def reset(self):
        self.previous_camera_extrinsics = None
        self.previous_camera_trace = np.full((3,), np.nan)

    def get_camera_extrinsics(self, markers, marker_extrinsics):
        marker_points_3d, marker_points_2d = self._prepare_data(
            markers, marker_extrinsics
        )

        if len(marker_points_3d) >= self._min_n_markers_per_frame_for_loc:
            current_camera_extrinsics = self._run_solvepnp(
                marker_points_2d, marker_points_3d
            )
        else:
            current_camera_extrinsics = None

        if self._check_camera_trace_distance(current_camera_extrinsics):
            # Do not set camera_extrinsics_previous to None
            # to ensure a decent initial guess for the next solvePnP call
            self.previous_camera_extrinsics = current_camera_extrinsics
            return current_camera_extrinsics
        else:
            return

    @staticmethod
    def _prepare_data(markers, marker_extrinsics):
        markers_id_available = markers.keys() & set(marker_extrinsics.keys())

        marker_points_3d = utils.params_to_points_3d(
            [marker_extrinsics[i] for i in markers_id_available]
        )
        marker_points_2d = np.array([markers[i]["verts"] for i in markers_id_available])
        return marker_points_3d, marker_points_2d

    def _run_solvepnp(self, marker_points_2d, marker_points_3d):
        try:
            rvec_prv, tvec_prv = utils.split_param(self.previous_camera_extrinsics)
        except AttributeError:
            retval, rvec, tvec = self.camera_model.solvePnP(
                marker_points_3d, marker_points_2d
            )
        else:
            retval, rvec, tvec = self.camera_model.solvePnP(
                marker_points_3d,
                marker_points_2d,
                useExtrinsicGuess=True,
                rvec=rvec_prv,
                tvec=tvec_prv,
            )

        if retval and utils.check_camera_extrinsics(marker_points_3d, rvec, tvec):
            return utils.merge_param(rvec, tvec)

    def _check_camera_trace_distance(self, current_camera_extrinsics):
        if current_camera_extrinsics is not None:
            current_camera_trace = utils.get_camera_trace_from_camera_extrinsics(
                current_camera_extrinsics
            )
            camera_trace_distance = utils.compute_camera_trace_distance(
                self.previous_camera_trace, current_camera_trace
            )
            if not camera_trace_distance > self.max_camera_trace_distance:
                self.previous_camera_trace = current_camera_trace
                return True

        self.previous_camera_trace = np.full((3,), np.nan)
        return False
