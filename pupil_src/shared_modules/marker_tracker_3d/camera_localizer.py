import numpy as np

from marker_tracker_3d import utils


class CameraLocalizer:
    def __init__(
        self,
        camera_model,
        min_n_markers_per_frame_for_loc=1,
        max_camera_trace_distance=10.0,
    ):
        self._camera_model = camera_model
        self._min_n_markers_per_frame_for_loc = min_n_markers_per_frame_for_loc
        self._max_camera_trace_distance = max_camera_trace_distance

        self._set_to_default_values()

    def _set_to_default_values(self):
        self._previous_camera_extrinsics = None
        self._previous_camera_trace = np.full((3,), np.nan)

    def reset(self):
        self._set_to_default_values()

    def get_current_camera_extrinsics(self, markers, marker_extrinsics):
        marker_points_3d, marker_points_2d = self._prepare_marker_points(
            markers, marker_extrinsics
        )

        # calculate current_camera_extrinsics only when the number of markers is
        # greater than or equal to self._min_n_markers_per_frame_for_loc
        if len(marker_points_3d) >= self._min_n_markers_per_frame_for_loc:
            current_camera_extrinsics = self._run_solvepnp(
                marker_points_2d, marker_points_3d
            )
        else:
            current_camera_extrinsics = None

        if current_camera_extrinsics is not None:
            current_camera_trace = utils.get_camera_trace_from_camera_extrinsics(
                current_camera_extrinsics
            )
        else:
            current_camera_trace = np.full((3,), np.nan)

        if self._check_camera_trace_distance(current_camera_trace):
            self._previous_camera_extrinsics = current_camera_extrinsics
            self._previous_camera_trace = current_camera_trace
            return current_camera_extrinsics
        else:
            # Do not set previous_camera_extrinsics to None to ensure
            # a decent initial guess for the next solvePnP call
            self._previous_camera_trace = np.full((3,), np.nan)
            return None

    @staticmethod
    def _prepare_marker_points(markers, marker_extrinsics):
        markers_id_available = markers.keys() & set(marker_extrinsics.keys())

        marker_points_3d = utils.extrinsics_to_marker_points_3d(
            [marker_extrinsics[i] for i in markers_id_available]
        )
        marker_points_2d = np.array([markers[i]["verts"] for i in markers_id_available])
        return marker_points_3d, marker_points_2d

    def _run_solvepnp(self, marker_points_2d, marker_points_3d):
        try:
            rotation_prv, translation_prv = utils.split_extrinsics(
                self._previous_camera_extrinsics
            )
        except AttributeError:
            retval, rotation, translation = self._camera_model.solvePnP(
                marker_points_3d, marker_points_2d
            )
        else:
            retval, rotation, translation = self._camera_model.solvePnP(
                marker_points_3d,
                marker_points_2d,
                useExtrinsicGuess=True,
                rvec=rotation_prv,
                tvec=translation_prv,
            )

        if retval and utils.check_solvepnp_output(
            marker_points_3d, rotation, translation
        ):
            camera_extrinsics = utils.merge_extrinsics(rotation, translation)
            return camera_extrinsics
        else:
            return None

    def _check_camera_trace_distance(self, current_camera_trace):
        camera_trace_distance = np.linalg.norm(
            current_camera_trace - self._previous_camera_trace
        )
        if camera_trace_distance > self._max_camera_trace_distance:
            return False
        else:
            return True
