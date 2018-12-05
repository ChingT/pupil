import numpy as np

from marker_tracker_3d import utils


class CameraLocalizer:
    def __init__(self, camera_model, min_number_of_markers_per_frame_for_loc=2):
        self.camera_model = camera_model
        self.min_number_of_markers_per_frame_for_loc = (
            min_number_of_markers_per_frame_for_loc
        )
        self.camera_extrinsics_previous = None

    def get_camera_extrinsics(self, markers, marker_extrinsics):
        camera_extrinsics = None

        if len(markers) >= self.min_number_of_markers_per_frame_for_loc:
            marker_points_3d, marker_points_2d = self._prepare_data(
                markers, marker_extrinsics
            )
            if self.camera_extrinsics_previous is not None:
                rvec, tvec = utils.split_param(self.camera_extrinsics_previous)

                retval, rvec, tvec = self.camera_model.solvePnP(
                    marker_points_3d,
                    marker_points_2d,
                    useExtrinsicGuess=True,
                    rvec=rvec.copy(),
                    tvec=tvec.copy(),
                )
            else:
                retval, rvec, tvec = self.camera_model.solvePnP(
                    marker_points_3d, marker_points_2d
                )

            if retval:
                if utils.check_camera_extrinsics(marker_points_3d, rvec, tvec):
                    camera_extrinsics = utils.merge_param(rvec, tvec)
                    # Do not set camera_extrinsics_previous to None to ensure a decent initial guess
                    # for the next solve_pnp call
                    self.camera_extrinsics_previous = camera_extrinsics

        return camera_extrinsics

    @staticmethod
    def _prepare_data(markers, marker_extrinsics):
        marker_keys_available = markers.keys() & set(marker_extrinsics.keys())

        marker_points_3d = utils.params_to_points_3d(
            [marker_extrinsics[i] for i in marker_keys_available]
        )
        marker_points_2d = np.array(
            [markers[i]["verts"] for i in marker_keys_available]
        )

        if len(marker_points_3d) and len(marker_points_2d):
            marker_points_3d.shape = 1, -1, 3
            marker_points_2d.shape = 1, -1, 2

        return marker_points_3d, marker_points_2d
