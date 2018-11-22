import numpy as np

from marker_tracker_3d import utils


class Localization:
    def __init__(self, storage):
        self.storage = storage

    def get_camera_extrinsics(
        self, markers, marker_extrinsics, camera_extrinsics_previous=None
    ):
        marker_points_3d, marker_points_2d = self._prepare_data(
            markers, marker_extrinsics
        )
        if camera_extrinsics_previous is not None:
            rvec, tvec = utils.split_param(camera_extrinsics_previous)

            retval, rvec, tvec = self.storage.camera_model.solvePnP(
                marker_points_3d,
                marker_points_2d,
                useExtrinsicGuess=True,
                rvec=rvec.copy(),
                tvec=tvec.copy(),
            )
        else:
            retval, rvec, tvec = self.storage.camera_model.solvePnP(
                marker_points_3d, marker_points_2d
            )

        if retval:
            if utils.check_camera_extrinsics(marker_points_3d, rvec, tvec):
                camera_extrinsics = utils.merge_param(rvec, tvec)
                return camera_extrinsics

    def _prepare_data(self, markers, marker_extrinsics):
        marker_keys_available = markers.keys() & set(marker_extrinsics.keys())

        marker_points_3d = self.storage.marker_model.params_to_points_3d(
            [marker_extrinsics[i] for i in marker_keys_available]
        )
        marker_points_2d = np.array(
            [markers[i]["verts"] for i in marker_keys_available]
        )

        if len(marker_points_3d) and len(marker_points_2d):
            marker_points_3d.shape = 1, -1, 3
            marker_points_2d.shape = 1, -1, 2

        return marker_points_3d, marker_points_2d
