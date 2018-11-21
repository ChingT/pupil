import numpy as np

from marker_tracker_3d.camera_model import CameraModel
from marker_tracker_3d.utils import merge_param, check_camera_extrinsics


class CameraLocalization:
    def __init__(self):
        super().__init__()
        self.camera_model = CameraModel()

    def get_camera_extrinsics(
        self, markers, marker_extrinsics, camera_extrinsics_previous=None
    ):
        marker_points_3d, marker_points_2d = self._prepare_data(
            markers, marker_extrinsics
        )

        retval, rvec, tvec = self.camera_model.run_solvePnP(
            marker_points_3d, marker_points_2d, camera_extrinsics_previous
        )

        if retval:
            if check_camera_extrinsics(marker_points_3d, rvec, tvec):
                camera_extrinsics = merge_param(rvec, tvec)
                return camera_extrinsics

    def _prepare_data(self, current_frame, marker_extrinsics):
        marker_keys_available = current_frame.keys() & set(marker_extrinsics.keys())

        marker_points_3d = self.camera_model.params_to_points_3d(
            [marker_extrinsics[i] for i in marker_keys_available]
        )
        marker_points_2d = np.array(
            [current_frame[i]["verts"] for i in marker_keys_available]
        )

        if len(marker_points_3d) and len(marker_points_2d):
            marker_points_3d.shape = 1, -1, 3
            marker_points_2d.shape = 1, -1, 2

        return marker_points_3d, marker_points_2d
