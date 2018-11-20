import numpy as np

import marker_tracker_3d.math
from marker_tracker_3d.camera_model import CameraModel
from marker_tracker_3d.utils import merge_param, check_camera_extrinsics


class CameraLocalizer:
    def __init__(self, storage):
        super().__init__()
        self.storage = storage
        self.localization = Localization()

    def update_marker_extrinsics(self):
        self.localization.marker_extrinsics = self.storage.marker_extrinsics

    def update_camera_extrinsics(self):
        self.storage.camera_extrinsics = self.localization.get_camera_extrinsics(
            self.storage.markers, self.storage.camera_extrinsics_previous
        )
        if self.storage.camera_extrinsics is None:
            # Do not set camera_extrinsics_previous to None to ensure a decent initial guess for the next solve_pnp call
            self.storage.camera_trace.append(None)
            self.storage.camera_trace_all.append(None)
        else:
            self.storage.camera_extrinsics_previous = self.storage.camera_extrinsics

            camera_pose_matrix = marker_tracker_3d.math.get_camera_pose_mat(
                self.storage.camera_extrinsics
            )
            self.storage.camera_trace.append(camera_pose_matrix[0:3, 3])
            self.storage.camera_trace_all.append(camera_pose_matrix[0:3, 3])


class Localization(CameraModel):
    def __init__(self, marker_extrinsics=None):
        super().__init__()

        if marker_extrinsics is None:
            self.marker_extrinsics = {}
        else:
            self.marker_extrinsics = marker_extrinsics

    @property
    def marker_extrinsics(self):
        return self.__marker_extrinsics

    @marker_extrinsics.setter
    def marker_extrinsics(self, marker_extrinsics_new):
        assert isinstance(marker_extrinsics_new, dict)
        self.__marker_extrinsics = marker_extrinsics_new

    def get_camera_extrinsics(self, markers, camera_extrinsics_previous=None):
        marker_points_3d_for_loc, marker_points_2d_for_loc = self._prepare_data(markers)

        retval, rvec, tvec = self.run_solvePnP(
            marker_points_3d_for_loc,
            marker_points_2d_for_loc,
            camera_extrinsics_previous,
        )

        if retval:
            if check_camera_extrinsics(marker_points_3d_for_loc, rvec, tvec):
                camera_extrinsics = merge_param(rvec, tvec)
                return camera_extrinsics

    def _prepare_data(self, current_frame):
        marker_keys_available = current_frame.keys() & set(
            self.marker_extrinsics.keys()
        )

        marker_points_3d_for_loc = self.params_to_points_3d(
            [self.marker_extrinsics[i] for i in marker_keys_available]
        )
        marker_points_2d_for_loc = np.array(
            [current_frame[i]["verts"] for i in marker_keys_available]
        )

        if len(marker_points_3d_for_loc) and len(marker_points_2d_for_loc):
            marker_points_3d_for_loc.shape = 1, -1, 3
            marker_points_2d_for_loc.shape = 1, -1, 2

        return marker_points_3d_for_loc, marker_points_2d_for_loc
