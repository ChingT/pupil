from marker_tracker_3d import math
from marker_tracker_3d.localization import Localization


class CameraLocalizer:
    def __init__(self, storage):
        self.storage = storage

        self.localization = Localization(
            self.storage.camera_model, self.storage.marker_model
        )
        self.min_number_of_markers_per_frame_for_loc = 2

    def update(self):
        self.update_camera_extrinsics()
        self.update_camera_pose_matrix()
        self.update_camera_trace()

    def update_camera_extrinsics(self):
        if len(self.storage.markers) >= self.min_number_of_markers_per_frame_for_loc:
            self.storage.camera_extrinsics = self.localization.get_camera_extrinsics(
                self.storage.markers,
                self.storage.marker_extrinsics,
                self.storage.camera_extrinsics_previous,
            )
        else:
            self.storage.camera_extrinsics = None

        if self.storage.camera_extrinsics is not None:
            # Do not set camera_extrinsics_previous to None to ensure a decent initial guess for the next solve_pnp call
            self.storage.camera_extrinsics_previous = self.storage.camera_extrinsics

    def update_camera_pose_matrix(self):
        self.storage.camera_pose_matrix = math.get_camera_pose_mat(
            self.storage.camera_extrinsics
        )

    def update_camera_trace(self):
        camera_trace = math.get_camera_trace(self.storage.camera_pose_matrix)
        self.storage.camera_trace.append(camera_trace)
