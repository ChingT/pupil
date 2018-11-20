import marker_tracker_3d.math

from marker_tracker_3d.localization.camera_localizer import CameraLocalizer


class Controller:
    def __init__(self, storage):
        self.storage = storage
        self.camera_localizer = CameraLocalizer()

    def update_marker_extrinsics(self):
        self.camera_localizer.marker_extrinsics = self.storage.marker_extrinsics

    def update_camera_extrinsics(self):
        self.storage.camera_extrinsics = self.camera_localizer.current_camera(
            self.storage.markers, self.storage.previous_camera_extrinsics
        )
        if self.storage.camera_extrinsics is None:
            # Do not set previous_camera_extrinsics to None to ensure a decent initial
            # guess for the next solve_pnp call
            self.storage.camera_trace.append(None)
            self.storage.camera_trace_all.append(None)
        else:
            self.storage.previous_camera_extrinsics = self.storage.camera_extrinsics

            camera_pose_matrix = marker_tracker_3d.math.get_camera_pose_mat(
                self.storage.camera_extrinsics
            )
            self.storage.camera_trace.append(camera_pose_matrix[0:3, 3])
            self.storage.camera_trace_all.append(camera_pose_matrix[0:3, 3])
