from marker_tracker_3d.localization import Localization


class CameraLocalizer:
    def __init__(self, storage):
        self.storage = storage
        self.localization = Localization(self.storage)
        self.min_number_of_markers_per_frame_for_loc = 2

    def update(self, markers, marker_extrinsics, camera_extrinsics_previous):
        if len(markers) >= self.min_number_of_markers_per_frame_for_loc:
            camera_extrinsics = self.localization.get_camera_extrinsics(
                markers, marker_extrinsics, camera_extrinsics_previous
            )
            return camera_extrinsics
