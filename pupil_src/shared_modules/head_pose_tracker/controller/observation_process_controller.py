"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

from head_pose_tracker import worker


class ObservationProcessController:
    def __init__(
        self,
        controller_storage,
        model_storage,
        camera_intrinsics,
        min_n_markers_per_frame=2,
    ):
        self._controller_storage = controller_storage
        self._model_storage = model_storage
        self._camera_intrinsics = camera_intrinsics

        self._decide_key_markers = worker.DecideKeyMarkers(controller_storage)

        self._min_n_markers_per_frame = min_n_markers_per_frame

    def run(self, frame):
        frame_id = frame.timestamp

        try:
            marker_id_to_detections = self._controller_storage.all_marker_id_to_detections[
                frame_id
            ]
        except KeyError:
            marker_id_to_detections = self.get_detections(frame)

        self._controller_storage.update_current_marker_id_to_detections(
            marker_id_to_detections
        )

        try:
            camera_extrinsics = self._controller_storage.all_camera_extrinsics[frame_id]
        except KeyError:
            camera_extrinsics = self.localize(marker_id_to_detections, frame_id)

        self._controller_storage.update_current_camera_pose(camera_extrinsics)

    def get_detections(self, frame):
        current_frame_id = frame.timestamp
        marker_id_to_detections = worker.detect_markers.detect(frame)
        self._controller_storage.save_all_marker_id_to_detections(
            marker_id_to_detections, current_frame_id
        )
        return marker_id_to_detections

    def localize(self, marker_id_to_detections, current_frame_id):
        camera_extrinsics = worker.localize_camera.localize(
            self._camera_intrinsics,
            marker_id_to_detections,
            self._model_storage.marker_id_to_extrinsics_opt,
            self._controller_storage.camera_extrinsics,
            self._min_n_markers_per_frame,
        )
        self._controller_storage.save_all_camera_extrinsics(
            camera_extrinsics, current_frame_id
        )
        return camera_extrinsics

    def pick_key_markers(self, marker_id_to_detections, current_frame_id):
        if self._decide_key_markers.run(marker_id_to_detections):
            self._controller_storage.save_key_markers(
                marker_id_to_detections, current_frame_id
            )

    def reset(self):
        self._decide_key_markers.reset()
