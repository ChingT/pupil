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
        current_frame_id = frame.timestamp

        self.get_marker_id_to_detections(frame, current_frame_id)
        self.localize(
            self._controller_storage.marker_id_to_detections, current_frame_id
        )
        self._controller_storage.update_current_camera_pose(
            self._controller_storage.camera_extrinsics
        )

        if self._model_storage.optimize_3d_model:
            self.pick_key_markers(
                self._controller_storage.marker_id_to_detections, current_frame_id
            )

    def get_marker_id_to_detections(self, frame, current_frame_id):
        marker_id_to_detections = worker.detect_markers.detect(frame)

        self._controller_storage.update_current_marker_id_to_detections(
            marker_id_to_detections
        )

        self._controller_storage.save_all_marker_id_to_detections(
            marker_id_to_detections, current_frame_id
        )

    def localize(self, marker_id_to_detections, current_frame_id):
        camera_extrinsics = worker.localize_camera.localize(
            self._camera_intrinsics,
            marker_id_to_detections,
            self._model_storage.marker_id_to_extrinsics_opt,
            self._controller_storage.camera_extrinsics,
            self._min_n_markers_per_frame,
        )
        self._controller_storage.update_current_camera_extrinsics(camera_extrinsics)
        self._controller_storage.save_all_camera_extrinsics(
            camera_extrinsics, current_frame_id
        )

    def pick_key_markers(self, marker_id_to_detections, current_frame_id):
        if self._decide_key_markers.run(marker_id_to_detections):
            self._controller_storage.save_key_markers(
                marker_id_to_detections, current_frame_id
            )

    def reset(self):
        self._decide_key_markers.reset()
