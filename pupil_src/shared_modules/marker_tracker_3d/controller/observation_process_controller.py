from marker_tracker_3d import worker


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
        marker_id_to_detections = worker.detect_markers.detect(frame)
        camera_extrinsics = self.localize(marker_id_to_detections)

        current_frame_id = frame.timestamp
        self._controller_storage.save_observation(
            marker_id_to_detections, camera_extrinsics, current_frame_id
        )

        if self._model_storage.optimize_3d_model and self._decide_key_markers.run(
            self._controller_storage.marker_id_to_detections
        ):
            self._controller_storage.save_key_markers(
                self._controller_storage.marker_id_to_detections, current_frame_id
            )

    def localize(self, marker_id_to_detections):
        return worker.localize_camera.localize(
            self._camera_intrinsics,
            marker_id_to_detections,
            self._model_storage.marker_id_to_extrinsics_opt,
            self._controller_storage.camera_extrinsics,
            self._min_n_markers_per_frame,
        )

    def reset(self):
        self._decide_key_markers.reset()
